import pandas as pd
import numpy as np
import requests
import json
import time
import os

BASE = r'C:\Users\danus\GreenScore'
os.makedirs(f'{BASE}/01_data/climate', exist_ok=True)

# ── 1. LOAD & SAMPLE ──────────────────────────────────────────
print("Loading cleaned dataset...")
df = pd.read_csv(f'{BASE}/01_data/processed/loans_cleaned.csv')

df_sample = df.groupby('default', group_keys=False).apply(
    lambda x: x.sample(frac=0.022, random_state=42)
).reset_index(drop=True)

print(f"Sample size: {len(df_sample):,} rows")
print(f"Default rate preserved: {df_sample['default'].mean():.2%}")

# ── 2. NASA POWER API ─────────────────────────────────────────
def get_nasa_climate(lat, lon):
    try:
        r = requests.get(
            'https://power.larc.nasa.gov/api/temporal/monthly/point',
            params={
                'parameters': 'T2M,PRECTOTCORR',
                'community':  'RE',
                'longitude':  round(lon, 1),
                'latitude':   round(lat, 1),
                'start':      2010,
                'end':        2023,
                'format':     'JSON'
            },
            timeout=30
        )
        if r.status_code == 200:
            return r.json()['properties']['parameter']
    except Exception as e:
        print(f"  API error ({lat},{lon}): {e}")
    return None

def engineer_physical_features(data, lat, lon):
    """Convert raw NASA data into 5 risk features."""

    # DEBUG: print what we actually received
    if data:
        t_vals = list(data.get('T2M', {}).values())
        p_vals = list(data.get('PRECTOTCORR', {}).values())
        print(f"    DEBUG ({lat},{lon}): T2M samples={len(t_vals)}, "
              f"T2M[0]={t_vals[0] if t_vals else 'EMPTY'}, "
              f"PRECIP[0]={p_vals[0] if p_vals else 'EMPTY'}")
    else:
        print(f"    DEBUG ({lat},{lon}): data is None — using fallback")
        return {
            'flood_freq_score':    0.3,
            'drought_idx':         0.3,
            'temp_anomaly':        0.0,
            'extreme_events':      2,
            'physical_risk_score': 0.3
        }

    temp   = [v for v in data.get('T2M', {}).values()         if v not in (-999, -999.0)]
    precip = [v for v in data.get('PRECTOTCORR', {}).values() if v not in (-999, -999.0)]

    if len(temp) < 12 or len(precip) < 12:
        print(f"    WARNING: insufficient data temp={len(temp)}, precip={len(precip)}")
        return {
            'flood_freq_score':    0.3,
            'drought_idx':         0.3,
            'temp_anomaly':        0.0,
            'extreme_events':      2,
            'physical_risk_score': 0.3
        }

    # Flood frequency — months above 95th percentile precipitation
    p95        = np.percentile(precip, 95)
    flood_freq = sum(1 for p in precip if p > p95) / len(precip)

    # Drought index
    mean_p  = np.mean(precip)
    std_p   = np.std(precip)
    drought = max(0, (mean_p - np.min(precip)) / (mean_p + std_p + 1e-6))
    drought = min(drought, 1.0)

    # Temperature anomaly — recent 3 years vs full history
    if len(temp) > 36:
        temp_anom = np.mean(temp[-36:]) - np.mean(temp[:-36])
    else:
        temp_anom = 0.0

    # Extreme events — both heat and flood extremes
    t95            = np.percentile(temp, 95)
    extreme_heat   = sum(1 for t in temp   if t > t95)
    extreme_flood  = sum(1 for p in precip if p > p95)
    extreme_events = extreme_heat + extreme_flood

    # Composite physical risk score (0-1)
    phys = (
        0.35 * min(flood_freq, 1.0) +
        0.30 * drought +
        0.20 * min(max(temp_anom / 3.0, 0), 1.0) +
        0.15 * min(extreme_events / 20.0, 1.0)
    )

    return {
        'flood_freq_score':    round(flood_freq, 4),
        'drought_idx':         round(drought, 4),
        'temp_anomaly':        round(temp_anom, 4),
        'extreme_events':      int(extreme_events),
        'physical_risk_score': round(phys, 4)
    }

# ── 3. FETCH CLIMATE DATA FOR UNIQUE LOCATIONS ────────────────
df_sample['lat_r'] = df_sample['lat'].round(1)
df_sample['lon_r'] = df_sample['lon'].round(1)

unique_locs = df_sample[['lat_r','lon_r']].drop_duplicates().values
print(f"\nUnique locations to fetch: {len(unique_locs)}")

# Use simple tuple string as key
climate_cache = {}
failed = 0

for i, (lat, lon) in enumerate(unique_locs):
    key = f"{lat}_{lon}"    # ← clean key: "42.2_-74.9"
    print(f"  [{i+1}/{len(unique_locs)}] lat={lat}, lon={lon}", end=" ... ")
    data     = get_nasa_climate(lat, lon)
    features = engineer_physical_features(data, lat, lon)
    climate_cache[key] = features
    print(f"✓  phys_risk={features['physical_risk_score']}")
    if not data:
        failed += 1
    time.sleep(0.4)

print(f"\nFetch complete. Success: {len(unique_locs)-failed} | Fallback: {failed}")

# Save cache
with open(f'{BASE}/01_data/climate/nasa_cache.json', 'w') as f:
    json.dump(climate_cache, f, indent=2)
print("Cache saved.")

# ── 4. MAP CLIMATE FEATURES ONTO SAMPLE ───────────────────────
def lookup_climate(row):
    key = f"{row['lat_r']}_{row['lon_r']}"    # ← same key format
    return climate_cache.get(key, {
        'flood_freq_score':    0.3,
        'drought_idx':         0.3,
        'temp_anomaly':        0.0,
        'extreme_events':      2,
        'physical_risk_score': 0.3
    })

print("\nMapping climate features onto loan records...")
climate_df = df_sample.apply(lookup_climate, axis=1, result_type='expand')
df_sample  = pd.concat([df_sample, climate_df], axis=1)
df_sample.drop(columns=['lat_r','lon_r'], inplace=True)

# ── 5. TRANSITION RISK FEATURES ───────────────────────────────
sector_carbon_intensity = {
    'Agriculture':   45.2,
    'Manufacturing': 89.7,
    'Real_Estate':   12.3,
    'Services':       8.1,
    'Energy':       210.5,
    'Transport':     67.4,
    'Construction':  55.8,
    'Retail':         6.2,
}

rbi_npa_rates = {
    'Agriculture':   0.095,
    'Manufacturing': 0.052,
    'Real_Estate':   0.068,
    'Services':      0.032,
    'Energy':        0.078,
    'Transport':     0.041,
    'Construction':  0.061,
    'Retail':        0.019,
}

CARBON_PRICE_DISORDERLY = 290

def compute_transition_features(sector):
    intensity  = sector_carbon_intensity.get(sector, 20.0)
    npa_rate   = rbi_npa_rates.get(sector, 0.04)
    burden     = (intensity * CARBON_PRICE_DISORDERLY) / 1_000_000
    profit_hit = min(burden / 0.08, 1.0)
    score = (
        0.50 * profit_hit +
        0.30 * min(intensity / 250.0, 1.0) +
        0.20 * min(npa_rate * 5, 1.0)
    )
    return {
        'carbon_intensity':      round(intensity, 2),
        'carbon_burden':         round(burden, 6),
        'transition_risk_score': round(min(score, 1.0), 4)
    }

print("Computing transition risk features...")
trans_df  = df_sample['sector'].apply(compute_transition_features)
trans_df  = pd.DataFrame(list(trans_df))
df_sample = pd.concat([df_sample, trans_df], axis=1)

# ── 6. FINAL SUMMARY & SAVE ───────────────────────────────────
print(f"\nFinal shape: {df_sample.shape}")

print("\nPhysical risk score stats:")
print(df_sample['physical_risk_score'].describe().round(4))

print("\nTransition risk score stats:")
print(df_sample['transition_risk_score'].describe().round(4))

print("\nSample — climate features by location:")
print(df_sample[['lat','lon','flood_freq_score','drought_idx',
                 'temp_anomaly','physical_risk_score']]
      .drop_duplicates(subset=['lat','lon'])
      .sort_values('physical_risk_score', ascending=False)
      .head(10).to_string())

out_path = f'{BASE}/01_data/processed/features_complete.csv'
df_sample.to_csv(out_path, index=False)
print(f"\n✅ Saved → {out_path}")