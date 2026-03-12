import pandas as pd
import folium
from folium.plugins import HeatMap
import os

BASE = r'C:\Users\danus\GreenScore'
os.makedirs(f'{BASE}/04_outputs', exist_ok=True)

print("Loading portfolio CPD...")
df = pd.read_csv(f'{BASE}/04_outputs/portfolio_cpd.csv')
print(f"Loans: {len(df):,}")

# ── BUILD FOLIUM HEATMAP ──────────────────────────────────────
m = folium.Map(
    location=[39.5, -98.4],   # center of US
    zoom_start=4,
    tiles='CartoDB positron'
)

# HeatMap layer — weighted by Disorderly CPD
heat_data = [[row['lat'], row['lon'], row['cpd_Disorderly']]
             for _, row in df.iterrows()]

HeatMap(
    heat_data,
    min_opacity = 0.3,
    max_val     = 0.8,
    radius      = 18,
    blur        = 12,
    gradient    = {
        '0.2': 'blue',
        '0.4': 'cyan',
        '0.6': 'lime',
        '0.75': 'yellow',
        '1.0': 'red'
    }
).add_to(m)

# Top-20 highest CPD loans as markers
top20 = df.nlargest(20, 'cpd_Disorderly')

for _, row in top20.iterrows():
    folium.CircleMarker(
        location  = [row['lat'], row['lon']],
        radius    = 8,
        color     = 'darkred',
        fill      = True,
        fill_color= 'red',
        fill_opacity = 0.8,
        popup = folium.Popup(
            f"<b>Sector:</b> {row['sector']}<br>"
            f"<b>Baseline PD:</b> {row['baseline_pd']:.3f}<br>"
            f"<b>CPD (Disorderly):</b> {row['cpd_Disorderly']:.3f}<br>"
            f"<b>CPD (TooLate):</b> {row['cpd_TooLate']:.3f}<br>"
            f"<b>Loan Amount:</b> ${row['loan_amnt']:,.0f}<br>"
            f"<b>Physical Risk:</b> {row['physical_risk_score']:.3f}<br>"
            f"<b>Transition Risk:</b> {row['transition_risk_score']:.3f}",
            max_width=250
        ),
        tooltip = f"CPD: {row['cpd_Disorderly']:.3f} | {row['sector']}"
    ).add_to(m)

# Legend
legend_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:white; padding:12px; border-radius:8px;
     border:2px solid #333; font-size:13px; font-family:Arial;">
  <b>GreenScore — Climate-Adjusted PD</b><br>
  <b>Scenario: Disorderly (NGFS)</b><br><br>
  <span style="color:blue">■</span> Low CPD (&lt; 0.40)<br>
  <span style="color:cyan">■</span> Moderate CPD (0.40–0.55)<br>
  <span style="color:lime">■</span> Elevated CPD (0.55–0.65)<br>
  <span style="color:orange">■</span> High CPD (0.65–0.75)<br>
  <span style="color:red">■</span> Critical CPD (&gt; 0.75)<br><br>
  <span style="color:darkred">●</span> Top-20 highest risk loans
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Title
title_html = """
<div style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
     z-index:1000; background:white; padding:10px 20px; border-radius:8px;
     border:2px solid #333; font-size:16px; font-family:Arial; font-weight:bold;">
  GreenScore Climate Risk Heatmap — Disorderly Scenario
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

out_path = f'{BASE}/04_outputs/heatmap.html'
m.save(out_path)
print(f"✅ Heatmap saved → {out_path}")
print("   Open in browser: start 04_outputs/heatmap.html")