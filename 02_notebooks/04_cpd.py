import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE = r'C:\Users\danus\GreenScore'
os.makedirs(f'{BASE}/04_outputs', exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────
print("Loading loans with baseline PD...")
df = pd.read_csv(f'{BASE}/01_data/processed/loans_with_pd.csv')
print(f"Shape: {df.shape}")
print(f"Baseline PD — mean: {df['baseline_pd'].mean():.4f}, "
      f"min: {df['baseline_pd'].min():.4f}, max: {df['baseline_pd'].max():.4f}")

# ── 2. NGFS SCENARIOS ─────────────────────────────────────────
# Each scenario defines physical and transition multiplier caps
# Based on NGFS Phase 4 (2023) scenario definitions
scenarios = {
    'Orderly':    {'phys_weight': 0.20, 'trans_weight': 0.15,
                   'description': 'Early, smooth climate policy — low disruption'},
    'Disorderly': {'phys_weight': 0.35, 'trans_weight': 0.40,
                   'description': 'Late policy action — high transition costs'},
    'HotHouse':   {'phys_weight': 0.55, 'trans_weight': 0.10,
                   'description': 'No policy action — severe physical risk'},
    'TooLate':    {'phys_weight': 0.55, 'trans_weight': 0.45,
                   'description': 'Too-late action — worst of both worlds'},
}

# ── 3. CPD FORMULA ────────────────────────────────────────────
# CPD = Baseline_PD × (1 + Physical_Risk_Factor) × (1 + Transition_Risk_Factor)
# Risk factors are scenario-weighted versions of our engineered scores

print("\nComputing CPD across 4 NGFS scenarios...")

for scenario, params in scenarios.items():
    phys_factor  = df['physical_risk_score']  * params['phys_weight']
    trans_factor = df['transition_risk_score'] * params['trans_weight']

    cpd = df['baseline_pd'] * (1 + phys_factor) * (1 + trans_factor)
    cpd = cpd.clip(0, 1)   # cap at 1.0

    df[f'cpd_{scenario}']          = cpd.round(6)
    df[f'pd_uplift_{scenario}_pct'] = ((cpd - df['baseline_pd']) /
                                        df['baseline_pd'] * 100).round(2)

    print(f"\n  {scenario} ({params['description']})")
    print(f"    CPD mean:   {cpd.mean():.4f}  (baseline: {df['baseline_pd'].mean():.4f})")
    print(f"    CPD max:    {cpd.max():.4f}")
    print(f"    Avg uplift: {((cpd - df['baseline_pd']) / df['baseline_pd'] * 100).mean():.1f}%")

# ── 4. PORTFOLIO SUMMARY BY SECTOR ───────────────────────────
print("\n\nPortfolio CPD by Sector (Disorderly scenario):")
sector_summary = df.groupby('sector').agg(
    loan_count       = ('baseline_pd', 'count'),
    mean_baseline_pd = ('baseline_pd', 'mean'),
    mean_cpd         = ('cpd_Disorderly', 'mean'),
    mean_uplift_pct  = ('pd_uplift_Disorderly_pct', 'mean')
).round(4).sort_values('mean_cpd', ascending=False)
print(sector_summary.to_string())

# ── 5. SAVE MAIN OUTPUT ───────────────────────────────────────
out_path = f'{BASE}/04_outputs/portfolio_cpd.csv'
df.to_csv(out_path, index=False)
print(f"\n✅ Portfolio CPD saved → {out_path}")
print(f"   Columns added: cpd_Orderly, cpd_Disorderly, cpd_HotHouse, cpd_TooLate")
print(f"   + uplift % columns for each scenario")

# ── 6. VISUALISATIONS ─────────────────────────────────────────

# Plot 1 — CPD distribution across scenarios
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GreenScore — Climate-Adjusted PD Distribution by NGFS Scenario',
             fontsize=13, fontweight='bold')

colors = {'Orderly':'#2ecc71', 'Disorderly':'#f39c12',
          'HotHouse':'#e74c3c', 'TooLate':'#8e44ad'}

for ax, scenario in zip(axes.flatten(), scenarios.keys()):
    ax.hist(df['baseline_pd'],        bins=50, alpha=0.5,
            color='steelblue', label='Baseline PD', density=True)
    ax.hist(df[f'cpd_{scenario}'],    bins=50, alpha=0.6,
            color=colors[scenario], label=f'CPD ({scenario})', density=True)
    ax.set_title(f'{scenario} — {scenarios[scenario]["description"]}',
                 fontsize=10)
    ax.set_xlabel('Probability of Default')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.axvline(df['baseline_pd'].mean(),     color='steelblue',
               linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(df[f'cpd_{scenario}'].mean(), color=colors[scenario],
               linestyle='--', alpha=0.9, linewidth=1.5)

plt.tight_layout()
plt.savefig(f'{BASE}/04_outputs/cpd_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("CPD distribution plot saved.")

# Plot 2 — Sector CPD heatmap
fig2, ax2 = plt.subplots(figsize=(10, 6))
scenario_cols = [f'cpd_{s}' for s in scenarios]
heatmap_data  = df.groupby('sector')[scenario_cols].mean()
heatmap_data.columns = list(scenarios.keys())

im = ax2.imshow(heatmap_data.values, cmap='RdYlGn_r', aspect='auto',
                vmin=0.3, vmax=0.7)
ax2.set_xticks(range(len(scenarios)))
ax2.set_xticklabels(list(scenarios.keys()), fontsize=11)
ax2.set_yticks(range(len(heatmap_data)))
ax2.set_yticklabels(heatmap_data.index, fontsize=11)
ax2.set_title('Mean CPD by Sector × NGFS Scenario', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax2, label='Mean CPD')

for i in range(len(heatmap_data)):
    for j in range(len(scenarios)):
        ax2.text(j, i, f'{heatmap_data.values[i,j]:.3f}',
                 ha='center', va='center', fontsize=10, fontweight='bold', color='black')

plt.tight_layout()
plt.savefig(f'{BASE}/04_outputs/sector_cpd_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Sector CPD heatmap saved.")

# Plot 3 — Uplift % by scenario
fig3, ax3 = plt.subplots(figsize=(10, 5))
uplift_data = {s: df[f'pd_uplift_{s}_pct'].mean() for s in scenarios}
bars = ax3.bar(uplift_data.keys(), uplift_data.values(),
               color=[colors[s] for s in scenarios], edgecolor='black', linewidth=0.5)
ax3.set_title('Average PD Uplift (%) vs Baseline by NGFS Scenario',
              fontsize=13, fontweight='bold')
ax3.set_ylabel('Mean PD Uplift (%)')
ax3.set_xlabel('NGFS Scenario')
for bar, val in zip(bars, uplift_data.values()):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}/04_outputs/scenario_uplift.png', dpi=150, bbox_inches='tight')
plt.show()
print("Scenario uplift plot saved.")

print("\n✅ Step 5 complete — all CPD outputs saved to 04_outputs/")