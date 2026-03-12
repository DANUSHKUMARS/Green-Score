import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title = "GreenScore — Climate Risk Engine",
    page_icon  = "🌍",
    layout     = "wide"
)

BASE = r'C:\Users\danus\GreenScore'

@st.cache_data
def load_data():
    return pd.read_csv(f'{BASE}/04_outputs/portfolio_cpd.csv')

df = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/320px-24701-nature-natural-beauty.jpg", width=280)
st.sidebar.title("🌍 GreenScore")
st.sidebar.markdown("**Climate-Adjusted Credit Risk Engine**")
st.sidebar.markdown("*MGT3013 — VIT Chennai*")
st.sidebar.divider()

scenario = st.sidebar.selectbox(
    "NGFS Climate Scenario",
    ['Orderly', 'Disorderly', 'HotHouse', 'TooLate'],
    index=1,
    help="Select an NGFS climate scenario to view its impact on credit risk"
)

scenario_desc = {
    'Orderly':    '🟢 Early, smooth policy — low disruption',
    'Disorderly': '🟡 Late policy action — high transition costs',
    'HotHouse':   '🔴 No policy action — severe physical risk',
    'TooLate':    '🟣 Too-late action — worst of both worlds',
}
st.sidebar.info(scenario_desc[scenario])
st.sidebar.divider()

sector_filter = st.sidebar.multiselect(
    "Filter by Sector",
    options=sorted(df['sector'].unique()),
    default=sorted(df['sector'].unique())
)

df_filtered = df[df['sector'].isin(sector_filter)]
cpd_col     = f'cpd_{scenario}'
uplift_col  = f'pd_uplift_{scenario}_pct'

# ── HEADER ────────────────────────────────────────────────────
st.title("🌍 GreenScore — Climate-Adjusted Credit Risk Engine")
st.markdown(f"**Active Scenario:** {scenario} &nbsp;|&nbsp; "
            f"**Loans Analysed:** {len(df_filtered):,} &nbsp;|&nbsp; "
            f"**Dataset:** LendingClub 2007–2018")
st.divider()

# ── KPI METRICS ───────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Baseline PD", f"{df_filtered['baseline_pd'].mean():.3f}",
              help="XGBoost predicted probability of default")
with col2:
    cpd_mean = df_filtered[cpd_col].mean()
    delta    = cpd_mean - df_filtered['baseline_pd'].mean()
    st.metric(f"CPD ({scenario})", f"{cpd_mean:.3f}",
              delta=f"+{delta:.3f}", delta_color="inverse")
with col3:
    st.metric("Avg PD Uplift", f"{df_filtered[uplift_col].mean():.1f}%",
              help="Mean increase in PD due to climate risk")
with col4:
    high_risk = (df_filtered[cpd_col] > 0.6).sum()
    st.metric("High Risk Loans", f"{high_risk:,}",
              help="Loans with CPD > 0.60")
with col5:
    st.metric("AUC-ROC", "0.706",
              help="XGBoost baseline model performance")

st.divider()

# ── ROW 1: CPD DISTRIBUTION + SECTOR BAR ─────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("CPD vs Baseline PD Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df_filtered['baseline_pd'], name='Baseline PD',
        opacity=0.6, marker_color='steelblue',
        nbinsx=50, histnorm='probability density'
    ))
    fig_dist.add_trace(go.Histogram(
        x=df_filtered[cpd_col], name=f'CPD ({scenario})',
        opacity=0.6, marker_color='crimson',
        nbinsx=50, histnorm='probability density'
    ))
    fig_dist.update_layout(
        barmode='overlay',
        xaxis_title='Probability of Default',
        yaxis_title='Density',
        legend=dict(x=0.6, y=0.95),
        height=380
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_b:
    st.subheader(f"Mean CPD by Sector — {scenario} Scenario")
    sector_df = df_filtered.groupby('sector').agg(
        Baseline_PD = ('baseline_pd', 'mean'),
        CPD         = (cpd_col, 'mean'),
        Uplift_pct  = (uplift_col, 'mean'),
        Count       = ('baseline_pd', 'count')
    ).reset_index().sort_values('CPD', ascending=True)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=sector_df['sector'], x=sector_df['Baseline_PD'],
        name='Baseline PD', orientation='h',
        marker_color='steelblue', opacity=0.7
    ))
    fig_bar.add_trace(go.Bar(
        y=sector_df['sector'], x=sector_df['CPD'],
        name=f'CPD ({scenario})', orientation='h',
        marker_color='crimson', opacity=0.7
    ))
    fig_bar.update_layout(
        barmode='group',
        xaxis_title='Probability of Default',
        height=380,
        legend=dict(x=0.5, y=0.05)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── ROW 2: SCENARIO COMPARISON + HEATMAP EMBED ───────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Scenario Comparison — Avg PD Uplift (%)")
    scenarios_all = ['Orderly', 'Disorderly', 'HotHouse', 'TooLate']
    uplift_vals   = [df_filtered[f'pd_uplift_{s}_pct'].mean() for s in scenarios_all]
    colors_map    = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']

    fig_scen = go.Figure(go.Bar(
        x=scenarios_all, y=uplift_vals,
        marker_color=colors_map,
        text=[f'{v:.1f}%' for v in uplift_vals],
        textposition='outside'
    ))
    fig_scen.update_layout(
        yaxis_title='Mean PD Uplift (%)',
        height=380,
        yaxis=dict(range=[0, max(uplift_vals)*1.2])
    )
    st.plotly_chart(fig_scen, use_container_width=True)

with col_d:
    st.subheader("Sector × Scenario CPD Heatmap")
    heat_df = df_filtered.groupby('sector')[
        [f'cpd_{s}' for s in scenarios_all]
    ].mean().round(3)
    heat_df.columns = scenarios_all

    fig_heat = px.imshow(
        heat_df, text_auto=True, aspect='auto',
        color_continuous_scale='RdYlGn_r',
        zmin=0.3, zmax=0.7,
        labels=dict(color='Mean CPD')
    )
    fig_heat.update_layout(height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── ROW 3: TOP-20 RISK TABLE + SCATTER ───────────────────────
col_e, col_f = st.columns([1.2, 1])

with col_e:
    st.subheader(f"Top 20 Highest Risk Loans — {scenario} Scenario")
    top20 = df_filtered.nlargest(20, cpd_col)[[
        'sector', 'loan_amnt', 'annual_inc', 'fico_range_low',
        'baseline_pd', cpd_col, uplift_col,
        'physical_risk_score', 'transition_risk_score'
    ]].round(4)
    top20.columns = [
        'Sector', 'Loan Amt', 'Annual Inc', 'FICO',
        'Baseline PD', 'CPD', 'Uplift %',
        'Phys Risk', 'Trans Risk'
    ]
    st.dataframe(top20, use_container_width=True, height=420)

with col_f:
    st.subheader("Physical Risk vs Transition Risk")
    sample = df_filtered.sample(min(2000, len(df_filtered)), random_state=42)
    fig_sc = px.scatter(
        sample,
        x='physical_risk_score',
        y='transition_risk_score',
        color=cpd_col,
        color_continuous_scale='RdYlGn_r',
        hover_data=['sector', 'baseline_pd'],
        labels={
            'physical_risk_score':   'Physical Risk Score',
            'transition_risk_score': 'Transition Risk Score',
            cpd_col:                 'CPD'
        },
        opacity=0.6,
        range_color=[0.3, 0.7]
    )
    fig_sc.update_layout(height=420)
    st.plotly_chart(fig_sc, use_container_width=True)

st.divider()

# ── FOOTER ────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; color:grey; font-size:12px;'>
    GreenScore — Climate-Adjusted Credit Risk Engine &nbsp;|&nbsp;
    MGT3013 Risk & Fraud Analytics &nbsp;|&nbsp; VIT Chennai &nbsp;|&nbsp;
    Data: LendingClub 2007–2018 + NASA POWER API + NGFS Phase 4 Scenarios
    </div>
    """,
    unsafe_allow_html=True
)