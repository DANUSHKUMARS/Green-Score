# 🌍 GreenScore — Climate-Adjusted Credit Risk Engine

**Course:** MGT3013 Risk & Fraud Analytics | VIT Chennai  
**Student:** Danush | 22MIA1039  
**Type:** J-Component Project

---

## What is GreenScore?

GreenScore computes a **Climate-Adjusted Probability of Default (CPD)** for loan portfolios by layering NASA satellite climate data and NGFS carbon transition scenarios on top of an XGBoost credit risk model.
```
CPD = Baseline_PD × (1 + Physical_Risk_Factor) × (1 + Transition_Risk_Factor)
```

---

## Architecture

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_eda.py` | Load & inspect LendingClub dataset |
| 2 | `02_features.py` | Clean data + NASA POWER API climate features |
| 3 | `03_model.py` | XGBoost baseline PD model (AUC 0.706) |
| 4 | `04_cpd.py` | CPD calculation across 4 NGFS scenarios |
| 5 | `05_heatmap.py` | Folium geographic risk heatmap |
| 6 | `05_dashboard/app.py` | Streamlit interactive dashboard |

---

## Key Results

| NGFS Scenario | Avg PD Uplift | Description |
|---------------|--------------|-------------|
| Orderly | +9.6% | Early smooth policy |
| Disorderly | +17.9% | Late policy action |
| HotHouse | +24.3% | No policy action |
| TooLate | +27.0% | Worst of both worlds |

**Highest risk sector:** Energy (+46.6% CPD uplift under Disorderly)

---

## Data Sources

- **LendingClub 2007–2018** — Kaggle (2.26M loans)
- **NASA POWER API** — Monthly temperature & precipitation per US state
- **NGFS Phase 4 Scenarios** — Carbon price pathways
- **RBI DBIE** — Sector NPA rates (hardcoded)

---

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/GreenScore.git
cd GreenScore
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download LendingClub dataset from Kaggle and place in `01_data/raw/`.

Then run scripts in order:
```bash
python 02_notebooks/01_eda.py
python 02_notebooks/02_features.py
python 02_notebooks/03_model.py
python 02_notebooks/04_cpd.py
python 02_notebooks/05_heatmap.py
streamlit run 05_dashboard/app.py
```

---

## Model Performance

- **AUC-ROC:** 0.706
- **CV AUC:** 0.711 ± 0.006
- **Features:** 18 (10 financial + 5 physical climate + 3 transition risk)