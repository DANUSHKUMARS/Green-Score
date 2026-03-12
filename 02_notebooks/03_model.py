import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                             classification_report, RocCurveDisplay)
import xgboost as xgb
import shap

BASE = r'C:\Users\danus\GreenScore'
os.makedirs(f'{BASE}/03_models',  exist_ok=True)
os.makedirs(f'{BASE}/04_outputs', exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────
print("Loading feature matrix...")
df = pd.read_csv(f'{BASE}/01_data/processed/features_complete.csv')
print(f"Shape: {df.shape} | Default rate: {df['default'].mean():.2%}")

FEATURE_COLS = [
    'loan_amnt', 'int_rate', 'installment', 'emp_length',
    'annual_inc', 'dti', 'fico_range_low', 'grade_num',
    'owns_home', 'loan_to_income',
    'flood_freq_score', 'drought_idx', 'temp_anomaly',
    'extreme_events', 'physical_risk_score',
    'carbon_intensity', 'carbon_burden', 'transition_risk_score'
]

X = df[FEATURE_COLS]
y = df['default']

# ── 2. SPLIT ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ── 3. MODEL — two instances ──────────────────────────────────
# model_es  : used for training with early stopping (needs eval_set)
# model_cv  : used for cross_val_score (no early_stopping_rounds)

model_es = xgb.XGBClassifier(
    n_estimators          = 500,
    max_depth             = 6,
    learning_rate         = 0.03,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 5,
    gamma                 = 1,
    reg_alpha             = 0.1,
    reg_lambda            = 1.5,
    scale_pos_weight      = scale_pos_weight,
    eval_metric           = 'auc',
    early_stopping_rounds = 40,
    random_state          = 42,
    verbosity             = 0
)

model_cv = xgb.XGBClassifier(
    n_estimators     = 300,       # fixed — no early stopping
    max_depth        = 6,
    learning_rate    = 0.03,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    gamma            = 1,
    reg_alpha        = 0.1,
    reg_lambda       = 1.5,
    scale_pos_weight = scale_pos_weight,
    eval_metric      = 'auc',
    random_state     = 42,
    verbosity        = 0
)

# ── 4. TRAIN WITH EARLY STOPPING ─────────────────────────────
print("\nTraining XGBoost...")
model_es.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)
best_iter = model_es.best_iteration
print(f"Best iteration: {best_iter}")

# ── 5. EVALUATE ───────────────────────────────────────────────
y_proba = model_es.predict_proba(X_test)[:, 1]
y_pred  = model_es.predict(X_test)

auc   = roc_auc_score(y_test, y_proba)
brier = brier_score_loss(y_test, y_proba)

print("\n" + "="*50)
print(f"  AUC-ROC : {auc:.4f}   (target > 0.72)")
print(f"  Brier   : {brier:.4f}  (target < 0.20)")
print("="*50)
print(classification_report(y_test, y_pred,
      target_names=['No Default', 'Default']))

# ── 6. CROSS VALIDATION (no early stopping) ───────────────────
print("Running 5-fold CV...")
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = cross_val_score(model_cv, X, y, cv=cv, scoring='roc_auc')
print(f"CV AUC: {cv_aucs.mean():.4f} (+/- {cv_aucs.std():.4f})")

# ── 7. PLOTS ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('GreenScore — XGBoost Model Evaluation', fontsize=14, fontweight='bold')

RocCurveDisplay.from_estimator(model_es, X_test, y_test, ax=axes[0])
axes[0].set_title(f'ROC Curve (AUC = {auc:.3f})')
axes[0].plot([0,1],[0,1],'k--', alpha=0.4)

xgb.plot_importance(model_es, ax=axes[1], max_num_features=15,
                    title='Feature Importance (Gain)',
                    importance_type='gain')
plt.tight_layout()
plt.savefig(f'{BASE}/04_outputs/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Model evaluation plot saved.")

# ── 8. SHAP ───────────────────────────────────────────────────
print("\nComputing SHAP values...")
explainer   = shap.TreeExplainer(model_es)
shap_sample = X_test.sample(1000, random_state=42)
shap_values = explainer.shap_values(shap_sample)

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('GreenScore — SHAP Analysis', fontsize=14, fontweight='bold')

plt.sca(axes2[0])
shap.summary_plot(shap_values, shap_sample,
                  feature_names=FEATURE_COLS,
                  plot_type='bar', show=False)
axes2[0].set_title('Mean |SHAP| — Global Importance')

plt.sca(axes2[1])
shap.summary_plot(shap_values, shap_sample,
                  feature_names=FEATURE_COLS, show=False)
axes2[1].set_title('SHAP Beeswarm — Direction of Impact')

plt.tight_layout()
plt.savefig(f'{BASE}/04_outputs/shap_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP plots saved.")

# ── 9. BASELINE PD FOR ALL LOANS ─────────────────────────────
print("\nGenerating baseline PD for all loans...")
df['baseline_pd'] = model_es.predict_proba(df[FEATURE_COLS])[:, 1]
print(df['baseline_pd'].describe().round(4))

out_path = f'{BASE}/01_data/processed/loans_with_pd.csv'
df.to_csv(out_path, index=False)
joblib.dump(model_es, f'{BASE}/03_models/xgb_model.pkl')

print(f"\n✅ Model saved  → 03_models/xgb_model.pkl")
print(f"✅ Loans + PD   → {out_path}")
print(f"\nSummary:")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"  Brier    : {brier:.4f}")
print(f"  CV AUC   : {cv_aucs.mean():.4f} (+/- {cv_aucs.std():.4f})")