import pandas as pd
import numpy as np
import os

BASE = r'C:\Users\danus\GreenScore'
RAW  = f'{BASE}/01_data/raw/accepted_2007_to_2018Q4.csv'
OUT  = f'{BASE}/01_data/processed/loans_cleaned.csv'

os.makedirs(f'{BASE}/01_data/processed', exist_ok=True)

# ── 1. LOAD ONLY THE 12 COLUMNS WE NEED ───────────────────────
print("Loading dataset (this takes ~30 seconds for 1.6GB)...")

COLS = [
    'loan_amnt', 'annual_inc', 'dti', 'loan_status',
    'purpose', 'addr_state', 'grade', 'emp_length',
    'home_ownership', 'fico_range_low', 'int_rate', 'installment'
]

df = pd.read_csv(RAW, usecols=COLS, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

# ── 2. INSPECT TARGET COLUMN ───────────────────────────────────
print("\nloan_status value counts:")
print(df['loan_status'].value_counts())

# ── 3. CREATE BINARY DEFAULT TARGET ───────────────────────────
default_labels = [
    'Charged Off',
    'Default',
    'Late (31-120 days)',
    'Does not meet the credit policy. Status:Charged Off'
]
df['default'] = df['loan_status'].isin(default_labels).astype(int)
print(f"\nDefault rate: {df['default'].mean():.2%}")
print(f"Defaults: {df['default'].sum():,} | Non-defaults: {(df['default']==0).sum():,}")

# Drop original loan_status
df.drop(columns=['loan_status'], inplace=True)

# ── 4. CLEAN NUMERIC COLUMNS ───────────────────────────────────
# int_rate comes as "13.56%" string — strip % and convert
df['int_rate'] = df['int_rate'].astype(str).str.replace('%','').str.strip()
df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')

# emp_length: "10+ years", "< 1 year" etc → numeric
emp_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
    '4 years': 4,  '5 years': 5, '6 years': 6, '7 years': 7,
    '8 years': 8,  '9 years': 9, '10+ years': 10
}
df['emp_length'] = df['emp_length'].map(emp_map)

# grade: A=1, B=2, ... G=7
grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
df['grade_num'] = df['grade'].map(grade_map)
df.drop(columns=['grade'], inplace=True)

# home_ownership → binary own vs not
df['owns_home'] = df['home_ownership'].isin(['OWN','MORTGAGE']).astype(int)
df.drop(columns=['home_ownership'], inplace=True)

# loan_to_income ratio
df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)

# ── 5. MISSING VALUES ──────────────────────────────────────────
print("\nMissing values before treatment:")
print(df.isnull().sum()[df.isnull().sum() > 0])

for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)

print("Missing values after treatment:", df.isnull().sum().sum())

# ── 6. WINSORIZE OUTLIERS ──────────────────────────────────────
for col in ['annual_inc', 'loan_amnt', 'dti', 'loan_to_income']:
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)

# ── 7. STANDARDIZE PURPOSE → SECTOR ───────────────────────────
sector_map = {
    'agricultural':      'Agriculture',
    'small_business':    'Manufacturing',
    'business':          'Manufacturing',
    'home_improvement':  'Real_Estate',
    'house':             'Real_Estate',
    'moving':            'Real_Estate',
    'car':               'Transport',
    'vacation':          'Retail',
    'wedding':           'Retail',
    'major_purchase':    'Retail',
    'medical':           'Services',
    'educational':       'Services',
    'debt_consolidation':'Services',
    'credit_card':       'Services',
    'other':             'Services',
    'renewable_energy':  'Energy',
}
df['sector'] = df['purpose'].map(sector_map).fillna('Services')
df.drop(columns=['purpose'], inplace=True)

print("\nSector distribution:")
print(df['sector'].value_counts())

# ── 8. MAP US STATES → GPS COORDS ─────────────────────────────
# LendingClub is US data — we map states to coords for NASA API
state_coords = {
    'AL':(32.8,-86.8), 'AK':(64.2,-153.4), 'AZ':(34.3,-111.1),
    'AR':(34.8,-92.2), 'CA':(36.8,-119.4), 'CO':(39.0,-105.5),
    'CT':(41.6,-72.7), 'DE':(39.0,-75.5),  'FL':(27.8,-81.6),
    'GA':(32.2,-83.4), 'HI':(19.9,-155.6), 'ID':(44.1,-114.5),
    'IL':(40.0,-89.2), 'IN':(39.8,-86.1),  'IA':(42.0,-93.2),
    'KS':(38.5,-98.4), 'KY':(37.5,-85.3),  'LA':(31.2,-91.8),
    'ME':(44.7,-69.4), 'MD':(39.1,-76.8),  'MA':(42.2,-71.5),
    'MI':(44.3,-85.4), 'MN':(46.4,-93.1),  'MS':(32.7,-89.7),
    'MO':(38.3,-92.5), 'MT':(46.9,-110.5), 'NE':(41.5,-99.9),
    'NV':(38.5,-117.1),'NH':(43.5,-71.6),  'NJ':(40.1,-74.5),
    'NM':(34.4,-106.1),'NY':(42.2,-74.9),  'NC':(35.6,-79.8),
    'ND':(47.5,-100.5),'OH':(40.4,-82.8),  'OK':(35.6,-96.9),
    'OR':(44.6,-122.1),'PA':(40.9,-77.8),  'RI':(41.6,-71.5),
    'SC':(33.8,-81.2), 'SD':(44.4,-100.2), 'TN':(35.7,-86.7),
    'TX':(31.1,-97.6), 'UT':(39.4,-111.1), 'VT':(44.0,-72.7),
    'VA':(37.5,-78.5), 'WA':(47.4,-121.5), 'WV':(38.6,-80.6),
    'WI':(44.3,-89.8), 'WY':(43.0,-107.6),
}
df['lat'] = df['addr_state'].map(lambda s: state_coords.get(s,(39.5,-98.4))[0])
df['lon'] = df['addr_state'].map(lambda s: state_coords.get(s,(39.5,-98.4))[1])
df.drop(columns=['addr_state'], inplace=True)

# ── 9. FINAL SHAPE & SAVE ──────────────────────────────────────
print(f"\nFinal shape: {df.shape}")
print("Final columns:", df.columns.tolist())
print(f"\nDefault rate in clean dataset: {df['default'].mean():.2%}")

df.to_csv(OUT, index=False)
print(f"\n✅ Saved to {OUT}")