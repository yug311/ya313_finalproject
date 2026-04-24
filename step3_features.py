import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("="*60)
print("STEP 3: FEATURE ENGINEERING FOR MODELING")
print("="*60)

# ── 1. LOAD CLEAN DATA ────────────────────────────────────────
print("\n[1] Loading clean dataset...")
df = pd.read_csv('/home/claude/clean_movies.csv')
df['genres_list']    = df['genres_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
df['companies_list'] = df['companies_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
print(f"    Loaded {len(df)} rows × {df.shape[1]} columns")

# ── 2. DIRECTOR / ACTOR POPULARITY FEATURES ──────────────────
print("\n[2] Building director & actor track record features...")

# For each director: how many prior films did they have in the dataset,
# and what was their historical median ROI going INTO this film?
# We sort by year so we only use past data (no leakage).
df = df.sort_values('year').reset_index(drop=True)

director_prior_roi    = {}
director_film_count   = {}
director_roi_feature  = []
director_count_feature= []

for idx, row in df.iterrows():
    d = row['director']
    if pd.isna(d):
        director_roi_feature.append(np.nan)
        director_count_feature.append(0)
    else:
        past_rois = director_prior_roi.get(d, [])
        director_roi_feature.append(np.median(past_rois) if past_rois else np.nan)
        director_count_feature.append(len(past_rois))
        # Now update with this film
        director_prior_roi.setdefault(d, []).append(row['roi'])

df['director_prior_roi']    = director_roi_feature
df['director_film_count']   = director_count_feature
df['director_is_established'] = (df['director_film_count'] >= 3).astype(int)

# Fill missing director ROI with global median (for debut directors)
global_median_roi = df['roi'].median()
df['director_prior_roi'] = df['director_prior_roi'].fillna(global_median_roi)

print(f"    ✓ director_prior_roi, director_film_count, director_is_established")

# ── 3. PRODUCTION COMPANY TIER ───────────────────────────────
print("\n[3] Classifying production company tier...")

# Major studios (historically dominant)
major_studios = {
    'Warner Bros.', 'Universal Pictures', 'Paramount Pictures',
    'Columbia Pictures', 'Walt Disney Pictures', 'Twentieth Century Fox Film Corporation',
    'New Line Cinema', 'DreamWorks', 'Metro-Goldwyn-Mayer', 'Lionsgate',
    'Touchstone Pictures', 'TriStar Pictures', 'Miramax Films',
    'Sony Pictures', 'Revolution Studios', 'Working Title Films'
}

def company_tier(companies):
    if not companies:
        return 'Independent'
    for c in companies:
        if c in major_studios:
            return 'Major'
    return 'Independent'

df['studio_tier'] = df['companies_list'].apply(company_tier)
df['is_major_studio'] = (df['studio_tier'] == 'Major').astype(int)
print(f"    Major studio: {df['is_major_studio'].sum()} | Independent: {(df['is_major_studio']==0).sum()}")

# ── 4. GENRE COUNT (MULTI-GENRE FILMS) ───────────────────────
print("\n[4] Computing genre complexity...")
genre_cols = [c for c in df.columns if c.startswith('genre_')]
df['genre_count'] = df[genre_cols].sum(axis=1)
print(f"    Mean genres per film: {df['genre_count'].mean():.2f}")

# ── 5. BUDGET TIER ───────────────────────────────────────────
print("\n[5] Creating budget tier categories...")
df['budget_tier'] = pd.qcut(df['budget'], q=4,
                             labels=['Low', 'Medium', 'High', 'Blockbuster'])
print(df['budget_tier'].value_counts().to_string())

# ── 6. RUNTIME FEATURES ──────────────────────────────────────
print("\n[6] Engineering runtime features...")
df['runtime_squared'] = df['runtime'] ** 2   # capture non-linear effect
df['is_long_film']    = (df['runtime'] > 120).astype(int)
df['is_short_film']   = (df['runtime'] < 90).astype(int)
print(f"    Long (>120min): {df['is_long_film'].sum()} | Short (<90min): {df['is_short_film'].sum()}")

# ── 7. VOTE COUNT BUCKETS (PROXY FOR AWARENESS) ───────────────
print("\n[7] Vote count as audience awareness proxy...")
df['log_vote_count'] = np.log1p(df['vote_count'])
df['high_awareness'] = (df['vote_count'] > df['vote_count'].median()).astype(int)

# ── 8. ENCODE CATEGORICAL VARIABLES ──────────────────────────
print("\n[8] Encoding categorical variables...")

# Primary genre → one-hot (already have genre dummies, so just ensure primary_genre is clean)
df['primary_genre'] = df['primary_genre'].fillna('Unknown')

# Release season → ordinal encoding (Summer/Holiday >> Spring >> Other)
season_map = {'Summer': 3, 'Holiday': 3, 'Spring': 2, 'Other': 1}
df['season_score'] = df['release_season'].map(season_map).fillna(1)

# Period → ordinal
period_map = {'Pre-2000': 0, '2000-2009': 1, '2010+': 2}
df['period_ord'] = df['period'].map(period_map)

# Budget tier → ordinal
tier_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Blockbuster': 3}
df['budget_tier_ord'] = df['budget_tier'].map(tier_map)

print("    ✓ season_score, period_ord, budget_tier_ord encoded")

# ── 9. DEFINE FEATURE SETS ────────────────────────────────────
print("\n[9] Defining feature sets for modeling...")

# Core numeric features
NUMERIC_FEATURES = [
    'log_budget',
    'runtime',
    'vote_average',
    'log_vote_count',
    'popularity',
    'director_prior_roi',
    'director_film_count',
    'genre_count',
    'season_score',
    'is_english',
    'is_major_studio',
    'director_is_established',
    'is_long_film',
    'is_short_film',
    'high_awareness',
    'budget_tier_ord',
]

# Genre dummies
GENRE_FEATURES = [c for c in df.columns if c.startswith('genre_')]

# All features combined
ALL_FEATURES = NUMERIC_FEATURES + GENRE_FEATURES

TARGET = 'success'

print(f"    Numeric features:  {len(NUMERIC_FEATURES)}")
print(f"    Genre features:    {len(GENRE_FEATURES)}")
print(f"    Total features:    {len(ALL_FEATURES)}")

# ── 10. BUILD MODELING DATAFRAME ─────────────────────────────
print("\n[10] Building final modeling dataframe...")

model_df = df[ALL_FEATURES + [TARGET, 'period', 'year', 'title']].copy()

# Drop any rows with nulls in features
before = len(model_df)
model_df = model_df.dropna(subset=ALL_FEATURES + [TARGET])
after = len(model_df)
print(f"    Dropped {before - after} rows with nulls → {after} rows ready for modeling")

# ── 11. SCALE NUMERIC FEATURES ───────────────────────────────
print("\n[11] Scaling numeric features...")

X = model_df[ALL_FEATURES]
y = model_df[TARGET]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])

print(f"    ✓ StandardScaler applied to {len(NUMERIC_FEATURES)} numeric features")
print(f"    Genre dummies left as-is (already 0/1)")

# ── 12. TRAIN / TEST SPLIT ───────────────────────────────────
print("\n[12] Train/test split (80/20, stratified)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"    Train success rate: {y_train.mean()*100:.1f}% | Test: {y_test.mean()*100:.1f}%")

# ── 13. PERIOD-SPECIFIC SPLITS ───────────────────────────────
print("\n[13] Creating period-specific datasets...")

period_datasets = {}
for period in ['Pre-2000', '2000-2009', '2010+']:
    mask   = model_df['period'] == period
    X_p    = X_scaled[mask]
    y_p    = y[mask]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_p, y_p, test_size=0.2, random_state=42, stratify=y_p
    )
    period_datasets[period] = {
        'X_train': X_tr, 'X_test': X_te,
        'y_train': y_tr, 'y_test': y_te,
        'n_total': len(X_p)
    }
    print(f"    {period}: {len(X_p)} total | train={len(X_tr)} test={len(X_te)} | success={y_p.mean()*100:.1f}%")

# ── 14. SAVE EVERYTHING ───────────────────────────────────────
print("\n[14] Saving modeling artifacts...")

import pickle

artifacts = {
    'X_train':         X_train,
    'X_test':          X_test,
    'y_train':         y_train,
    'y_test':          y_test,
    'X_scaled':        X_scaled,
    'y':               y,
    'model_df':        model_df,
    'scaler':          scaler,
    'ALL_FEATURES':    ALL_FEATURES,
    'NUMERIC_FEATURES':NUMERIC_FEATURES,
    'GENRE_FEATURES':  GENRE_FEATURES,
    'TARGET':          TARGET,
    'period_datasets': period_datasets,
}

with open('/home/claude/modeling_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# Also save the enriched df for reference
df.to_csv('/home/claude/clean_movies_enriched.csv', index=False)

print("    ✓ modeling_artifacts.pkl saved")
print("    ✓ clean_movies_enriched.csv saved")

# ── 15. FEATURE SUMMARY ───────────────────────────────────────
print("\n" + "="*60)
print("FEATURE ENGINEERING SUMMARY")
print("="*60)
print(f"\n  Total modeling rows:  {len(model_df)}")
print(f"  Total features:       {len(ALL_FEATURES)}")
print(f"  Target:               {TARGET} (ROI ≥ 100%)")
print(f"  Class balance:        {y.mean()*100:.1f}% positive (successful)")

print("\n  Feature list:")
for i, f in enumerate(ALL_FEATURES, 1):
    print(f"    {i:2d}. {f}")

print("\n✅ Step 3 Complete — Ready for modeling")
