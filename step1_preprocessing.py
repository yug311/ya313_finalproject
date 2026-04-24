import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STEP 1: DATA PREPROCESSING & CLEANING")
print("="*60)

# ── 1. LOAD RAW DATA ─────────────────────────────────────────
print("\n[1] Loading raw data...")
movies  = pd.read_csv('/mnt/user-data/uploads/tmdb_5000_movies.csv')
credits = pd.read_csv('/mnt/user-data/uploads/tmdb_5000_credits.csv')
print(f"    Movies:  {movies.shape}")
print(f"    Credits: {credits.shape}")

# ── 2. MERGE ──────────────────────────────────────────────────
print("\n[2] Merging datasets on movie id...")
credits = credits.rename(columns={'movie_id': 'id'})
df = movies.merge(credits[['id', 'cast', 'crew']], on='id', how='left')
print(f"    Merged shape: {df.shape}")

# ── 3. PARSE JSON COLUMNS ─────────────────────────────────────
print("\n[3] Parsing JSON columns...")

def parse_json_names(cell, key='name', top_n=None):
    """Extract 'name' field from a JSON list string."""
    try:
        items = json.loads(cell)
        names = [item[key] for item in items if key in item]
        return names[:top_n] if top_n else names
    except:
        return []

def extract_director(crew_cell):
    """Extract director name from crew JSON."""
    try:
        crew = json.loads(crew_cell)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name', np.nan)
    except:
        pass
    return np.nan

def extract_cast_ordered(cast_cell, top_n=3):
    """Extract top N billed cast members (by order field)."""
    try:
        cast = json.loads(cast_cell)
        cast_sorted = sorted(cast, key=lambda x: x.get('order', 999))
        return [m['name'] for m in cast_sorted[:top_n]]
    except:
        return []

# Parse genres → list of names, and primary genre (first listed)
df['genres_list']    = df['genres'].apply(parse_json_names)
df['primary_genre']  = df['genres_list'].apply(lambda x: x[0] if x else np.nan)

# Parse keywords
df['keywords_list']  = df['keywords'].apply(parse_json_names)

# Parse production companies
df['companies_list'] = df['production_companies'].apply(parse_json_names)

# Parse production countries
df['countries_list'] = df['production_countries'].apply(
    lambda x: parse_json_names(x, key='iso_3166_1')
)
df['primary_country'] = df['countries_list'].apply(lambda x: x[0] if x else np.nan)

# Extract director
df['director'] = df['crew'].apply(extract_director)

# Extract top 3 cast
df['top_cast'] = df['cast'].apply(lambda x: extract_cast_ordered(x, top_n=3))
df['lead_actor'] = df['top_cast'].apply(lambda x: x[0] if x else np.nan)

print("    ✓ genres, keywords, companies, countries, director, cast parsed")

# ── 4. DATE & TIME FEATURES ───────────────────────────────────
print("\n[4] Extracting date/time features...")
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['year']         = df['release_date'].dt.year
df['month']        = df['release_date'].dt.month
df['decade']       = (df['year'] // 10 * 10).astype('Int64')

# Map month to season (useful feature)
def month_to_season(m):
    if pd.isna(m): return np.nan
    if m in [6, 7, 8]:   return 'Summer'
    if m in [11, 12]:    return 'Holiday'
    if m in [3, 4, 5]:   return 'Spring'
    return 'Other'

df['release_season'] = df['month'].apply(month_to_season)
print("    ✓ year, month, decade, release_season extracted")

# ── 5. FILTER: KEEP ONLY RELEASED FILMS WITH FINANCIAL DATA ──
print("\n[5] Filtering rows...")
before = len(df)

# Keep only released movies
df = df[df['status'] == 'Released']

# Remove zero budgets and revenues (missing data masquerading as zeros)
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

# Remove extreme outliers on budget (< $1000 likely data errors)
df = df[df['budget'] >= 1000]

# Keep films from 1980 onward (pre-1980 has too few entries to be meaningful)
df = df[df['year'] >= 1980]

after = len(df)
print(f"    Removed {before - after} rows → {after} usable movies remaining")

# ── 6. ENGINEER TARGET FEATURES ──────────────────────────────
print("\n[6] Engineering financial features...")
df['profit']     = df['revenue'] - df['budget']
df['roi']        = (df['profit'] / df['budget']) * 100
df['log_budget'] = np.log1p(df['budget'])
df['log_revenue']= np.log1p(df['revenue'])

# Define SUCCESS:
#   A movie is "successful" if it made at least 2x its budget back (ROI >= 100%)
#   This is a common industry rule-of-thumb (studios want 2x budget to break even
#   accounting for marketing/distribution costs)
df['success'] = (df['roi'] >= 100).astype(int)

success_rate = df['success'].mean() * 100
print(f"    ✓ profit, roi, log_budget, log_revenue computed")
print(f"    ✓ 'success' defined as ROI ≥ 100% → {success_rate:.1f}% of films qualify")

# ── 7. LANGUAGE FLAG ──────────────────────────────────────────
print("\n[7] Creating language feature...")
df['is_english'] = (df['original_language'] == 'en').astype(int)
print(f"    English: {df['is_english'].sum()} | Non-English: {(df['is_english']==0).sum()}")

# ── 8. TIME PERIOD COLUMN ─────────────────────────────────────
print("\n[8] Creating time period buckets...")
def assign_period(year):
    if pd.isna(year):   return np.nan
    if year < 2000:     return 'Pre-2000'
    if year <= 2009:    return '2000-2009'
    return '2010+'

df['period'] = df['year'].apply(assign_period)
print(df['period'].value_counts().sort_index())

# ── 9. GENRE DUMMIES ─────────────────────────────────────────
print("\n[9] Creating genre dummy variables...")
all_genres = ['Action','Adventure','Animation','Comedy','Crime',
              'Documentary','Drama','Family','Fantasy','Horror',
              'Mystery','Romance','Science Fiction','Thriller','War']

for g in all_genres:
    df[f'genre_{g.lower().replace(" ", "_")}'] = df['genres_list'].apply(
        lambda lst: 1 if g in lst else 0
    )
print(f"    ✓ {len(all_genres)} genre dummy columns created")

# ── 10. HANDLE REMAINING NULLS ────────────────────────────────
print("\n[10] Handling remaining nulls...")
# Runtime: fill with median
df['runtime'] = df['runtime'].fillna(df['runtime'].median())

# vote fields: keep as-is (very few nulls)
print("    Nulls remaining in key columns:")
key_cols = ['budget','revenue','runtime','vote_average','vote_count',
            'year','primary_genre','director','lead_actor']
print(df[key_cols].isnull().sum().to_string())

# ── 11. SELECT & SAVE FINAL COLUMNS ──────────────────────────
print("\n[11] Selecting final columns and saving...")

final_cols = [
    # Identifiers
    'id', 'title', 'release_date', 'year', 'month', 'decade',
    'period', 'release_season',
    # Financial
    'budget', 'revenue', 'profit', 'roi',
    'log_budget', 'log_revenue',
    'success',
    # Film attributes
    'runtime', 'original_language', 'is_english',
    'vote_average', 'vote_count', 'popularity',
    # Categorical
    'primary_genre', 'genres_list', 'director', 'lead_actor',
    'companies_list', 'primary_country',
    # Genre dummies
] + [f'genre_{g.lower().replace(" ", "_")}' for g in all_genres]

df_clean = df[final_cols].copy()
df_clean.to_csv('/home/claude/clean_movies.csv', index=False)
print(f"    ✓ Saved clean_movies.csv → {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")

# ── 12. SUMMARY REPORT ───────────────────────────────────────
print("\n" + "="*60)
print("PREPROCESSING COMPLETE — SUMMARY")
print("="*60)
print(f"  Final dataset size:     {df_clean.shape[0]} movies")
print(f"  Year range:             {int(df_clean['year'].min())} – {int(df_clean['year'].max())}")
print(f"  Success rate (ROI≥100): {df_clean['success'].mean()*100:.1f}%")
print(f"  Median budget:          ${df_clean['budget'].median():,.0f}")
print(f"  Median revenue:         ${df_clean['revenue'].median():,.0f}")
print(f"  Median ROI:             {df_clean['roi'].median():.1f}%")
print(f"  Top primary genres:")
print(df_clean['primary_genre'].value_counts().head(8).to_string())
print(f"\n  Period breakdown:")
print(df_clean['period'].value_counts().sort_index().to_string())
print(f"\n  Columns in clean dataset:")
print([c for c in df_clean.columns])
print("\n✅ Ready for Step 2: EDA")
