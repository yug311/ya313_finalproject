import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import ast
import warnings
warnings.filterwarnings('ignore')

# ── SETUP ─────────────────────────────────────────────────────
df = pd.read_csv('/home/claude/clean_movies.csv')
df['genres_list'] = df['genres_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

PERIOD_ORDER  = ['Pre-2000', '2000-2009', '2010+']
PERIOD_COLORS = {'Pre-2000': '#5B8DB8', '2000-2009': '#E8834A', '2010+': '#6BBF7A'}
PALETTE       = ['#5B8DB8', '#E8834A', '#6BBF7A', '#C96B8A', '#F2C94C', '#8B7BC8']

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
})

output_dir = '/home/claude/'
print("="*60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

# ════════════════════════════════════════════════════════════
# FIGURE 1 — Revenue, Budget & Profit Trends Over Time
# ════════════════════════════════════════════════════════════
print("\n[Fig 1] Revenue, Budget & Profit trends over time...")

yearly = df.groupby('year').agg(
    median_budget  =('budget',  'median'),
    median_revenue =('revenue', 'median'),
    median_profit  =('profit',  'median'),
    count          =('title',   'count')
).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(12, 9))

ax = axes[0]
ax.plot(yearly['year'], yearly['median_budget']  / 1e6, label='Budget',  color='#5B8DB8', lw=2.5, marker='o', ms=4)
ax.plot(yearly['year'], yearly['median_revenue'] / 1e6, label='Revenue', color='#6BBF7A', lw=2.5, marker='o', ms=4)
ax.plot(yearly['year'], yearly['median_profit']  / 1e6, label='Profit',  color='#E8834A', lw=2.5, marker='o', ms=4)
ax.set_title('Median Budget, Revenue & Profit Over Time (1980–2016)')
ax.set_ylabel('USD (Millions)')
ax.legend()
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax.axhline(0, color='gray', lw=0.8, ls='--')

ax2 = axes[1]
ax2.bar(yearly['year'], yearly['count'], color='#9B9BE8', alpha=0.8)
ax2.set_title('Number of Films Per Year (with Financial Data)')
ax2.set_ylabel('Film Count')
ax2.set_xlabel('Year')

plt.tight_layout()
plt.savefig(output_dir + 'fig1_trends_over_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig1_trends_over_time.png")

# ════════════════════════════════════════════════════════════
# FIGURE 2 — Distribution of Budget, Revenue, ROI
# ════════════════════════════════════════════════════════════
print("\n[Fig 2] Financial distributions...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, col, label, color in zip(
    axes,
    ['log_budget', 'log_revenue', 'roi'],
    ['Log Budget', 'Log Revenue', 'ROI (%)'],
    ['#5B8DB8', '#6BBF7A', '#E8834A']
):
    data = df[col].dropna()
    if col == 'roi':
        data = data.clip(-200, 1000)   # clip extreme outliers for display
    ax.hist(data, bins=50, color=color, alpha=0.85, edgecolor='white', lw=0.4)
    ax.axvline(data.median(), color='black', lw=1.8, ls='--', label=f'Median: {data.median():.1f}')
    ax.set_title(f'Distribution of {label}')
    ax.set_xlabel(label)
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)

plt.suptitle('Financial Distributions (Clipped for Display)', y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig(output_dir + 'fig2_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig2_distributions.png")

# ════════════════════════════════════════════════════════════
# FIGURE 3 — Genre Performance: Revenue & ROI
# ════════════════════════════════════════════════════════════
print("\n[Fig 3] Genre performance...")

genre_cols = [c for c in df.columns if c.startswith('genre_')]
genre_stats = []
for col in genre_cols:
    name = col.replace('genre_', '').replace('_', ' ').title()
    sub  = df[df[col] == 1]
    if len(sub) < 20: continue
    genre_stats.append({
        'genre':          name,
        'count':          len(sub),
        'median_revenue': sub['revenue'].median() / 1e6,
        'median_roi':     sub['roi'].median(),
        'success_rate':   sub['success'].mean() * 100,
    })
genre_df = pd.DataFrame(genre_stats).sort_values('median_revenue', ascending=True)

fig, axes = plt.subplots(1, 3, figsize=(17, 6))

# Revenue by genre
axes[0].barh(genre_df['genre'], genre_df['median_revenue'], color='#5B8DB8', alpha=0.85)
axes[0].set_title('Median Revenue by Genre')
axes[0].set_xlabel('USD (Millions)')
axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:.0f}M'))

# ROI by genre
genre_roi = genre_df.sort_values('median_roi', ascending=True)
axes[1].barh(genre_roi['genre'], genre_roi['median_roi'], color='#E8834A', alpha=0.85)
axes[1].set_title('Median ROI by Genre')
axes[1].set_xlabel('ROI (%)')
axes[1].axvline(100, color='black', lw=1.2, ls='--', label='100% ROI threshold')
axes[1].legend(fontsize=9)

# Success rate by genre
genre_sr = genre_df.sort_values('success_rate', ascending=True)
axes[2].barh(genre_sr['genre'], genre_sr['success_rate'], color='#6BBF7A', alpha=0.85)
axes[2].set_title('Success Rate by Genre (ROI ≥ 100%)')
axes[2].set_xlabel('Success Rate (%)')
axes[2].axvline(54.2, color='black', lw=1.2, ls='--', label='Overall avg (54.2%)')
axes[2].legend(fontsize=9)

plt.suptitle('Genre Performance Comparison', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(output_dir + 'fig3_genre_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig3_genre_performance.png")

# ════════════════════════════════════════════════════════════
# FIGURE 4 — Correlation Heatmap
# ════════════════════════════════════════════════════════════
print("\n[Fig 4] Correlation heatmap...")

corr_cols = ['budget', 'revenue', 'profit', 'roi', 'runtime',
             'vote_average', 'vote_count', 'popularity', 'success', 'is_english']
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, annot_kws={'size': 9})
ax.set_title('Correlation Heatmap — Key Variables', pad=12)
plt.tight_layout()
plt.savefig(output_dir + 'fig4_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig4_correlation_heatmap.png")

# ════════════════════════════════════════════════════════════
# FIGURE 5 — Budget vs Revenue Scatter (by Period)
# ════════════════════════════════════════════════════════════
print("\n[Fig 5] Budget vs Revenue scatter by period...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, period in zip(axes, PERIOD_ORDER):
    sub = df[df['period'] == period]
    color = PERIOD_COLORS[period]
    ax.scatter(sub['log_budget'], sub['log_revenue'],
               alpha=0.35, s=18, color=color, edgecolors='none')
    # Trend line
    m, b = np.polyfit(sub['log_budget'].dropna(), sub['log_revenue'].dropna(), 1)
    x_line = np.linspace(sub['log_budget'].min(), sub['log_budget'].max(), 100)
    ax.plot(x_line, m * x_line + b, color='black', lw=1.8, ls='--')
    ax.set_title(f'{period}\n(n={len(sub)})')
    ax.set_xlabel('Log Budget')
    if ax == axes[0]: ax.set_ylabel('Log Revenue')
    # Annotate R²
    from numpy.polynomial.polynomial import polyfit
    corr_val = sub[['log_budget','log_revenue']].dropna().corr().iloc[0,1]
    ax.text(0.05, 0.93, f'r = {corr_val:.2f}', transform=ax.transAxes,
            fontsize=10, color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.suptitle('Log Budget vs Log Revenue by Time Period', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(output_dir + 'fig5_budget_vs_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig5_budget_vs_revenue.png")

# ════════════════════════════════════════════════════════════
# FIGURE 6 — Vote Average vs Revenue (Ratings vs Success)
# ════════════════════════════════════════════════════════════
print("\n[Fig 6] Ratings vs Revenue...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: vote_average vs revenue (log)
ax = axes[0]
colors = df['success'].map({1: '#6BBF7A', 0: '#C96B8A'})
ax.scatter(df['vote_average'], df['log_revenue'], c=colors, alpha=0.3, s=15, edgecolors='none')
ax.set_xlabel('IMDb Vote Average')
ax.set_ylabel('Log Revenue')
ax.set_title('Vote Average vs Log Revenue')
from matplotlib.patches import Patch
legend_els = [Patch(facecolor='#6BBF7A', label='Successful (ROI≥100%)'),
              Patch(facecolor='#C96B8A', label='Unsuccessful')]
ax.legend(handles=legend_els, fontsize=9)

# Box: vote_average by success group
ax2 = axes[1]
df.boxplot(column='vote_average', by='success', ax=ax2,
           boxprops=dict(color='#5B8DB8'),
           medianprops=dict(color='#E8834A', lw=2),
           whiskerprops=dict(color='#5B8DB8'),
           capprops=dict(color='#5B8DB8'),
           flierprops=dict(marker='o', alpha=0.3, ms=3))
ax2.set_title('Vote Average by Success')
ax2.set_xlabel('Success (0=No, 1=Yes)')
ax2.set_ylabel('Vote Average')
plt.suptitle('')

plt.suptitle('Audience Ratings vs Financial Success', fontsize=13)
plt.tight_layout()
plt.savefig(output_dir + 'fig6_ratings_vs_success.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig6_ratings_vs_success.png")

# ════════════════════════════════════════════════════════════
# FIGURE 7 — Genre Share Over Time (Stacked Area)
# ════════════════════════════════════════════════════════════
print("\n[Fig 7] Genre share over time...")

top_genres = ['Action','Comedy','Drama','Adventure','Horror','Thriller','Animation']
decade_genre = []
for decade in sorted(df['decade'].dropna().unique()):
    sub = df[df['decade'] == decade]
    row = {'decade': int(decade)}
    for g in top_genres:
        col = f"genre_{g.lower().replace(' ','_')}"
        row[g] = sub[col].sum() / len(sub) * 100
    decade_genre.append(row)
dg = pd.DataFrame(decade_genre).set_index('decade')

fig, ax = plt.subplots(figsize=(12, 6))
dg.plot(kind='bar', stacked=True, ax=ax,
        color=['#5B8DB8','#E8834A','#6BBF7A','#C96B8A','#F2C94C','#8B7BC8','#5BBFBF'],
        width=0.75, edgecolor='white', linewidth=0.5)
ax.set_title('Genre Composition by Decade (% of Films in Dataset)', fontsize=13)
ax.set_xlabel('Decade')
ax.set_ylabel('Share of Films (%)')
ax.set_xticklabels([str(int(d)) + 's' for d in dg.index], rotation=0)
ax.legend(loc='upper left', fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(output_dir + 'fig7_genre_over_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig7_genre_over_time.png")

# ════════════════════════════════════════════════════════════
# FIGURE 8 — Success Rate & Median ROI by Period
# ════════════════════════════════════════════════════════════
print("\n[Fig 8] Success rate and ROI by period...")

period_stats = df.groupby('period').agg(
    success_rate =('success', 'mean'),
    median_roi   =('roi',     'median'),
    median_budget=('budget',  'median'),
    median_revenue=('revenue','median'),
    count        =('title',   'count')
).reindex(PERIOD_ORDER)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = [
    ('success_rate', 'Success Rate', '%', lambda x: f'{x*100:.1f}%'),
    ('median_roi',   'Median ROI',   '%', lambda x: f'{x:.0f}%'),
    ('median_budget','Median Budget','$M', lambda x: f'${x/1e6:.0f}M'),
]
for ax, (col, label, unit, fmt) in zip(axes, metrics):
    vals   = period_stats[col]
    colors = [PERIOD_COLORS[p] for p in PERIOD_ORDER]
    bars   = ax.bar(PERIOD_ORDER, vals if col != 'success_rate' else vals * 100,
                    color=colors, alpha=0.85, width=0.5)
    ax.set_title(f'{label} by Time Period')
    ax.set_ylabel(f'{label} ({unit})')
    for bar, v in zip(bars, vals):
        disp = v * 100 if col == 'success_rate' else v
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + disp*0.01,
                fmt(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Key Performance Metrics by Time Period', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(output_dir + 'fig8_period_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig8_period_comparison.png")

# ════════════════════════════════════════════════════════════
# FIGURE 9 — Release Season vs Success
# ════════════════════════════════════════════════════════════
print("\n[Fig 9] Release season analysis...")

season_stats = df.groupby('release_season').agg(
    success_rate =('success', 'mean'),
    median_revenue=('revenue','median'),
    count        =('title',  'count')
).reset_index().sort_values('median_revenue', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
season_colors = {'Summer':'#F2C94C','Holiday':'#C96B8A','Spring':'#6BBF7A','Other':'#9B9BE8'}

for ax, col, label in zip(axes,
    ['median_revenue', 'success_rate'],
    ['Median Revenue ($M)', 'Success Rate (%)']):
    vals = season_stats[col] * (1 if col == 'median_revenue' else 100)
    if col == 'median_revenue': vals = vals / 1e6
    colors = [season_colors.get(s, '#aaa') for s in season_stats['release_season']]
    bars = ax.bar(season_stats['release_season'], vals, color=colors, alpha=0.85, width=0.5)
    ax.set_title(f'{label} by Release Season')
    ax.set_ylabel(label)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + v*0.01,
                f'${v:.0f}M' if col == 'median_revenue' else f'{v:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Release Season Impact on Performance', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(output_dir + 'fig9_release_season.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig9_release_season.png")

# ════════════════════════════════════════════════════════════
# PRINT KEY EDA FINDINGS
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("KEY EDA FINDINGS")
print("="*60)

print("\n--- Correlation with Revenue ---")
corr_rev = df[['budget','vote_average','vote_count','popularity','runtime','success']].corrwith(df['revenue'])
print(corr_rev.sort_values(ascending=False).round(3).to_string())

print("\n--- Correlation with Success (ROI≥100%) ---")
corr_suc = df[['budget','revenue','vote_average','vote_count','popularity','runtime','is_english']].corrwith(df['success'])
print(corr_suc.sort_values(ascending=False).round(3).to_string())

print("\n--- Period Stats Table ---")
print(period_stats[['count','success_rate','median_roi','median_budget','median_revenue']].round(2).to_string())

print("\n--- Top 5 Most Profitable Genres (Median ROI) ---")
print(genre_df[['genre','median_roi','success_rate','count']].sort_values('median_roi', ascending=False).head(5).to_string(index=False))

print("\n✅ Step 2 Complete — 9 figures saved")
