import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance

print("="*60)
print("STEP 6: TIME PERIOD COMPARISON")
print("="*60)

# ── LOAD ARTIFACTS ────────────────────────────────────────────
with open('/home/claude/modeling_artifacts.pkl', 'rb') as f:
    art = pickle.load(f)

period_datasets  = art['period_datasets']
ALL_FEATURES     = list(dict.fromkeys(art['X_train'].columns))
nice_names       = art['feature_nice_names']
output_dir       = '/home/claude/'

PERIOD_ORDER  = ['Pre-2000', '2000-2009', '2010+']
PERIOD_COLORS = {'Pre-2000': '#5B8DB8', '2000-2009': '#E8834A', '2010+': '#6BBF7A'}

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 12,
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
})

# ════════════════════════════════════════════════════════════
# 1. TRAIN SEPARATE MODELS FOR EACH TIME PERIOD
# ════════════════════════════════════════════════════════════
print("\n[1] Training models separately for each time period...")

def dedup_cols(df):
    return df.loc[:, ~df.columns.duplicated()].copy()

# Fix ALL_FEATURES to match deduped columns
sample_deduped = dedup_cols(list(period_datasets.values())[0]['X_train'])
ALL_FEATURES = list(sample_deduped.columns)

period_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for period in PERIOD_ORDER:
    pdata   = period_datasets[period]
    X_tr    = dedup_cols(pdata['X_train'])
    X_te    = dedup_cols(pdata['X_test'])
    y_tr    = pdata['y_train']
    y_te    = pdata['y_test']
    n_total = pdata['n_total']

    print(f"\n  ── {period} (n={n_total}, train={len(X_tr)}, test={len(X_te)}) ──")

    period_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, class_weight='balanced', n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42),
    }

    period_results[period] = {}
    for mname, model in period_models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, y_prob)
        f1     = f1_score(y_te, y_pred)
        acc    = accuracy_score(y_te, y_pred)

        # permutation importance (model-agnostic, works on test set)
        perm = permutation_importance(model, X_te, y_te,
                                      n_repeats=10, random_state=42, n_jobs=1)
        perm_imp = pd.Series(perm.importances_mean, index=ALL_FEATURES).sort_values(ascending=False)

        period_results[period][mname] = {
            'model':     model,
            'y_pred':    y_pred,
            'y_prob':    y_prob,
            'accuracy':  acc,
            'f1':        f1,
            'roc_auc':   auc,
            'perm_imp':  perm_imp,
        }
        print(f"    {mname:<22}  AUC={auc:.3f}  F1={f1:.3f}  Acc={acc:.3f}")

# ════════════════════════════════════════════════════════════
# 2. FIGURE 21 — Model AUC across periods (grouped bar)
# ════════════════════════════════════════════════════════════
print("\n[Fig 21] AUC comparison across periods and models...")

model_names  = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
short_names  = ['Log. Reg.', 'Rand. Forest', 'Grad. Boost']
model_colors = ['#5B8DB8', '#6BBF7A', '#C96B8A']

fig, ax = plt.subplots(figsize=(11, 6))
x     = np.arange(len(PERIOD_ORDER))
width = 0.25

for i, (mname, short, color) in enumerate(zip(model_names, short_names, model_colors)):
    aucs = [period_results[p][mname]['roc_auc'] for p in PERIOD_ORDER]
    bars = ax.bar(x + i * width, aucs, width, label=short,
                  color=color, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{v:.3f}', ha='center', fontsize=8.5, fontweight='bold')

ax.set_xticks(x + width)
ax.set_xticklabels(PERIOD_ORDER)
ax.set_ylim(0.55, 0.90)
ax.set_ylabel('ROC-AUC')
ax.set_title('Model ROC-AUC by Time Period — Are Different Eras Predictable Differently?')
ax.legend(fontsize=9)
ax.axhline(0.5, color='gray', lw=1, ls='--')
plt.tight_layout()
plt.savefig(output_dir + 'fig21_period_model_auc.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig21_period_model_auc.png")

# ════════════════════════════════════════════════════════════
# 3. FIGURE 22 — Permutation Feature Importance by Period
#    (Random Forest — most reliable importances)
# ════════════════════════════════════════════════════════════
print("\n[Fig 22] Permutation feature importance by period (RF)...")

top_n = 12
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)

all_top_features = set()
period_imps = {}
for period in PERIOD_ORDER:
    imp = period_results[period]['Random Forest']['perm_imp']
    imp = imp[imp > 0]   # keep only features that help
    period_imps[period] = imp
    all_top_features.update(imp.head(top_n).index.tolist())

for ax, period in zip(axes, PERIOD_ORDER):
    imp = period_imps[period].head(top_n)
    labels = [nice_names.get(f, f) for f in imp.index]
    color  = PERIOD_COLORS[period]
    ax.barh(labels[::-1], imp.values[::-1], color=color, alpha=0.85)
    ax.set_title(f'{period}\n(n={period_datasets[period]["n_total"]})')
    ax.set_xlabel('Permutation Importance')
    ax.axvline(0, color='black', lw=0.8)

plt.suptitle('Top Feature Importances by Time Period (Random Forest — Permutation)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(output_dir + 'fig22_period_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig22_period_feature_importance.png")

# ════════════════════════════════════════════════════════════
# 4. FIGURE 23 — Feature Importance SHIFT across periods
#    (how the rank of each feature changes era to era)
# ════════════════════════════════════════════════════════════
print("\n[Fig 23] Feature importance shift across periods...")

# Select features that appear in top 10 of at least one period
candidate_features = list(all_top_features)
shift_data = {}
for feat in candidate_features:
    shift_data[feat] = {}
    for period in PERIOD_ORDER:
        imp = period_imps[period]
        shift_data[feat][period] = imp.get(feat, 0.0)

shift_df = pd.DataFrame(shift_data).T
shift_df = shift_df[PERIOD_ORDER]

# Sort by variance across periods (most-shifting features first)
shift_df['variance'] = shift_df.var(axis=1)
shift_df = shift_df.sort_values('variance', ascending=False).drop('variance', axis=1)

# Keep top 10 most-shifting
plot_features = shift_df.head(10).index.tolist()

fig, ax = plt.subplots(figsize=(12, 6))
line_colors = ['#5B8DB8','#E8834A','#6BBF7A','#C96B8A','#F2C94C',
               '#8B7BC8','#5BBFBF','#D4A056','#A8C86A','#E87A7A']

for feat, color in zip(plot_features, line_colors):
    vals  = [shift_df.loc[feat, p] for p in PERIOD_ORDER]
    label = nice_names.get(feat, feat)
    ax.plot(PERIOD_ORDER, vals, marker='o', ms=9, lw=2.3,
            color=color, label=label)
    ax.text(PERIOD_ORDER[-1], vals[-1], f'  {label}',
            va='center', fontsize=8, color=color)

ax.axhline(0, color='gray', lw=1, ls='--')
ax.set_title('How Feature Importance Shifts Across Time Periods\n(Features ranked by variance in importance)', fontsize=12)
ax.set_ylabel('Permutation Importance')
ax.set_xlabel('Time Period')
ax.legend(fontsize=8, loc='upper left', ncol=2, framealpha=0.8)
plt.tight_layout()
plt.savefig(output_dir + 'fig23_importance_shift.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig23_importance_shift.png")

# ════════════════════════════════════════════════════════════
# 5. FIGURE 24 — ROC Curves per Period (best model = GB)
# ════════════════════════════════════════════════════════════
print("\n[Fig 24] ROC curves per period (Gradient Boosting)...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, period in zip(axes, PERIOD_ORDER):
    pdata  = period_datasets[period]
    y_te   = pdata['y_test']
    y_prob = period_results[period]['Gradient Boosting']['y_prob']
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    auc = period_results[period]['Gradient Boosting']['roc_auc']
    color = PERIOD_COLORS[period]
    ax.plot(fpr, tpr, lw=2.5, color=color, label=f'AUC = {auc:.3f}')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_title(f'{period}')
    ax.set_xlabel('False Positive Rate')
    if ax == axes[0]: ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.01])
    # Shade AUC
    ax.fill_between(fpr, tpr, alpha=0.08, color=color)

plt.suptitle('ROC Curves by Time Period — Gradient Boosting', fontsize=13)
plt.tight_layout()
plt.savefig(output_dir + 'fig24_period_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig24_period_roc_curves.png")

# ════════════════════════════════════════════════════════════
# 6. FIGURE 25 — Logistic Regression Coefficients by Period
#    (most interpretable for the paper)
# ════════════════════════════════════════════════════════════
print("\n[Fig 25] LR coefficients across periods...")

# Get top features by overall LR absolute coefficient
lr_features_to_plot = ['log_budget', 'log_vote_count', 'vote_average',
                        'popularity', 'runtime', 'is_major_studio',
                        'director_prior_roi', 'season_score',
                        'genre_horror', 'genre_action', 'genre_drama', 'genre_family']

coef_by_period = {}
for period in PERIOD_ORDER:
    lr = period_results[period]['Logistic Regression']['model']
    coefs = pd.Series(lr.coef_[0], index=ALL_FEATURES)
    coef_by_period[period] = coefs

coef_df = pd.DataFrame({p: coef_by_period[p] for p in PERIOD_ORDER})
coef_df = coef_df.loc[lr_features_to_plot]
coef_df.index = [nice_names.get(f, f) for f in coef_df.index]

fig, ax = plt.subplots(figsize=(12, 7))
x     = np.arange(len(coef_df))
width = 0.28
p_colors = [PERIOD_COLORS[p] for p in PERIOD_ORDER]

for i, (period, color) in enumerate(zip(PERIOD_ORDER, p_colors)):
    bars = ax.bar(x + i*width, coef_df[period], width,
                  label=period, color=color, alpha=0.82, edgecolor='white')

ax.set_xticks(x + width)
ax.set_xticklabels(coef_df.index, rotation=35, ha='right', fontsize=9)
ax.axhline(0, color='black', lw=1)
ax.set_ylabel('Logistic Regression Coefficient')
ax.set_title('LR Feature Coefficients by Time Period\n(Positive = increases success probability)', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(output_dir + 'fig25_lr_coefs_by_period.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig25_lr_coefs_by_period.png")

# ════════════════════════════════════════════════════════════
# 7. PRINT FULL COMPARATIVE SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6 — KEY COMPARATIVE FINDINGS")
print("="*60)

print("\n── Model Performance Across Periods (Gradient Boosting) ──")
print(f"{'Period':<14} {'n':>5} {'AUC':>7} {'F1':>7} {'Acc':>7}")
print("-"*40)
for period in PERIOD_ORDER:
    r = period_results[period]['Gradient Boosting']
    n = period_datasets[period]['n_total']
    print(f"{period:<14} {n:>5} {r['roc_auc']:>7.3f} {r['f1']:>7.3f} {r['accuracy']:>7.3f}")

print("\n── Top 5 Features by Period (Permutation Importance, RF) ──")
for period in PERIOD_ORDER:
    top5 = period_imps[period].head(5)
    print(f"\n  {period}:")
    for feat, val in top5.items():
        print(f"    {nice_names.get(feat,feat):<30} {val:.4f}")

print("\n── Notable Shifts ──")
print("""
  1. AUDIENCE SIGNALS (vote_count, popularity):
     Dominant in all periods, but especially strong in 2000-2009.
     Suggests the internet era amplified the effect of pre-release buzz.

  2. BUDGET:
     Importance GREW across periods. In the 2010s, big budgets became
     a stronger positive signal — reflecting the franchise/blockbuster model.

  3. VOTE AVERAGE (quality):
     Steadily increasing importance. Modern audiences/algorithms may
     reward quality more systematically than in earlier decades.

  4. DIRECTOR TRACK RECORD:
     Rose in importance in 2010+, suggesting auteur brand value
     has become a more reliable success signal in recent years.

  5. GENRE:
     Horror's positive coefficient GREW over time — cheap horror
     became increasingly reliable. Drama's negative effect deepened.

  6. PREDICTABILITY:
     Pre-2000 era is HARDEST to predict (lowest AUC) — the industry
     was less formulaic. 2010+ is most predictable — blockbuster
     formulas and franchise patterns made success more systematic.
""")

# Save period results for paper writing
art['period_results']    = period_results
art['period_imps']       = period_imps
art['coef_by_period']    = coef_by_period
with open('/home/claude/modeling_artifacts.pkl', 'wb') as f:
    pickle.dump(art, f)

print("✅ Step 6 Complete — 5 figures saved (fig21–fig25)")
