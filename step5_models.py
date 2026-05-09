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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance

print("="*60)
print("STEP 5: MACHINE LEARNING MODELS")
print("="*60)

# ── LOAD ARTIFACTS ────────────────────────────────────────────
with open('/home/claude/modeling_artifacts.pkl', 'rb') as f:
    art = pickle.load(f)

X_train      = art['X_train'].loc[:, ~art['X_train'].columns.duplicated()]
X_test       = art['X_test'].loc[:, ~art['X_test'].columns.duplicated()]
y_train      = art['y_train']
y_test       = art['y_test']
ALL_FEATURES = list(dict.fromkeys(art['X_train'].columns))  # deduplicated col names
output_dir   = '/home/claude/'

PERIOD_ORDER  = ['Pre-2000', '2000-2009', '2010+']
PERIOD_COLORS = {'Pre-2000': '#5B8DB8', '2000-2009': '#E8834A', '2010+': '#6BBF7A'}

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 12,
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
})

# ── DEFINE MODELS ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, random_state=42, class_weight='balanced'
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=20, random_state=42, class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=10,
        random_state=42, class_weight='balanced', n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
}

# ════════════════════════════════════════════════════════════
# 1. TRAIN & EVALUATE ALL MODELS
# ════════════════════════════════════════════════════════════
print("\n[1] Training and evaluating all models...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n  → {name}")
    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    # Cross-val on full training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring='roc_auc', n_jobs=-1)

    metrics = {
        'model':     model,
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob),
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std':  cv_scores.std(),
    }
    results[name] = metrics

    print(f"    Accuracy:  {metrics['accuracy']:.3f}")
    print(f"    Precision: {metrics['precision']:.3f}")
    print(f"    Recall:    {metrics['recall']:.3f}")
    print(f"    F1:        {metrics['f1']:.3f}")
    print(f"    ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"    CV AUC:    {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")

# ── FIGURE 16: Model comparison bar chart ─────────────────────
print("\n[Fig 16] Model comparison...")

metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
model_names  = list(results.keys())
short_names  = ['Log. Reg.', 'Dec. Tree', 'Rand. Forest', 'Grad. Boost']

fig, ax = plt.subplots(figsize=(13, 6))
x      = np.arange(len(metric_names))
width  = 0.2
colors = ['#5B8DB8', '#E8834A', '#6BBF7A', '#C96B8A']

for i, (name, short, color) in enumerate(zip(model_names, short_names, colors)):
    vals = [results[name][m] for m in metric_names]
    bars = ax.bar(x + i * width, vals, width, label=short,
                  color=color, alpha=0.85, edgecolor='white')

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])
ax.set_ylim(0.45, 0.85)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison — Test Set')
ax.legend(loc='lower right', fontsize=9)
ax.axhline(0.542, color='gray', lw=1, ls='--', label='Baseline (majority class)')
ax.text(4.85, 0.548, 'Baseline', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig(output_dir + 'fig16_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig16_model_comparison.png")

# ── FIGURE 17: ROC Curves ─────────────────────────────────────
print("\n[Fig 17] ROC curves...")

fig, ax = plt.subplots(figsize=(8, 7))
for (name, short, color) in zip(model_names, short_names, colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    auc = results[name]['roc_auc']
    ax.plot(fpr, tpr, lw=2.2, color=color, label=f'{short} (AUC={auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.01])
plt.tight_layout()
plt.savefig(output_dir + 'fig17_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig17_roc_curves.png")

# ── FIGURE 18: Confusion matrices ─────────────────────────────
print("\n[Fig 18] Confusion matrices...")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, short, color) in zip(axes, zip(model_names, short_names, colors)):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=['Pred: Fail', 'Pred: Success'],
                yticklabels=['True: Fail', 'True: Success'],
                annot_kws={'size': 11})
    ax.set_title(f'{short}\n(ACC={results[name]["accuracy"]:.3f})')

plt.suptitle('Confusion Matrices (% of True Class)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(output_dir + 'fig18_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig18_confusion_matrices.png")

# ════════════════════════════════════════════════════════════
# 2. FEATURE IMPORTANCE — BEST MODEL (Random Forest)
# ════════════════════════════════════════════════════════════
print("\n[2] Feature importance analysis (Random Forest)...")

rf_model = results['Random Forest']['model']
rf_importances = pd.Series(
    rf_model.feature_importances_, index=ALL_FEATURES
).sort_values(ascending=False)

# Logistic regression coefficients
lr_model = results['Logistic Regression']['model']
lr_coefs = pd.Series(
    np.abs(lr_model.coef_[0]), index=ALL_FEATURES
).sort_values(ascending=False)

print("\n  Top 15 RF Feature Importances:")
print(rf_importances.head(15).round(4).to_string())

print("\n  Top 15 LR Absolute Coefficients:")
print(lr_coefs.head(15).round(4).to_string())

# ── FIGURE 19: Feature importances ────────────────────────────
print("\n[Fig 19] Feature importances...")

top_n = 15
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Random Forest importances
rf_top = rf_importances.head(top_n)
nice_names = {
    'log_budget': 'Log Budget', 'log_vote_count': 'Log Vote Count',
    'vote_average': 'Vote Average', 'popularity': 'Popularity',
    'runtime': 'Runtime', 'director_prior_roi': 'Director Prior ROI',
    'director_film_count': 'Director Film Count', 'season_score': 'Release Season',
    'is_major_studio': 'Major Studio', 'genre_count': 'Genre Count',
    'budget_tier_ord': 'Budget Tier', 'high_awareness': 'High Awareness',
    'is_long_film': 'Long Film (>120min)', 'is_short_film': 'Short Film (<90min)',
    'director_is_established': 'Established Director', 'is_english': 'English Language',
    'genre_horror': 'Genre: Horror', 'genre_drama': 'Genre: Drama',
    'genre_action': 'Genre: Action', 'genre_comedy': 'Genre: Comedy',
    'genre_animation': 'Genre: Animation', 'genre_family': 'Genre: Family',
    'genre_adventure': 'Genre: Adventure', 'genre_thriller': 'Genre: Thriller',
    'genre_crime': 'Genre: Crime', 'genre_romance': 'Genre: Romance',
    'genre_science_fiction': 'Genre: Sci-Fi', 'genre_mystery': 'Genre: Mystery',
    'genre_war': 'Genre: War', 'genre_fantasy': 'Genre: Fantasy',
    'genre_documentary': 'Genre: Documentary',
}
labels_rf = [nice_names.get(f, f) for f in rf_top.index]
axes[0].barh(labels_rf[::-1], rf_top.values[::-1], color='#5B8DB8', alpha=0.85)
axes[0].set_title('Random Forest\nFeature Importances (Top 15)')
axes[0].set_xlabel('Importance')

# Logistic Regression coefficients (signed, not abs)
lr_signed = pd.Series(lr_model.coef_[0], index=ALL_FEATURES).reindex(lr_coefs.head(top_n).index)
labels_lr  = [nice_names.get(f, f) for f in lr_signed.index]
colors_lr  = ['#6BBF7A' if v > 0 else '#C96B8A' for v in lr_signed.values]
axes[1].barh(labels_lr[::-1], lr_signed.values[::-1], color=colors_lr[::-1], alpha=0.85)
axes[1].axvline(0, color='black', lw=1)
axes[1].set_title('Logistic Regression\nCoefficients (Top 15 by |magnitude|)')
axes[1].set_xlabel('Coefficient Value')

from matplotlib.patches import Patch
legend_els = [Patch(facecolor='#6BBF7A', label='Positive (↑ success)'),
              Patch(facecolor='#C96B8A', label='Negative (↓ success)')]
axes[1].legend(handles=legend_els, fontsize=8)

plt.suptitle('Feature Importance — Random Forest & Logistic Regression', fontsize=13)
plt.tight_layout()
plt.savefig(output_dir + 'fig19_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig19_feature_importance.png")

# ════════════════════════════════════════════════════════════
# 3. CROSS-VALIDATION STABILITY CHART
# ════════════════════════════════════════════════════════════
print("\n[Fig 20] Cross-validation scores...")

fig, ax = plt.subplots(figsize=(9, 5))
cv_means = [results[n]['cv_auc_mean'] for n in model_names]
cv_stds  = [results[n]['cv_auc_std']  for n in model_names]

bars = ax.bar(short_names, cv_means, yerr=cv_stds, capsize=6,
              color=colors, alpha=0.85, edgecolor='white',
              error_kw=dict(elinewidth=2, ecolor='black'))
ax.set_ylim(0.55, 0.80)
ax.set_ylabel('ROC-AUC')
ax.set_title('5-Fold Cross-Validation ROC-AUC (Mean ± Std)')
for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + std + 0.003,
            f'{mean:.3f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir + 'fig20_crossval_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved fig20_crossval_scores.png")

# ════════════════════════════════════════════════════════════
# 4. SAVE TRAINED MODELS FOR STEP 6
# ════════════════════════════════════════════════════════════
print("\n[3] Saving trained models...")

art['trained_models']    = {n: results[n]['model'] for n in model_names}
art['model_results']     = results
art['rf_importances']    = rf_importances
art['lr_coefs']          = lr_coefs
art['ALL_FEATURES']      = ALL_FEATURES
art['feature_nice_names']= nice_names

with open('/home/claude/modeling_artifacts.pkl', 'wb') as f:
    pickle.dump(art, f)
print("    ✓ modeling_artifacts.pkl updated with trained models")

# ════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5 RESULTS SUMMARY")
print("="*60)
print(f"\n{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'CV-AUC':>10}")
print("-"*70)
for name, short in zip(model_names, short_names):
    r = results[name]
    print(f"{short:<22} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
          f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['roc_auc']:>6.3f} "
          f"{r['cv_auc_mean']:>6.3f}±{r['cv_auc_std']:.3f}")

best = max(results, key=lambda n: results[n]['roc_auc'])
print(f"\n  Best model: {best} (ROC-AUC = {results[best]['roc_auc']:.3f})")
print(f"  Top 5 features (RF): {', '.join(rf_importances.head(5).index.tolist())}")
print("\n✅ Step 5 Complete — 5 figures saved (fig16–fig20)")
