import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from utils import load_and_prepare_data
from config import RANDOM_STATE, CV_FOLDS, RF_PARAMS, GB_PARAMS, LR_PARAMS

print("=" * 60)
print("CROSS-VALIDATION ANALYSIS")
print("=" * 60)

# Load data using shared utility function
X, y, all_features = load_and_prepare_data('../data/train.csv')

print(f"\nDataset: {len(X)} accounts")
print(f"Features: {len(all_features)}")

# K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Models to evaluate (LR wrapped in Pipeline for automatic scaling)
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(**LR_PARAMS))
    ]),
    'Random Forest': RandomForestClassifier(**RF_PARAMS),
    'Gradient Boosting': GradientBoostingClassifier(**GB_PARAMS)
}

# Store results
results = {}
cv_scores = {}

print("\n" + "=" * 60)
print(f"RUNNING {CV_FOLDS}-FOLD CROSS-VALIDATION")
print("=" * 60)

for name, model in models.items():
    print(f"\n{name}:")

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_scores[name] = scores

    print(f"  Fold scores: {[f'{s:.4f}' for s in scores]}")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std: {scores.std():.4f}")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")

    results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max()
    }

# Create comparison DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('mean', ascending=False)

print("\n" + "=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)
print(results_df.to_string())

# Save results
results_df.to_csv('../results/cross_validation_results.csv')
print("\nSaved: results/cross_validation_results.csv")

# Visualization 1: Box plot
print("\nCreating visualizations...")
plt.figure(figsize=(12, 6))
cv_data = pd.DataFrame(cv_scores)
cv_data.boxplot()
plt.ylabel('Accuracy')
plt.title(f'{CV_FOLDS}-Fold Cross-Validation Results')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/cv_boxplot.png', dpi=300, bbox_inches='tight')
print("Saved: results/cv_boxplot.png")
plt.close()

# Visualization 2: Bar plot with error bars
plt.figure(figsize=(10, 6))
means = [results[name]['mean'] for name in models.keys()]
stds = [results[name]['std'] for name in models.keys()]
x_pos = np.arange(len(models))

plt.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Mean Accuracy with Standard Deviation')
plt.xticks(x_pos, models.keys(), rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../results/cv_barplot.png', dpi=300, bbox_inches='tight')
print("Saved: results/cv_barplot.png")
plt.close()

# Statistical analysis
print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS")
print("=" * 60)

best_model = results_df.index[0]
print(f"\nBest model: {best_model}")
print(f"Mean CV accuracy: {results_df.loc[best_model, 'mean']:.4f}")
print(f"Standard deviation: {results_df.loc[best_model, 'std']:.4f}")
print(f"95% confidence interval: [{results_df.loc[best_model, 'mean'] - 1.96 * results_df.loc[best_model, 'std']:.4f}, "
      f"{results_df.loc[best_model, 'mean'] + 1.96 * results_df.loc[best_model, 'std']:.4f}]")

print("\n" + "=" * 60)
print("CROSS-VALIDATION COMPLETE")
print("=" * 60)