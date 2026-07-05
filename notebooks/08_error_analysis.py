import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from utils import load_and_prepare_data
from config import RANDOM_STATE, TEST_SIZE, RF_PARAMS

print("=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

# Load data using shared utility function
X, y, all_features = load_and_prepare_data('../data/train.csv')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\nDataset: {len(X)} accounts")
print(f"Test set: {len(X_test)} accounts")

# Train best model (Random Forest with tuned params from config)
print("\nTraining Random Forest...")
rf = RandomForestClassifier(**RF_PARAMS)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# 1. Confusion Matrix
print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print(f"\n{cm}")

# Detailed breakdown
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (Real correctly identified): {tn}")
print(f"False Positives (Real misclassified as Fake): {fp}")
print(f"False Negatives (Fake misclassified as Real): {fn}")
print(f"True Positives (Fake correctly identified): {tp}")

print(f"\nFalse Positive Rate: {fp / (fp + tn):.4f}")
print(f"False Negative Rate: {fn / (fn + tp):.4f}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.savefig('../results/confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/confusion_matrix_detailed.png")
plt.close()

# 2. ROC Curve
print("\n" + "=" * 60)
print("ROC CURVE")
print("=" * 60)

fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"AUC-ROC: {roc_auc:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/roc_curve.png', dpi=300, bbox_inches='tight')
print("Saved: results/roc_curve.png")
plt.close()

# 3. Precision-Recall Curve
print("\n" + "=" * 60)
print("PRECISION-RECALL CURVE")
print("=" * 60)

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print(f"AUC-PR: {pr_auc:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("Saved: results/precision_recall_curve.png")
plt.close()

# 4. Error Analysis - Misclassified samples
print("\n" + "=" * 60)
print("MISCLASSIFIED SAMPLES ANALYSIS")
print("=" * 60)

# Get misclassified indices
X_test_df = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
misclassified = X_test_df[y_pred != y_test_reset]

print(f"\nTotal misclassified: {len(misclassified)}")
print(f"False Positives: {fp} (Real predicted as Fake)")
print(f"False Negatives: {fn} (Fake predicted as Real)")

# Analyze misclassified samples
if len(misclassified) > 0:
    print("\nMisclassified samples characteristics:")

    # False Positives (Real predicted as Fake)
    fp_indices = (y_test_reset == 0) & (y_pred == 1)
    if fp_indices.sum() > 0:
        print("\nFalse Positives (Real accounts misclassified as Fake):")
        fp_samples = X_test_df[fp_indices]
        print(fp_samples[['#followers', '#follows', '#posts', 'follower_following_ratio']].describe())

    # False Negatives (Fake predicted as Real)
    fn_indices = (y_test_reset == 1) & (y_pred == 0)
    if fn_indices.sum() > 0:
        print("\nFalse Negatives (Fake accounts misclassified as Real):")
        fn_samples = X_test_df[fn_indices]
        print(fn_samples[['#followers', '#follows', '#posts', 'follower_following_ratio']].describe())

# 5. Prediction confidence distribution
print("\n" + "=" * 60)
print("PREDICTION CONFIDENCE")
print("=" * 60)

correct_mask = y_pred == y_test_reset
incorrect_mask = ~correct_mask

print(f"Correct predictions - mean confidence: {y_pred_proba[correct_mask].mean():.4f}")
print(f"Incorrect predictions - mean confidence: {y_pred_proba[incorrect_mask].mean():.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_pred_proba[correct_mask], bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
plt.hist(y_pred_proba[incorrect_mask], bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
plt.xlabel('Prediction Probability')
plt.ylabel('Count')
plt.title('Prediction Confidence Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot([y_pred_proba[correct_mask], y_pred_proba[incorrect_mask]],
            labels=['Correct', 'Incorrect'])
plt.ylabel('Prediction Probability')
plt.title('Prediction Confidence Comparison')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/prediction_confidence.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/prediction_confidence.png")
plt.close()

print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)
print(f"Total errors: {fp + fn} out of {len(y_test)}")
print(f"Error rate: {(fp + fn) / len(y_test) * 100:.2f}%")
print(f"Accuracy: {(tp + tn) / len(y_test) * 100:.2f}%")
print("=" * 60)