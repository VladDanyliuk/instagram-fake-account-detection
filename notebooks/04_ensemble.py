import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import pickle
from utils import load_and_prepare_data
from config import RANDOM_STATE, TEST_SIZE

print("=" * 60)
print("ENSEMBLE METHODS")
print("=" * 60)

# Load data using shared utility function
X, y, all_features = load_and_prepare_data('../data/train.csv')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# Individual models
print("\nTraining individual models...")

# Model 1: Logistic Regression (uses scaled data)
lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
print(f"Logistic Regression: {lr_acc:.4f}")

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest: {rf_acc:.4f}")

# Model 3: Extra Trees
et = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
et.fit(X_train, y_train)
et_acc = accuracy_score(y_test, et.predict(X_test))
print(f"Extra Trees: {et_acc:.4f}")

# Model 4: Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=RANDOM_STATE
)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"Gradient Boosting: {gb_acc:.4f}")

# ENSEMBLE: Voting Classifier (Soft Voting)
# Using Pipeline for LR to handle scaling automatically
print("\n" + "=" * 60)
print("ENSEMBLE - VOTING CLASSIFIER")
print("=" * 60)

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
])

voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr_pipeline),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                          max_depth=5, random_state=RANDOM_STATE))
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)

y_pred_ensemble = voting_clf.predict(X_test)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)

print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ensemble))

# Save ensemble model
with open('../models/ensemble_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)
print("\nModel saved: models/ensemble_model.pkl")

print("\n" + "=" * 60)
print("FINAL COMPARISON:")
print("=" * 60)
print(f"Logistic Regression: {lr_acc:.4f}")
print(f"Random Forest: {rf_acc:.4f}")
print(f"Extra Trees: {et_acc:.4f}")
print(f"Gradient Boosting: {gb_acc:.4f}")
print(f"ENSEMBLE: {ensemble_acc:.4f}")
print("=" * 60)