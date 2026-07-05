import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
from utils import load_and_prepare_data
from config import (RANDOM_STATE, TEST_SIZE, CV_FOLDS,
                    RF_PARAM_GRID, GB_PARAM_GRID)

print("=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

# Load data using shared utility function
X, y, all_features = load_and_prepare_data('../data/train.csv')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\nDataset: {len(X)} accounts")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# 1. Random Forest Tuning
print("\n" + "=" * 60)
print("TUNING RANDOM FOREST")
print("=" * 60)

print("Parameter grid:")
for param, values in RF_PARAM_GRID.items():
    print(f"  {param}: {values}")

rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_grid = GridSearchCV(
    rf,
    RF_PARAM_GRID,
    cv=CV_FOLDS,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearch (this may take 5-10 minutes)...")
rf_grid.fit(X_train, y_train)

print("\nBest parameters:")
print(rf_grid.best_params_)
print(f"Best CV score: {rf_grid.best_score_:.4f}")

# Evaluate on test
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
rf_test_acc = accuracy_score(y_test, y_pred_rf)
print(f"Test accuracy: {rf_test_acc:.4f}")

# 2. Gradient Boosting Tuning
print("\n" + "=" * 60)
print("TUNING GRADIENT BOOSTING")
print("=" * 60)

print("Parameter grid:")
for param, values in GB_PARAM_GRID.items():
    print(f"  {param}: {values}")

gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
gb_grid = GridSearchCV(
    gb,
    GB_PARAM_GRID,
    cv=CV_FOLDS,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearch (this may take 10-15 minutes)...")
gb_grid.fit(X_train, y_train)

print("\nBest parameters:")
print(gb_grid.best_params_)
print(f"Best CV score: {gb_grid.best_score_:.4f}")

# Evaluate on test
gb_best = gb_grid.best_estimator_
y_pred_gb = gb_best.predict(X_test)
gb_test_acc = accuracy_score(y_test, y_pred_gb)
print(f"Test accuracy: {gb_test_acc:.4f}")

# Save best models
with open('../models/rf_tuned.pkl', 'wb') as f:
    pickle.dump(rf_best, f)
print("\nSaved: models/rf_tuned.pkl")

with open('../models/gb_tuned.pkl', 'wb') as f:
    pickle.dump(gb_best, f)
print("Saved: models/gb_tuned.pkl")

# Save best parameters
best_params = {
    'random_forest': {
        'params': rf_grid.best_params_,
        'cv_score': float(rf_grid.best_score_),
        'test_accuracy': float(rf_test_acc)
    },
    'gradient_boosting': {
        'params': gb_grid.best_params_,
        'cv_score': float(gb_grid.best_score_),
        'test_accuracy': float(gb_test_acc)
    }
}

with open('../results/best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=4)
print("Saved: results/best_hyperparameters.json")

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING COMPLETE")
print("=" * 60)
print(f"Random Forest - CV: {rf_grid.best_score_:.4f}, Test: {rf_test_acc:.4f}")
print(f"Gradient Boosting - CV: {gb_grid.best_score_:.4f}, Test: {gb_test_acc:.4f}")
print("=" * 60)