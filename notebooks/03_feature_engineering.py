import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils import load_and_prepare_data
from config import ORIGINAL_FEATURES, RANDOM_STATE, TEST_SIZE

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Load data using shared utility function
X, y, all_features = load_and_prepare_data('../data/train.csv')

print(f"Original features: {len(ORIGINAL_FEATURES)}")
print(f"Total features: {len(all_features)}")
print(f"New features added: {len(all_features) - len(ORIGINAL_FEATURES)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting with more features
print("\nTraining Gradient Boosting with engineered features...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=RANDOM_STATE
)
gb.fit(X_train_scaled, y_train)

y_pred = gb.predict(X_test_scaled)
new_acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("RESULTS WITH FEATURE ENGINEERING")
print("=" * 60)
print(f"Accuracy: {new_acc:.4f}")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

print("\n" + "=" * 60)
print("COMPARISON:")
print(f"Baseline (Gradient Boosting): 0.8966")
print(f"Neural Network: 0.9224")
print(f"With Feature Engineering: {new_acc:.4f}")
print("=" * 60)