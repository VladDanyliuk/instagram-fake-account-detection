import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import TRAIN_DATA_PATH

# Load data
df = pd.read_csv(TRAIN_DATA_PATH)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 1. Basic information
print("\n1. Dataset Info:")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# 2. Statistics
print("\n2. Statistics:")
print(df.describe())

# 3. Class distribution
print("\n3. Class Distribution:")
print(df['fake'].value_counts())

# 4. Visualization - Class distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='fake')
plt.title('Fake vs Real Accounts')
plt.xlabel('0=Real, 1=Fake')
plt.ylabel('Count')
plt.savefig('../results/01_class_distribution.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/01_class_distribution.png")
plt.show()

# 5. Visualization - Features comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Followers
df.boxplot(column='#followers', by='fake', ax=axes[0,0])
axes[0,0].set_title('Followers Distribution')
axes[0,0].set_xlabel('Account Type')

# Follows
df.boxplot(column='#follows', by='fake', ax=axes[0,1])
axes[0,1].set_title('Follows Distribution')
axes[0,1].set_xlabel('Account Type')

# Posts
df.boxplot(column='#posts', by='fake', ax=axes[1,0])
axes[1,0].set_title('Posts Distribution')
axes[1,0].set_xlabel('Account Type')

# Username length
df.boxplot(column='nums/length username', by='fake', ax=axes[1,1])
axes[1,1].set_title('Username Digits Ratio')
axes[1,1].set_xlabel('Account Type')

plt.tight_layout()
plt.savefig('../results/02_features_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/02_features_comparison.png")
plt.show()

print("\n" + "=" * 60)
print("EDA COMPLETE!")
print("=" * 60)