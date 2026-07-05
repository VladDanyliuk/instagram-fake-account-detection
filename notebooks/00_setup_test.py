import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
from config import TRAIN_DATA_PATH, ALL_FEATURES

print("=" * 60)
print("SETUP VERIFICATION")
print("=" * 60)

# Versions
print("\nVersions:")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"torch: {torch.__version__}")

# Device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"\nDevice: {device}")

# Dataset
df = pd.read_csv(TRAIN_DATA_PATH)
print(f"\nDataset: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"Features configured: {len(ALL_FEATURES)}")
print(f"Fake: {df['fake'].sum():,} ({df['fake'].sum()/len(df)*100:.1f}%)")
print(f"Real: {(~df['fake'].astype(bool)).sum():,}")

print("\nAll systems ready!")
print("=" * 60)