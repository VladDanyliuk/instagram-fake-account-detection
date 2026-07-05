"""
Dataset Download Script for Instagram Fake Account Detection project

This script automatically downloads the dataset from Kaggle if it's not present.
Run this script before running any other notebooks/scripts.

Requirements:
    - Kaggle API credentials configured (~/.kaggle/kaggle.json)
    - kaggle package installed (pip install kaggle)

Usage:
    python download_dataset.py
"""

import os
import sys

# Import paths from project configuration
from config import DATA_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH

# Kaggle dataset identifier
KAGGLE_DATASET = "free4ever1/instagram-fake-spammer-genuine-accounts"


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("=" * 60)
        print("ERROR: Kaggle API credentials not found!")
        print("=" * 60)
        print("\nTo configure Kaggle API:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. Download kaggle.json")
        print("5. Place it in ~/.kaggle/kaggle.json")
        print("6. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("=" * 60)
        return False
    return True


def dataset_exists():
    """Check if dataset files already exist."""
    return os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH)


def download_dataset():
    """Download dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("Run: pip install kaggle")
        return False

    print("Initializing Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    print(f"Destination: {DATA_DIR}")

    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download and unzip
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=DATA_DIR,
        unzip=True
    )

    print("Download complete!")
    return True


def verify_download():
    """Verify that all required files are present."""
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"WARNING: train.csv not found at {TRAIN_DATA_PATH}")
        return False
    if not os.path.exists(TEST_DATA_PATH):
        print(f"WARNING: test.csv not found at {TEST_DATA_PATH}")
        return False

    # Print file info
    import pandas as pd
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples:  {len(test_df)}")
    print(f"Features:      {len(train_df.columns)}")
    print("=" * 60)

    return True


def main():
    """Main function to download and verify dataset."""
    print("=" * 60)
    print("Instagram Fake Account Detection - Dataset Setup")
    print("=" * 60)

    # Check if dataset already exists
    if dataset_exists():
        print("\nDataset already exists!")
        verify_download()
        print("\nYou're ready to run the project.")
        return 0

    print("\nDataset not found. Downloading from Kaggle...")

    # Check Kaggle credentials
    if not check_kaggle_credentials():
        return 1

    # Download dataset
    if not download_dataset():
        return 1

    # Verify download
    if not verify_download():
        print("\nDownload may have failed. Please check manually.")
        return 1

    print("\nDataset downloaded successfully!")
    print("You're ready to run the project.")
    return 0


if __name__ == "__main__":
    sys.exit(main())