"""
Configuration file for Instagram Fake Account Detection project
Contains all hyperparameters, paths, and settings
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')

# Data paths
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')

# Model save paths
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, 'gradient_boosting_baseline.pkl')
NN_MODEL_PATH = os.path.join(MODELS_DIR, 'neural_network.pt')
RF_TUNED_PATH = os.path.join(MODELS_DIR, 'rf_tuned.pkl')
GB_TUNED_PATH = os.path.join(MODELS_DIR, 'gb_tuned.pkl')
ENSEMBLE_MODEL_PATH = os.path.join(MODELS_DIR, 'ensemble_model.pkl')

# Feature names
ORIGINAL_FEATURES = [
    'profile pic',
    'nums/length username',
    'fullname words',
    'nums/length fullname',
    'name==username',
    'description length',
    'external URL',
    'private',
    '#posts',
    '#followers',
    '#follows'
]

ENGINEERED_FEATURES = [
    'follower_following_ratio',
    'posts_followers_ratio',
    'following_posts_ratio',
    'has_profile_pic',
    'has_external_url',
    'is_private',
    'username_digit_heavy',
    'no_fullname',
    'short_description',
    'high_following'
]

ALL_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Random Forest parameters (best from tuning)
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Gradient Boosting parameters (best from tuning)
GB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.2,
    'max_depth': 5,
    'min_samples_split': 10,
    'random_state': RANDOM_STATE
}

# Logistic Regression parameters
LR_PARAMS = {
    'random_state': RANDOM_STATE,
    'max_iter': 2000
}

# GridSearch parameters for tuning
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

GB_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Results file paths
BEST_PARAMS_JSON = os.path.join(RESULTS_DIR, 'best_hyperparameters.json')
CV_RESULTS_CSV = os.path.join(RESULTS_DIR, 'cross_validation_results.csv')
SHAP_IMPORTANCE_CSV = os.path.join(RESULTS_DIR, 'shap_feature_importance.csv')

# Project metadata
PROJECT_NAME = "Instagram Fake Account Detection"
PROJECT_VERSION = "1.0.0"
AUTHOR = "Vladyslav Danyliuk"
UNIVERSITY = "University of Greenwich"
DEGREE = "BSc Software Engineering"
YEAR = "Year 3"

# Performance thresholds
BASELINE_ACCURACY = 0.8966
TARGET_ACCURACY = 0.94
ACCEPTABLE_FPR = 0.15  # False Positive Rate
ACCEPTABLE_FNR = 0.10  # False Negative Rate

# Feature engineering thresholds
USERNAME_DIGIT_THRESHOLD = 0.5
SHORT_DESCRIPTION_THRESHOLD = 20
HIGH_FOLLOWING_THRESHOLD = 1000

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Display settings
PANDAS_DISPLAY_MAX_ROWS = 100
PANDAS_DISPLAY_MAX_COLS = 50

# Reproducibility
SEED = RANDOM_STATE

def setup_project_structure():
    """
    Create necessary directories if they don't exist
    """
    directories = [DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Project structure verified.")

def print_config():
    """
    Print current configuration
    """
    print("=" * 60)
    print(f"{PROJECT_NAME} v{PROJECT_VERSION}")
    print("=" * 60)
    print(f"Author: {AUTHOR}")
    print(f"University: {UNIVERSITY}")
    print(f"Degree: {DEGREE} - {YEAR}")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Random State: {RANDOM_STATE}")
    print(f"  Test Size: {TEST_SIZE}")
    print(f"  CV Folds: {CV_FOLDS}")
    print(f"  Total Features: {len(ALL_FEATURES)}")
    print(f"  Target Accuracy: {TARGET_ACCURACY:.2%}")
    print("=" * 60)

if __name__ == "__main__":
    setup_project_structure()
    print_config()