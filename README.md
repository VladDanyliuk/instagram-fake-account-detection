# Instagram Fake Account Detection

A machine learning project for detecting fake Instagram accounts using advanced ensemble methods and explainable AI.

**Author:** Vladyslav Danyliuk  
**Student ID:** 001348956  
**University:** University of Greenwich  
**Degree:** BSc Software Engineering, Year 3  
**Academic Year:** 2025/2026

---

## Project Overview

This project implements a comprehensive machine learning system to detect fake Instagram accounts using metadata features. The system achieves up to **93.97% accuracy** on individual train/test splits, with cross-validation confirming **93.23% ± 1.01%** (Random Forest), and provides explainable predictions using SHAP analysis.

### Key Features

- **Advanced Feature Engineering**: 21 features including engineered ratios and binary indicators
- **Ensemble Learning**: Combines multiple models for optimal performance
- **Hyperparameter Tuning**: GridSearch optimization for best parameters
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Explainable AI**: SHAP analysis for model interpretability
- **Comprehensive Testing**: Unit tests for data processing and model training

---

## Results

**Single split results (development evaluation):**

| Model | Accuracy | Improvement |
|-------|----------|-------------|
| Baseline (Gradient Boosting) | 89.66% | - |
| Random Forest | 90.52% | +0.86% |
| Neural Network (PyTorch) | 92.24% | +2.58% |
| Logistic Regression | 93.97% | +4.31% |
| **Ensemble (Final)** | **93.97%** | **+4.31%** |

**5-Fold Cross-Validation Results (robust evaluation):**

| Model | CV Mean | CV Std | CV Min | CV Max |
|-------|---------|--------|--------|--------|
| **Random Forest** | **93.23%** | **±1.01%** | 92.17% | 94.78% |
| Gradient Boosting | 91.67% | ±2.10% | 87.83% | 93.91% |
| Logistic Regression | 90.80% | ±3.17% | 87.83% | 96.52% |

Random Forest ranks first on cross-validation despite Logistic Regression and the Ensemble matching it on the single development split. This is because CV averages over 5 different train/test partitions, making it a more reliable indicator of generalisation: RF's lower variance (±1.01%) compared to LR (±3.17%) confirms its superior stability.

**Held-out Test Set (confusion matrix — Random Forest):**
- Test Accuracy: **91.38%** (106/116 correct predictions)
  - True Real: 51 correct, 7 misclassified as fake (FPR 12.07%)
  - True Fake: 55 correct, 3 misclassified as real (FNR 5.17%)
- This result is consistent with the CV estimate of 93.23% ± 1.01%; a score of 91.38% falls within the expected range of normal split-to-split variation.

**Additional Metrics:**
- AUC-ROC: 98.13%
- False Positive Rate: 12.07%
- False Negative Rate: 5.17%

---

## Project Structure
```
Instagram_Fake_Detection/
|-- data/                          # Dataset files
|   |-- train.csv                  # Training data (576 accounts: 288 real + 288 fake)
|   +-- test.csv                   # Test data (120 accounts; 116 evaluated post preprocessing)
|-- models/                        # Saved models
|   |-- gradient_boosting_baseline.pkl  # Baseline model
|   |-- neural_network.pt          # PyTorch neural network
|   |-- rf_tuned.pkl               # Tuned Random Forest
|   |-- gb_tuned.pkl               # Tuned Gradient Boosting
|   +-- ensemble_model.pkl         # Final ensemble model
|-- notebooks/                     # Analysis scripts
|   |-- 00_setup_test.py           # Environment verification
|   |-- 01_EDA.py                  # Exploratory data analysis
|   |-- 02_baseline.py             # Baseline models
|   |-- 03_feature_engineering.py  # Feature engineering
|   |-- 04_ensemble.py             # Ensemble methods
|   |-- 05_shap.py                 # SHAP explainability
|   |-- 06_hyperparameter_tuning.py # GridSearch optimization
|   |-- 07_cross_validation.py     # K-fold validation
|   +-- 08_error_analysis.py       # Error analysis
|-- tests/                         # Unit tests
|   |-- __init__.py
|   |-- test_utils.py              # Utility function tests
|   +-- test_models.py             # Model training tests
|-- results/                       # Outputs
|   |-- *.png                      # Visualizations
|   |-- *.csv                      # Metrics
|   +-- *.json                     # Hyperparameters
|-- download_dataset.py            # Dataset download script
|-- utils.py                       # Utility functions
|-- config.py                      # Configuration settings
|-- requirements.txt               # Dependencies
+-- README.md                      # This file
```

---

## Installation

### Prerequisites

- Python 3.12+
- pip3
- Virtual environment (recommended)
- Kaggle account (for dataset download)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Instagram_Fake_Detection
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip3 install -r requirements.txt
```

4. **Configure Kaggle API** (required for dataset download)
```bash
# 1. Go to https://www.kaggle.com/settings
# 2. Scroll to 'API' section and click 'Create New Token'
# 3. Download kaggle.json and place it in ~/.kaggle/
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

5. **Download the dataset**
```bash
python3 download_dataset.py
```

6. **Verify installation**
```bash
python3 config.py
```

### Dataset

The project uses the [Instagram Fake Spammer Genuine Accounts](https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts) dataset from Kaggle.

The `download_dataset.py` script will automatically:
- Check if dataset files exist
- Download from Kaggle if missing
- Verify the download

**Manual download** (alternative):
1. Go to https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts
2. Download and extract files to `data/` folder
3. Ensure `train.csv` and `test.csv` are in `data/`

---

## Usage

### Running the Complete Pipeline

Execute notebooks in order:
```bash
# 1. Setup and verification
python3 notebooks/00_setup_test.py

# 2. Exploratory Data Analysis
python3 notebooks/01_EDA.py

# 3. Baseline models
python3 notebooks/02_baseline.py

# 4. Feature engineering
python3 notebooks/03_feature_engineering.py

# 5. Ensemble methods
python3 notebooks/04_ensemble.py

# 6. SHAP explainability
python3 notebooks/05_shap.py

# 7. Hyperparameter tuning (takes 15-25 minutes)
python3 notebooks/06_hyperparameter_tuning.py

# 8. Cross-validation
python3 notebooks/07_cross_validation.py

# 9. Error analysis
python3 notebooks/08_error_analysis.py
```

### Running Tests
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 tests/test_utils.py
python3 tests/test_models.py
```

---

## Features

### Original Features (11)

- `profile pic`: Has profile picture (1/0)
- `nums/length username`: Ratio of digits in username
- `fullname words`: Number of words in full name
- `nums/length fullname`: Ratio of digits in full name
- `name==username`: Full name equals username (1/0)
- `description length`: Biography character count
- `external URL`: Has external URL (1/0)
- `private`: Private account (1/0)
- `#posts`: Number of posts
- `#followers`: Number of followers
- `#follows`: Number of following

### Engineered Features (10)

- `follower_following_ratio`: Followers / (Following + 1)
- `posts_followers_ratio`: Posts / (Followers + 1)
- `following_posts_ratio`: Following / (Posts + 1)
- `has_profile_pic`: Binary indicator
- `has_external_url`: Binary indicator
- `is_private`: Binary indicator
- `username_digit_heavy`: Username > 50% digits
- `no_fullname`: Empty full name
- `short_description`: Bio < 20 characters
- `high_following`: Following > 1000

---

## Methodology

### 1. Data Preprocessing
- Handle missing values (fill with 0)
- Feature engineering (ratio calculations)
- Train/test split (80/20)
- Feature scaling for Logistic Regression

### 2. Model Development
- **Baseline**: Logistic Regression, Random Forest, Gradient Boosting, Neural Network (PyTorch)
- **Feature Engineering**: Additional 10 features
- **Ensemble**: Voting classifier with soft voting
- **Optimization**: GridSearch hyperparameter tuning

### 3. Evaluation
- Accuracy, Precision, Recall, F1-Score
- 5-fold cross-validation
- ROC curve and AUC
- Confusion matrix analysis
- SHAP feature importance

### 4. Explainability
- SHAP (SHapley Additive exPlanations)
- Feature importance ranking
- Individual prediction explanations
- Global model interpretation

---

## Key Findings

### Most Important Features (SHAP Analysis)

1. **#followers** (56.7% importance)
2. **follower_following_ratio** (12.5%)
3. **#posts** (8.3%)
4. **nums/length username** (6.5%)
5. **has_profile_pic** (3.7%)

### Insights

- Fake accounts typically have:
  - Lower follower counts
  - High following/follower ratios
  - More digits in usernames
  - Shorter or missing bios
  - Fewer posts

---

## Technologies Used

- **Python 3.12**: Core programming language
- **scikit-learn**: Machine learning algorithms
- **PyTorch**: Neural network implementation
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Visualization
- **SHAP**: Model explainability
- **pytest**: Unit testing
- **pickle**: Model serialization

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Training Time | ~2 minutes |
| Inference Time | <1ms per prediction |
| Model Size | ~50MB |
| Memory Usage | ~500MB peak |

---

## Future Improvements

- [ ] Add text analysis (if username/bio data available)
- [ ] Implement deeper neural network architectures (CNN, Transformer)
- [ ] Add real-time prediction API
- [ ] Expand dataset size
- [ ] Deploy as web service
- [ ] Add more social media platforms

---

## Limitations

- Dataset contains only metadata (no text content)
- Small dataset: 576 training samples (288 real, 288 fake) and 120 test samples (total 696), limiting generalisation
- No temporal analysis (account age, activity patterns)
- Metadata-only approach may miss sophisticated fake accounts
- Test accuracy (91.38%) is ~2 pp below the best single-split result (93.97%), reflecting normal variance across different data splits

---

## Academic Context

This project is submitted as the final year project for BSc Software Engineering at the University of Greenwich. The project demonstrates:

- Software engineering best practices
- Machine learning pipeline development
- Statistical analysis and evaluation
- Code quality and testing
- Documentation and reproducibility

**Supervisor:** Mr. Nkwo Makuochi \
**Supervisor:** Ms. Razia Sulthana Abdul Kareem

---

## License

This project is submitted for academic purposes. All rights reserved.

---

## Contact

**Vladyslav Danyliuk**  
Student ID: 001348956  
Email: vd3147r@gre.ac.uk  
University: University of Greenwich

---

## Acknowledgments

- University of Greenwich, School of Computing and Mathematical Sciences
- Kaggle for the Instagram fake accounts dataset
- scikit-learn and SHAP communities for excellent documentation

---

**Last Updated:** April 2026