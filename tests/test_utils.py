"""
Unit tests for utility functions
Run with: python -m pytest tests/
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    load_and_prepare_data,
    calculate_metrics,
    print_metrics,
    print_dataset_info
)
import config


class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests"""
        # Create sample dataset
        cls.sample_data = pd.DataFrame({
            'profile pic': [1, 0, 1, 0, 1],
            'nums/length username': [0.2, 0.6, 0.3, 0.8, 0.1],
            'fullname words': [2, 0, 1, 0, 3],
            'nums/length fullname': [0.0, 0.5, 0.0, 0.3, 0.0],
            'name==username': [0, 1, 0, 1, 0],
            'description length': [50, 10, 30, 5, 100],
            'external URL': [1, 0, 1, 0, 1],
            'private': [0, 0, 1, 0, 0],
            '#posts': [100, 5, 50, 2, 200],
            '#followers': [1000, 50, 500, 20, 2000],
            '#follows': [500, 1000, 300, 2000, 400],
            'fake': [0, 1, 0, 1, 0]
        })

        # Save to temporary CSV
        cls.temp_csv = 'temp_test_data.csv'
        cls.sample_data.to_csv(cls.temp_csv, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if os.path.exists(cls.temp_csv):
            os.remove(cls.temp_csv)

    def test_load_and_prepare_data(self):
        """Test data loading and feature engineering"""
        X, y, feature_names = load_and_prepare_data(self.temp_csv)

        # Check dimensions
        self.assertEqual(len(X), 5, "Should have 5 samples")
        self.assertEqual(len(y), 5, "Should have 5 labels")
        self.assertEqual(len(feature_names), 21, "Should have 21 features")

        # Check engineered features exist
        self.assertIn('follower_following_ratio', X.columns)
        self.assertIn('posts_followers_ratio', X.columns)
        self.assertIn('username_digit_heavy', X.columns)

        # Check no NaN values
        self.assertEqual(X.isnull().sum().sum(), 0, "Should have no NaN values")

    def test_calculate_metrics(self):
        """Test metric calculation"""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])

        metrics = calculate_metrics(y_true, y_pred)

        # Check all metrics present
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)

        # Check values are in valid range
        for key, value in metrics.items():
            self.assertGreaterEqual(value, 0.0, f"{key} should be >= 0")
            self.assertLessEqual(value, 1.0, f"{key} should be <= 1")

        # Check specific values
        self.assertEqual(metrics['accuracy'], 0.8, "Accuracy should be 0.8")
        self.assertEqual(metrics['recall'], 0.5, "Recall should be 0.5")

    def test_feature_engineering(self):
        """Test engineered features are calculated correctly"""
        X, y, _ = load_and_prepare_data(self.temp_csv)

        # Test follower_following_ratio
        expected_ratio = 1000 / (500 + 1)  # First sample
        actual_ratio = X['follower_following_ratio'].iloc[0]
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=4)

        # Test username_digit_heavy (threshold > 0.5)
        self.assertEqual(X['username_digit_heavy'].iloc[0], 0)  # 0.2 < 0.5
        self.assertEqual(X['username_digit_heavy'].iloc[1], 1)  # 0.6 > 0.5

        # Test short_description (threshold < 20)
        self.assertEqual(X['short_description'].iloc[0], 0)  # 50 > 20
        self.assertEqual(X['short_description'].iloc[1], 1)  # 10 < 20

    def test_config_constants(self):
        """Test configuration constants"""
        # Check feature lists
        self.assertEqual(len(config.ORIGINAL_FEATURES), 11)
        self.assertEqual(len(config.ENGINEERED_FEATURES), 10)
        self.assertEqual(len(config.ALL_FEATURES), 21)

        # Check hyperparameters
        self.assertEqual(config.RANDOM_STATE, 42)
        self.assertEqual(config.TEST_SIZE, 0.2)
        self.assertEqual(config.CV_FOLDS, 5)

        # Check thresholds
        self.assertGreater(config.TARGET_ACCURACY, 0.9)
        self.assertLess(config.ACCEPTABLE_FPR, 0.2)


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases"""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        empty_csv = 'empty_test.csv'
        empty_df.to_csv(empty_csv, index=False)

        try:
            with self.assertRaises(Exception):
                load_and_prepare_data(empty_csv)
        finally:
            if os.path.exists(empty_csv):
                os.remove(empty_csv)

    def test_missing_columns(self):
        """Test handling of missing required columns"""
        incomplete_df = pd.DataFrame({
            'profile pic': [1, 0],
            'fake': [0, 1]
        })
        incomplete_csv = 'incomplete_test.csv'
        incomplete_df.to_csv(incomplete_csv, index=False)

        try:
            with self.assertRaises(Exception):
                load_and_prepare_data(incomplete_csv)
        finally:
            if os.path.exists(incomplete_csv):
                os.remove(incomplete_csv)

    def test_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = calculate_metrics(y_true, y_pred)

        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)

    def test_metrics_all_wrong(self):
        """Test metrics with all wrong predictions"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])

        metrics = calculate_metrics(y_true, y_pred)

        self.assertEqual(metrics['accuracy'], 0.0)


class TestFeatureThresholds(unittest.TestCase):
    """Test feature engineering thresholds"""

    def test_username_digit_threshold(self):
        """Test username digit threshold"""
        self.assertEqual(config.USERNAME_DIGIT_THRESHOLD, 0.5)

    def test_description_length_threshold(self):
        """Test description length threshold"""
        self.assertEqual(config.SHORT_DESCRIPTION_THRESHOLD, 20)

    def test_following_threshold(self):
        """Test high following threshold"""
        self.assertEqual(config.HIGH_FOLLOWING_THRESHOLD, 1000)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)