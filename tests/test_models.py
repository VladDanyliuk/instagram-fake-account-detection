"""
Unit tests for model training and predictions
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import config


class TestModelTraining(unittest.TestCase):
    """Test model training functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        np.random.seed(42)

        # Generate synthetic data
        n_samples = 100
        n_features = 21

        cls.X = np.random.randn(n_samples, n_features)
        cls.y = np.random.randint(0, 2, n_samples)

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

    def test_random_forest_training(self):
        """Test Random Forest can be trained"""
        rf = RandomForestClassifier(**config.RF_PARAMS)
        rf.fit(self.X_train, self.y_train)

        # Check model is fitted
        self.assertTrue(hasattr(rf, 'estimators_'))

        # Check predictions
        predictions = rf.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

        # Check prediction values are binary
        unique_preds = np.unique(predictions)
        self.assertTrue(all(p in [0, 1] for p in unique_preds))

    def test_gradient_boosting_training(self):
        """Test Gradient Boosting can be trained"""
        gb = GradientBoostingClassifier(**config.GB_PARAMS)
        gb.fit(self.X_train, self.y_train)

        # Check model is fitted
        self.assertTrue(hasattr(gb, 'estimators_'))

        # Check predictions
        predictions = gb.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    def test_logistic_regression_training(self):
        """Test Logistic Regression can be trained"""
        lr = LogisticRegression(**config.LR_PARAMS)
        lr.fit(self.X_train, self.y_train)

        # Check model is fitted
        self.assertTrue(hasattr(lr, 'coef_'))

        # Check predictions
        predictions = lr.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    def test_prediction_probabilities(self):
        """Test models can output probabilities"""
        rf = RandomForestClassifier(**config.RF_PARAMS)
        rf.fit(self.X_train, self.y_train)

        probas = rf.predict_proba(self.X_test)

        # Check shape
        self.assertEqual(probas.shape, (len(self.y_test), 2))

        # Check probabilities sum to 1
        prob_sums = probas.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(self.y_test)))

        # Check probabilities are in valid range
        self.assertTrue(np.all(probas >= 0))
        self.assertTrue(np.all(probas <= 1))


class TestModelParameters(unittest.TestCase):
    """Test model parameters from config"""

    def test_rf_parameters(self):
        """Test Random Forest parameters"""
        self.assertIn('n_estimators', config.RF_PARAMS)
        self.assertIn('max_depth', config.RF_PARAMS)
        self.assertIn('random_state', config.RF_PARAMS)

        self.assertGreater(config.RF_PARAMS['n_estimators'], 0)
        self.assertEqual(config.RF_PARAMS['random_state'], 42)

    def test_gb_parameters(self):
        """Test Gradient Boosting parameters"""
        self.assertIn('n_estimators', config.GB_PARAMS)
        self.assertIn('learning_rate', config.GB_PARAMS)
        self.assertIn('random_state', config.GB_PARAMS)

        self.assertGreater(config.GB_PARAMS['learning_rate'], 0)
        self.assertLessEqual(config.GB_PARAMS['learning_rate'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)