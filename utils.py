"""
Utility functions for Instagram Fake Account Detection project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_and_prepare_data(filepath):
    """
    Load dataset and prepare features

    Args:
        filepath (str): Path to CSV file

    Returns:
        tuple: X (features), y (labels), feature_names
    """
    df = pd.read_csv(filepath)

    # Original features
    original_features = ['profile pic', 'nums/length username', 'fullname words',
                         'nums/length fullname', 'name==username', 'description length',
                         'external URL', 'private', '#posts', '#followers', '#follows']

    # Engineer new features
    df['follower_following_ratio'] = df['#followers'] / (df['#follows'] + 1)
    df['posts_followers_ratio'] = df['#posts'] / (df['#followers'] + 1)
    df['following_posts_ratio'] = df['#follows'] / (df['#posts'] + 1)
    df['has_profile_pic'] = df['profile pic']
    df['has_external_url'] = df['external URL']
    df['is_private'] = df['private']
    df['username_digit_heavy'] = (df['nums/length username'] > 0.5).astype(int)
    df['no_fullname'] = (df['fullname words'] == 0).astype(int)
    df['short_description'] = (df['description length'] < 20).astype(int)
    df['high_following'] = (df['#follows'] > 1000).astype(int)

    all_features = original_features + [
        'follower_following_ratio', 'posts_followers_ratio', 'following_posts_ratio',
        'has_profile_pic', 'has_external_url', 'is_private', 'username_digit_heavy',
        'no_fullname', 'short_description', 'high_following'
    ]

    X = df[all_features].fillna(0)
    y = df['fake']

    return X, y, all_features


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        dict: Dictionary with metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }


def print_metrics(metrics, model_name="Model"):
    """
    Print metrics in formatted way

    Args:
        metrics (dict): Dictionary with metrics
        model_name (str): Name of the model
    """
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")


def plot_feature_importance(importances, feature_names, top_n=10, save_path=None):
    """
    Plot feature importances

    Args:
        importances: Array of feature importances
        feature_names: List of feature names
        top_n (int): Number of top features to show
        save_path (str): Path to save plot
    """
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()


def compare_models(results_dict, save_path=None):
    """
    Compare multiple models performance

    Args:
        results_dict (dict): Dictionary with model names as keys and metrics as values
        save_path (str): Path to save plot
    """
    df = pd.DataFrame(results_dict).T

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        df[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Model')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(df.index, rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()


def print_dataset_info(df):
    """
    Print dataset information

    Args:
        df: pandas DataFrame
    """
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"\nClass distribution:")
    print(df['fake'].value_counts())
    print(f"\nClass balance:")
    print(df['fake'].value_counts(normalize=True))
    print(f"\nMissing values:")
    print(df.isnull().sum().sum())
    print("=" * 60)


def save_model_summary(model_name, metrics, params, filepath):
    """
    Save model summary to text file

    Args:
        model_name (str): Name of the model
        metrics (dict): Model metrics
        params (dict): Model parameters
        filepath (str): Path to save file
    """
    with open(filepath, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Parameters:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nPerformance Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")

    print(f"Model summary saved: {filepath}")