"""
Advanced Visualizations for XGBoost

Project: XGBoost Gradient Boosting
Author: Molla Samser (Founder)
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
GitHub: https://github.com/rskworld
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_learning_curves():
    """
    Plot learning curves for XGBoost model.
    """
    print("=" * 60)
    print("Learning Curves Visualization")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Get evaluation results
    results = model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    
    # Plot learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Log loss
    axes[0].plot(x_axis, results['validation_0']['logloss'], label='Train', linewidth=2, marker='o', markersize=3)
    axes[0].plot(x_axis, results['validation_0']['logloss'], label='Validation', linewidth=2, marker='s', markersize=3)
    axes[0].set_xlabel('Epochs', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Log Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Learning Curve - Log Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy (if available)
    if 'error' in results['validation_0']:
        error = results['validation_0']['error']
        accuracy = [1 - e for e in error]
        axes[1].plot(x_axis, accuracy, label='Validation Accuracy', linewidth=2, marker='o', markersize=3, color='green')
        axes[1].set_xlabel('Epochs', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('Learning Curve - Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Plot feature importance instead
        importance = model.feature_importances_
        top_10_idx = np.argsort(importance)[-10:]
        axes[1].barh(range(10), importance[top_10_idx], color='steelblue')
        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels([f'Feature {i+1}' for i in top_10_idx])
        axes[1].set_xlabel('Importance', fontsize=12, fontweight='bold')
        axes[1].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('XGBoost Learning Curves', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    print("\nLearning curves saved as 'learning_curves.png'")
    plt.show()


def plot_hyperparameter_sensitivity():
    """
    Plot hyperparameter sensitivity analysis.
    """
    print("\n" + "=" * 60)
    print("Hyperparameter Sensitivity Analysis")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test different hyperparameter values
    param_name = 'max_depth'
    param_range = [3, 4, 5, 6, 7, 8, 9, 10]
    
    train_scores = []
    test_scores = []
    
    print(f"\nTesting {param_name} values...")
    for param_value in param_range:
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=param_value,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = (train_pred == y_train).mean()
        test_acc = (test_pred == y_test).mean()
        
        train_scores.append(train_acc)
        test_scores.append(test_acc)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores, 'o-', label='Train Accuracy', linewidth=2, markersize=8)
    plt.plot(param_range, test_scores, 's-', label='Test Accuracy', linewidth=2, markersize=8)
    plt.xlabel(f'{param_name}', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title(f'Hyperparameter Sensitivity: {param_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
    print("\nSensitivity plot saved as 'hyperparameter_sensitivity.png'")
    plt.show()


def plot_model_comparison():
    """
    Compare different XGBoost configurations.
    """
    print("\n" + "=" * 60)
    print("Model Comparison Visualization")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Different model configurations
    configs = [
        {'name': 'Shallow (depth=3)', 'max_depth': 3, 'n_estimators': 200},
        {'name': 'Medium (depth=6)', 'max_depth': 6, 'n_estimators': 100},
        {'name': 'Deep (depth=9)', 'max_depth': 9, 'n_estimators': 100},
        {'name': 'Fast (lr=0.2)', 'max_depth': 6, 'learning_rate': 0.2, 'n_estimators': 50},
        {'name': 'Slow (lr=0.05)', 'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 200},
    ]
    
    results = []
    
    print("\nTraining different model configurations...")
    for config in configs:
        params = {
            'objective': 'binary:logistic',
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        params.update({k: v for k, v in config.items() if k != 'name'})
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = (train_pred == y_train).mean()
        test_acc = (test_pred == y_test).mean()
        
        results.append({
            'Model': config['name'],
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc
        })
    
    # Create comparison plot
    df_results = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_results))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_results['Train Accuracy'], width, 
                   label='Train Accuracy', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, df_results['Test Accuracy'], width,
                   label='Test Accuracy', color='coral', edgecolor='black')
    
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - Different Configurations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nModel comparison saved as 'model_comparison.png'")
    plt.show()
    
    print("\nResults Summary:")
    print(df_results.to_string(index=False))


def plot_roc_pr_curves():
    """
    Plot ROC and Precision-Recall curves.
    """
    print("\n" + "=" * 60)
    print("ROC and Precision-Recall Curves")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = np.trapz(tpr, fpr)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = np.trapz(precision, recall)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='steelblue')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    axes[0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    axes[1].plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})', color='coral')
    axes[1].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Model Performance Curves', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('roc_pr_curves.png', dpi=300, bbox_inches='tight')
    print("\nROC and PR curves saved as 'roc_pr_curves.png'")
    plt.show()


def main():
    """
    Main function to generate all visualizations.
    """
    print("\n" + "=" * 60)
    print("XGBoost Advanced Visualizations")
    print("Author: Molla Samser (Founder)")
    print("Website: https://rskworld.in")
    print("=" * 60)
    
    # Learning curves
    plot_learning_curves()
    
    # Hyperparameter sensitivity
    plot_hyperparameter_sensitivity()
    
    # Model comparison
    plot_model_comparison()
    
    # ROC and PR curves
    plot_roc_pr_curves()
    
    print("\n" + "=" * 60)
    print("All Visualizations Generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()

