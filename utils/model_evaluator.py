"""
Model evaluation utilities

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
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classification(y_true, y_pred, y_pred_proba=None, plot=True):
    """
    Evaluate classification model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities
    plot : bool, default=True
        Whether to create plots
    
    Returns:
    --------
    metrics : dict
        Dictionary with evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
    }
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics.update(report)
    
    # AUC-ROC if probabilities available
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Plot if requested
    if plot:
        fig, axes = plt.subplots(1, 2 if y_pred_proba is not None else 1, figsize=(14, 6))
        if y_pred_proba is None:
            axes = [axes]
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # ROC Curve
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {metrics["auc_roc"]:.4f})')
            axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return metrics


def evaluate_regression(y_true, y_pred, plot=True):
    """
    Evaluate regression model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    plot : bool, default=True
        Whether to create plots
    
    Returns:
    --------
    metrics : dict
        Dictionary with evaluation metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }
    
    # Additional metrics
    residuals = y_true - y_pred
    metrics['mean_residual'] = np.mean(residuals)
    metrics['std_residual'] = np.std(residuals)
    
    # Plot if requested
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Predicted vs Actual
        axes[0].scatter(y_true, y_pred, alpha=0.6, s=50)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Predicted vs Actual (R² = {metrics["r2"]:.4f})',
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=50)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
        axes[1].set_title('Residuals Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return metrics


def print_metrics(metrics, task_type='classification', y_true=None, y_pred=None):
    """
    Print evaluation metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with evaluation metrics
    task_type : str, default='classification'
        Type of task: 'classification' or 'regression'
    y_true : array-like, optional
        True labels/values (needed for classification report)
    y_pred : array-like, optional
        Predicted labels/values (needed for classification report)
    """
    print("=" * 60)
    print("Model Evaluation Metrics")
    print("=" * 60)
    
    if task_type == 'classification':
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        if 'auc_roc' in metrics:
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"Average Precision: {metrics['average_precision']:.4f}")
        if y_true is not None and y_pred is not None:
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred))
    else:
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"Mean Residual: {metrics['mean_residual']:.4f}")
        print(f"Std Residual: {metrics['std_residual']:.4f}")
    
    print("=" * 60)

