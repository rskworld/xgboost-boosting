"""
XGBoost Model Training and Evaluation Script

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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, roc_curve
)
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def train_classification_model():
    """
    Train and evaluate XGBoost classification model.
    """
    print("=" * 60)
    print("XGBoost Classification Model Training")
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("\nTraining model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    print("\nPerforming Cross-Validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy', n_jobs=-1)
    print(f"CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Classification Model Evaluation', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('classification_evaluation.png', dpi=300, bbox_inches='tight')
    print("\nEvaluation plots saved as 'classification_evaluation.png'")
    plt.show()
    
    # Learning curve
    results = model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['logloss'], label='Train', linewidth=2)
    plt.plot(x_axis, results['validation_0']['logloss'], label='Validation', linewidth=2)
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Log Loss', fontsize=12, fontweight='bold')
    plt.title('Learning Curve - Classification', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('classification_learning_curve.png', dpi=300, bbox_inches='tight')
    print("Learning curve saved as 'classification_learning_curve.png'")
    plt.show()
    
    return model


def train_regression_model():
    """
    Train and evaluate XGBoost regression model.
    """
    print("\n" + "=" * 60)
    print("XGBoost Regression Model Training")
    print("=" * 60)
    
    # Create dataset
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse'
    )
    
    print("\nTraining model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Cross-validation
    print("\nPerforming Cross-Validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
    cv_rmse = np.sqrt(-cv_scores)
    print(f"CV Mean RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Predicted vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.6, s=50)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Predicted vs Actual (R² = {r2:.4f})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=50)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
    axes[1].set_title('Residuals Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Regression Model Evaluation', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('regression_evaluation.png', dpi=300, bbox_inches='tight')
    print("\nEvaluation plots saved as 'regression_evaluation.png'")
    plt.show()
    
    # Learning curve
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['rmse'], label='Train', linewidth=2)
    plt.plot(x_axis, results['validation_0']['rmse'], label='Validation', linewidth=2)
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12, fontweight='bold')
    plt.title('Learning Curve - Regression', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regression_learning_curve.png', dpi=300, bbox_inches='tight')
    print("Learning curve saved as 'regression_learning_curve.png'")
    plt.show()
    
    return model


def main():
    """
    Main function to train and evaluate both classification and regression models.
    """
    print("\n" + "=" * 60)
    print("XGBoost Model Training and Evaluation")
    print("Author: Molla Samser (Founder)")
    print("Website: https://rskworld.in")
    print("=" * 60)
    
    # Train classification model
    class_model = train_classification_model()
    class_model.save_model('trained_classification_model.json')
    print("\nClassification model saved as 'trained_classification_model.json'")
    
    # Train regression model
    reg_model = train_regression_model()
    reg_model.save_model('trained_regression_model.json')
    print("Regression model saved as 'trained_regression_model.json'")
    
    print("\n" + "=" * 60)
    print("Model Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

