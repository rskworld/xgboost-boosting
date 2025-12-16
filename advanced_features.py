"""
Advanced XGBoost Features

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
    mean_squared_error, r2_score, f1_score, precision_score, recall_score
)
from sklearn.datasets import make_classification, make_regression
# StandardScaler not used in this file, removed to avoid unused import
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def multi_class_classification():
    """
    Multi-class classification with XGBoost.
    """
    print("=" * 60)
    print("Multi-Class Classification with XGBoost")
    print("=" * 60)
    
    # Create multi-class dataset
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create multi-class classifier
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    print("\nTraining multi-class model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title('Multi-Class Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('multi_class_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'multi_class_confusion_matrix.png'")
    plt.show()
    
    return model


def custom_objective_regression():
    """
    Custom objective function for regression.
    """
    print("\n" + "=" * 60)
    print("Custom Objective Function - Regression")
    print("=" * 60)
    
    # Create dataset
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Custom objective: Huber loss
    def huber_obj(y_pred, y_true):
        """
        Huber loss objective function.
        """
        d = y_pred - y_true
        h = 1.0  # threshold
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt
        return grad, hess
    
    # Custom evaluation metric
    def huber_eval(y_pred, y_true):
        """
        Huber loss evaluation metric.
        """
        d = y_pred - y_true
        h = 1.0
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        loss = h * h * (scale_sqrt - 1)
        return 'huber', np.mean(loss)
    
    # Train with custom objective
    model = xgb.XGBRegressor(
        objective=huber_obj,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    print("\nTraining model with custom Huber loss...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=huber_eval,
        verbose=False
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nMSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return model


def model_ensemble():
    """
    Ensemble of multiple XGBoost models.
    """
    print("\n" + "=" * 60)
    print("Model Ensemble - Multiple XGBoost Models")
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
    
    # Create multiple models with different parameters
    models = []
    model_configs = [
        {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100},
        {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100},
        {'max_depth': 9, 'learning_rate': 0.05, 'n_estimators': 200},
        {'max_depth': 4, 'learning_rate': 0.15, 'n_estimators': 150},
    ]
    
    print("\nTraining ensemble models...")
    for i, config in enumerate(model_configs):
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42 + i,
            eval_metric='logloss',
            **config
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        models.append(model)
        print(f"  Model {i+1} trained with config: {config}")
    
    # Ensemble predictions (voting)
    predictions = np.array([model.predict(X_test) for model in models])
    ensemble_pred = (predictions.mean(axis=0) > 0.5).astype(int)
    
    # Evaluate ensemble
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    print(f"\nEnsemble Accuracy: {ensemble_accuracy:.4f}")
    
    # Individual model accuracies
    print("\nIndividual Model Accuracies:")
    for i, model in enumerate(models):
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"  Model {i+1}: {acc:.4f}")
    
    # Plot comparison
    model_accs = [accuracy_score(y_test, model.predict(X_test)) for model in models]
    model_accs.append(ensemble_accuracy)
    
    plt.figure(figsize=(10, 6))
    labels = [f'Model {i+1}' for i in range(len(models))] + ['Ensemble']
    colors = ['steelblue'] * len(models) + ['coral']
    bars = plt.bar(labels, model_accs, color=colors, edgecolor='black')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Ensemble Comparison', fontsize=14, fontweight='bold')
    plt.ylim([min(model_accs) - 0.01, max(model_accs) + 0.01])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_ensemble_comparison.png', dpi=300, bbox_inches='tight')
    print("\nEnsemble comparison saved as 'model_ensemble_comparison.png'")
    plt.show()
    
    return models, ensemble_pred


def feature_engineering_demo():
    """
    Feature engineering techniques for XGBoost.
    """
    print("\n" + "=" * 60)
    print("Feature Engineering for XGBoost")
    print("=" * 60)
    
    # Create dataset
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=10,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(10)])
    df['target'] = y
    
    # Feature engineering
    print("\nCreating engineered features...")
    
    # Polynomial features
    df['feature_1_squared'] = df['feature_1'] ** 2
    df['feature_2_squared'] = df['feature_2'] ** 2
    
    # Interaction features
    df['feature_1_x_feature_2'] = df['feature_1'] * df['feature_2']
    df['feature_3_x_feature_4'] = df['feature_3'] * df['feature_4']
    
    # Statistical features
    df['feature_mean'] = df[[f'feature_{i+1}' for i in range(10)]].mean(axis=1)
    df['feature_std'] = df[[f'feature_{i+1}' for i in range(10)]].std(axis=1)
    df['feature_max'] = df[[f'feature_{i+1}' for i in range(10)]].max(axis=1)
    df['feature_min'] = df[[f'feature_{i+1}' for i in range(10)]].min(axis=1)
    
    # Prepare data
    feature_cols = [col for col in df.columns if col != 'target']
    X_engineered = df[feature_cols].values
    y = df['target'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42
    )
    
    # Train model with engineered features
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nModel with Engineered Features:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Number of features: {len(feature_cols)}")
    
    # Feature importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features (including engineered):")
    print(importance_df.head(10).to_string(index=False))
    
    return model, importance_df


def main():
    """
    Main function to run all advanced features.
    """
    print("\n" + "=" * 60)
    print("XGBoost Advanced Features")
    print("Author: Molla Samser (Founder)")
    print("Website: https://rskworld.in")
    print("=" * 60)
    
    # Multi-class classification
    multi_model = multi_class_classification()
    
    # Custom objective
    custom_model = custom_objective_regression()
    
    # Model ensemble
    ensemble_models, ensemble_pred = model_ensemble()
    
    # Feature engineering
    eng_model, importance_df = feature_engineering_demo()
    
    print("\n" + "=" * 60)
    print("Advanced Features Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

