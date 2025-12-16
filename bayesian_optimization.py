"""
Bayesian Optimization for XGBoost Hyperparameter Tuning

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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

# Try to import optuna for Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")
    print("Using alternative optimization method...")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def bayesian_optimization_classification(X, y, n_trials=50):
    """
    Bayesian optimization for classification hyperparameters.
    """
    if not OPTUNA_AVAILABLE:
        return None, None
    
    print("=" * 60)
    print("Bayesian Optimization - Classification")
    print("=" * 60)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def objective(trial):
        """Objective function for Optuna."""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 2),
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    print(f"\nRunning Bayesian optimization with {n_trials} trials...")
    study = optuna.create_study(direction='maximize', study_name='xgboost_classification')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\nBest Parameters:")
    print("=" * 60)
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    
    print(f"\nBest Cross-Validation Score: {study.best_value:.4f}")
    
    # Train best model
    best_params = study.best_params.copy()
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    
    best_model = xgb.XGBClassifier(**best_params)
    
    # Plot optimization history
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Optimization history
    try:
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0])
        axes[0].set_title('Optimization History', fontsize=14, fontweight='bold')
    except Exception as e:
        # Fallback: manual plot
        trials = study.trials
        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials if t.value is not None]
        trial_nums = [t.number for t in trials if t.value is not None]
        axes[0].plot(trial_nums, values, 'o-', linewidth=2, markersize=4)
        axes[0].set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Objective Value', fontsize=12, fontweight='bold')
        axes[0].set_title('Optimization History', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
    
    # Parameter importance
    try:
        optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[1])
        axes[1].set_title('Parameter Importance', fontsize=14, fontweight='bold')
    except Exception as e:
        axes[1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                    ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('bayesian_optimization_classification.png', dpi=300, bbox_inches='tight')
    print("\nOptimization plots saved as 'bayesian_optimization_classification.png'")
    plt.show()
    
    return best_model, study


def bayesian_optimization_regression(X, y, n_trials=50):
    """
    Bayesian optimization for regression hyperparameters.
    """
    if not OPTUNA_AVAILABLE:
        return None, None
    
    print("\n" + "=" * 60)
    print("Bayesian Optimization - Regression")
    print("=" * 60)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def objective(trial):
        """Objective function for Optuna."""
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 2),
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
        return -scores.mean()  # Return negative because we want to minimize
    
    print(f"\nRunning Bayesian optimization with {n_trials} trials...")
    study = optuna.create_study(direction='minimize', study_name='xgboost_regression')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\nBest Parameters:")
    print("=" * 60)
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    
    print(f"\nBest Cross-Validation RMSE: {np.sqrt(study.best_value):.4f}")
    
    # Train best model
    best_params = study.best_params.copy()
    best_params['objective'] = 'reg:squarederror'
    best_params['eval_metric'] = 'rmse'
    best_params['random_state'] = 42
    
    best_model = xgb.XGBRegressor(**best_params)
    
    return best_model, study


def main():
    """
    Main function for Bayesian optimization.
    """
    print("\n" + "=" * 60)
    print("Bayesian Optimization for XGBoost")
    print("Author: Molla Samser (Founder)")
    print("Website: https://rskworld.in")
    print("=" * 60)
    
    if not OPTUNA_AVAILABLE:
        print("\nOptuna is required for Bayesian optimization.")
        print("Install it with: pip install optuna")
        return
    
    # Classification
    X_class, y_class = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    class_model, class_study = bayesian_optimization_classification(X_class, y_class, n_trials=30)
    
    # Regression
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    reg_model, reg_study = bayesian_optimization_regression(X_reg, y_reg, n_trials=30)
    
    print("\n" + "=" * 60)
    print("Bayesian Optimization Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

