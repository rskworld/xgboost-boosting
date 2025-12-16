"""
XGBoost Hyperparameter Tuning Script

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
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')


def tune_classification_model():
    """
    Hyperparameter tuning for XGBoost classification model.
    """
    print("=" * 60)
    print("XGBoost Classification - Hyperparameter Tuning")
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
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'n_estimators': [50, 100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    # Create base model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    # Grid Search
    print("\nPerforming Grid Search...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Results
    print("\n" + "=" * 60)
    print("Best Parameters (Grid Search):")
    print("=" * 60)
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    print(f"\nBest Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_model, grid_search.best_params_


def tune_regression_model():
    """
    Hyperparameter tuning for XGBoost regression model.
    """
    print("\n" + "=" * 60)
    print("XGBoost Regression - Hyperparameter Tuning")
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
    
    # Define parameter grid (smaller for faster execution)
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Create base model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        eval_metric='rmse',
        n_jobs=-1
    )
    
    # Randomized Search (faster than Grid Search)
    print("\nPerforming Randomized Search...")
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings sampled
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    # Results
    print("\n" + "=" * 60)
    print("Best Parameters (Randomized Search):")
    print("=" * 60)
    for param, value in random_search.best_params_.items():
        print(f"{param}: {value}")
    
    print(f"\nBest Cross-Validation Score (Neg MSE): {random_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Set Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return best_model, random_search.best_params_


def main():
    """
    Main function to run hyperparameter tuning for both classification and regression.
    """
    print("\n" + "=" * 60)
    print("XGBoost Hyperparameter Tuning")
    print("Author: Molla Samser (Founder)")
    print("Website: https://rskworld.in")
    print("=" * 60)
    
    # Tune classification model
    class_model, class_params = tune_classification_model()
    
    # Tune regression model
    reg_model, reg_params = tune_regression_model()
    
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Complete!")
    print("=" * 60)
    
    # Save models
    class_model.save_model('best_classification_model.json')
    reg_model.save_model('best_regression_model.json')
    print("\nModels saved successfully!")
    print("- best_classification_model.json")
    print("- best_regression_model.json")


if __name__ == "__main__":
    main()

