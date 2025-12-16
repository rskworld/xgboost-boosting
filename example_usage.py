"""
Example Usage of XGBoost Gradient Boosting Project

Project: XGBoost Gradient Boosting
Author: Molla Samser (Founder)
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
GitHub: https://github.com/rskworld
"""

import xgboost as xgb
from utils.data_loader import load_data, prepare_data, get_data_info
from utils.model_evaluator import evaluate_classification, evaluate_regression, print_metrics


def example_classification():
    """
    Example: Classification with XGBoost
    """
    print("=" * 60)
    print("Example: Classification")
    print("=" * 60)
    
    # Load data
    X, y, feature_names = load_data(task_type='classification')
    
    # Get data info
    info = get_data_info(X, y, feature_names)
    print(f"\nDataset Info:")
    print(f"  Samples: {info['n_samples']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Task Type: {info['target_type']}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, _ = prepare_data(
        X, y, test_size=0.2, stratify=y
    )
    
    # Train model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    metrics = evaluate_classification(y_test, y_pred, y_pred_proba, plot=False)
    print_metrics(metrics, task_type='classification', y_true=y_test, y_pred=y_pred)
    
    return model


def example_regression():
    """
    Example: Regression with XGBoost
    """
    print("\n" + "=" * 60)
    print("Example: Regression")
    print("=" * 60)
    
    # Load data
    X, y, feature_names = load_data(task_type='regression')
    
    # Get data info
    info = get_data_info(X, y, feature_names)
    print(f"\nDataset Info:")
    print(f"  Samples: {info['n_samples']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Task Type: {info['target_type']}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, _ = prepare_data(
        X, y, test_size=0.2
    )
    
    # Train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_regression(y_test, y_pred, plot=False)
    print_metrics(metrics, task_type='regression')
    
    return model


def main():
    """
    Main function demonstrating project usage.
    """
    print("\n" + "=" * 60)
    print("XGBoost Gradient Boosting - Example Usage")
    print("Author: Molla Samser (Founder)")
    print("Website: https://rskworld.in")
    print("=" * 60)
    
    # Classification example
    class_model = example_classification()
    
    # Regression example
    reg_model = example_regression()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
    
    # Save models
    class_model.save_model('example_classification_model.json')
    reg_model.save_model('example_regression_model.json')
    print("\nModels saved:")
    print("  - example_classification_model.json")
    print("  - example_regression_model.json")


if __name__ == "__main__":
    main()

