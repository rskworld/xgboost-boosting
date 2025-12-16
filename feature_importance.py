"""
XGBoost Feature Importance Analysis Script

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
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def analyze_feature_importance_classification():
    """
    Analyze feature importance for classification task.
    """
    print("=" * 60)
    print("Feature Importance Analysis - Classification")
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
    
    # Feature names
    feature_names = [f'feature_{i+1}' for i in range(20)]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Get feature importance
    importance_gain = model.feature_importances_
    importance_dict = dict(zip(feature_names, importance_gain))
    importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Create DataFrame
    importance_df = pd.DataFrame(importance_sorted, columns=['Feature', 'Importance'])
    
    # Plot
    plt.figure(figsize=(12, 8))
    importance_df_sorted = importance_df.sort_values('Importance', ascending=True)
    plt.barh(importance_df_sorted['Feature'], importance_df_sorted['Importance'], 
             color='steelblue', edgecolor='black')
    plt.xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title('XGBoost Feature Importance - Classification', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance_classification.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'feature_importance_classification.png'")
    plt.show()
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    print(importance_df.head(10).to_string(index=False))
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    return model, importance_df


def analyze_feature_importance_regression():
    """
    Analyze feature importance for regression task.
    """
    print("\n" + "=" * 60)
    print("Feature Importance Analysis - Regression")
    print("=" * 60)
    
    # Create dataset
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    # Feature names
    feature_names = [f'feature_{i+1}' for i in range(20)]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='rmse'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Get feature importance
    importance_gain = model.feature_importances_
    importance_dict = dict(zip(feature_names, importance_gain))
    importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Create DataFrame
    importance_df = pd.DataFrame(importance_sorted, columns=['Feature', 'Importance'])
    
    # Plot
    plt.figure(figsize=(12, 8))
    importance_df_sorted = importance_df.sort_values('Importance', ascending=True)
    plt.barh(importance_df_sorted['Feature'], importance_df_sorted['Importance'],
             color='coral', edgecolor='black')
    plt.xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title('XGBoost Feature Importance - Regression', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance_regression.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'feature_importance_regression.png'")
    plt.show()
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    print(importance_df.head(10).to_string(index=False))
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return model, importance_df


def compare_importance_methods():
    """
    Compare different feature importance methods.
    """
    print("\n" + "=" * 60)
    print("Comparing Feature Importance Methods")
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
    
    feature_names = [f'feature_{i+1}' for i in range(20)]
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
    
    # Get different importance types
    importance_gain = model.get_booster().get_score(importance_type='gain')
    importance_weight = model.get_booster().get_score(importance_type='weight')
    importance_cover = model.get_booster().get_score(importance_type='cover')
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'Gain': [importance_gain.get(f'f{i}', 0) for i in range(20)],
        'Weight': [importance_weight.get(f'f{i}', 0) for i in range(20)],
        'Cover': [importance_cover.get(f'f{i}', 0) for i in range(20)]
    })
    
    # Normalize for comparison
    for col in ['Gain', 'Weight', 'Cover']:
        comparison_df[col] = comparison_df[col] / comparison_df[col].sum()
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (col, ax) in enumerate(zip(['Gain', 'Weight', 'Cover'], axes)):
        sorted_df = comparison_df.sort_values(col, ascending=True)
        ax.barh(sorted_df['Feature'], sorted_df[col], color=['steelblue', 'coral', 'lightgreen'][idx])
        ax.set_xlabel('Normalized Importance', fontsize=10, fontweight='bold')
        ax.set_title(f'Importance Type: {col}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Feature Importance Comparison - Different Methods', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'feature_importance_comparison.png'")
    plt.show()
    
    print("\nTop 5 Features by Each Method:")
    print("-" * 60)
    for method in ['Gain', 'Weight', 'Cover']:
        top5 = comparison_df.nlargest(5, method)[['Feature', method]]
        print(f"\n{method}:")
        print(top5.to_string(index=False))


def main():
    """
    Main function to run feature importance analysis.
    """
    print("\n" + "=" * 60)
    print("XGBoost Feature Importance Analysis")
    print("Author: Molla Samser (Founder)")
    print("Website: https://rskworld.in")
    print("=" * 60)
    
    # Classification
    class_model, class_importance = analyze_feature_importance_classification()
    
    # Regression
    reg_model, reg_importance = analyze_feature_importance_regression()
    
    # Compare methods
    compare_importance_methods()
    
    print("\n" + "=" * 60)
    print("Feature Importance Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

