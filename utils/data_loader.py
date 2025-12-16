"""
Data loading and preparation utilities

Project: XGBoost Gradient Boosting
Author: Molla Samser (Founder)
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
GitHub: https://github.com/rskworld
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification, make_regression
import os


def load_data(file_path=None, task_type='classification'):
    """
    Load data from file or generate synthetic data.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to CSV file. If None, generates synthetic data.
    task_type : str, default='classification'
        Type of task: 'classification' or 'regression'
    
    Returns:
    --------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_names : list
        List of feature names
    """
    if file_path and os.path.exists(file_path):
        # Load from file
        df = pd.read_csv(file_path)
        
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = df.columns[:-1].tolist()
        
        # Handle categorical target for classification
        if task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        return X, y, feature_names
    else:
        # Generate synthetic data
        if task_type == 'classification':
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                random_state=42
            )
        else:
            X, y = make_regression(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                noise=10,
                random_state=42
            )
        
        feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
        return X, y, feature_names


def prepare_data(X, y, test_size=0.2, random_state=42, stratify=None, scale=False):
    """
    Prepare data for training by splitting and optionally scaling.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    test_size : float, default=0.2
        Proportion of data for testing
    random_state : int, default=42
        Random seed
    stratify : array-like, optional
        For stratified splitting (classification)
    scale : bool, default=False
        Whether to scale features
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split data
    scaler : StandardScaler or None
        Fitted scaler if scale=True
    """
    # Split data
    if stratify is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Scale if requested
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def save_data(X, y, feature_names, file_path):
    """
    Save data to CSV file.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_names : list
        List of feature names
    file_path : str
        Path to save CSV file
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


def get_data_info(X, y, feature_names=None):
    """
    Get information about the dataset.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_names : list, optional
        List of feature names
    
    Returns:
    --------
    info : dict
        Dictionary with dataset information
    """
    info = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'feature_names': feature_names if feature_names else [f'feature_{i+1}' for i in range(X.shape[1])],
        'target_type': 'classification' if len(np.unique(y)) < 20 else 'regression',
        'n_classes': len(np.unique(y)) if len(np.unique(y)) < 20 else None,
        'target_distribution': pd.Series(y).value_counts().to_dict() if len(np.unique(y)) < 20 else None,
        'target_stats': {
            'mean': np.mean(y),
            'std': np.std(y),
            'min': np.min(y),
            'max': np.max(y)
        } if len(np.unique(y)) >= 20 else None
    }
    
    return info

