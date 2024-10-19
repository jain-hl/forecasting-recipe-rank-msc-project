import pandas as pd
from sklearn.preprocessing import StandardScaler

import os; import sys; os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def normalize_data(X_train, X_test):
    """
    Normalizes the training and test datasets using StandardScaler.
    
    Parameters:
    X_train (pd.DataFrame): Training features.
    X_test (pd.DataFrame): Test features.
    
    Returns:
    X_train_scaled (pd.DataFrame): Normalized training features.
    X_test_scaled (pd.DataFrame): Normalized test features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to maintain the index and column names
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled

def custom_time_series_cv(X_train, n_validation_weeks=3):
    """
    Custom time series cross-validation generator for the final n menu weeks.
    
    Parameters:
    X_train (pd.DataFrame): Training data containing the 'menu_week' column.
    n_validation_weeks (int): Number of final weeks to be used for validation.
    
    Yields:
    train_index (pd.Index): The indices for the training set in each split.
    valid_index (pd.Index): The indices for the validation set in each split.
    """
    validation_weeks = X_train['menu_week'].unique()[-n_validation_weeks:]
    
    for valid_week in validation_weeks:
        # Split the data into training and validation based on the menu week
        train_df = X_train[X_train['menu_week'] < valid_week]
        valid_df = X_train[X_train['menu_week'] == valid_week]
        
        yield train_df.index, valid_df.index