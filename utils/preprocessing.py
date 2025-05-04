import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def check_missing_values(data):
    """
    Check for missing values in the data
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
        
    Returns:
    --------
    pd.DataFrame
        Summary of missing values
    """
    missing_values = data.isnull().sum()
    missing_percent = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': missing_percent
    })
    # Only return columns with missing values
    return missing_df[missing_df['Missing Values'] > 0]

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data with missing values
    strategy: str
        Strategy for handling missing values ('mean', 'median', 'most_frequent', 'drop')
        
    Returns:
    --------
    pd.DataFrame
        Data with missing values handled
    """
    # Make a copy to avoid modifying the original dataframe
    processed_data = data.copy()
    
    if strategy == 'drop':
        # Drop rows with missing values
        processed_data = processed_data.dropna()
        return processed_data
    
    # Handle numeric and categorical columns separately
    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
    categorical_cols = processed_data.select_dtypes(include=['object']).columns
    
    # Handle numeric columns
    if len(numeric_cols) > 0:
        if strategy in ['mean', 'median']:
            imputer = SimpleImputer(strategy=strategy)
            processed_data[numeric_cols] = imputer.fit_transform(processed_data[numeric_cols])
    
    # Handle categorical columns
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        processed_data[categorical_cols] = cat_imputer.fit_transform(processed_data[categorical_cols])
    
    return processed_data

def remove_outliers(data, columns, method='iqr', threshold=1.5):
    """
    Remove outliers from the data
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    columns: list
        List of column names to check for outliers
    method: str
        Method to use for outlier detection ('iqr' or 'zscore')
    threshold: float
        Threshold for outlier detection
        
    Returns:
    --------
    pd.DataFrame
        Data with outliers removed
    """
    # Make a copy to avoid modifying the original dataframe
    processed_data = data.copy()
    
    for col in columns:
        if col not in processed_data.columns or not pd.api.types.is_numeric_dtype(processed_data[col]):
            continue
            
        if method == 'iqr':
            # IQR method
            Q1 = processed_data[col].quantile(0.25)
            Q3 = processed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            processed_data = processed_data[(processed_data[col] >= lower_bound) & 
                                          (processed_data[col] <= upper_bound)]
        elif method == 'zscore':
            # Z-score method
            mean = processed_data[col].mean()
            std = processed_data[col].std()
            z_scores = abs((processed_data[col] - mean) / std)
            processed_data = processed_data[z_scores <= threshold]
    
    return processed_data

def normalize_data(data, columns, method='standard'):
    """
    Normalize numeric columns in the data
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    columns: list
        List of column names to normalize
    method: str
        Method to use for normalization ('standard' or 'minmax')
        
    Returns:
    --------
    pd.DataFrame, object
        Normalized data and the scaler object
    """
    # Make a copy to avoid modifying the original dataframe
    processed_data = data.copy()
    
    valid_columns = [col for col in columns if col in processed_data.columns 
                    and pd.api.types.is_numeric_dtype(processed_data[col])]
    
    if not valid_columns:
        return processed_data, None
    
    if method == 'standard':
        scaler = StandardScaler()
    else:  # minmax
        scaler = MinMaxScaler()
    
    processed_data[valid_columns] = scaler.fit_transform(processed_data[valid_columns])
    
    return processed_data, scaler

def encode_categorical(data, columns):
    """
    Encode categorical variables
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    columns: list
        List of column names to encode
        
    Returns:
    --------
    pd.DataFrame
        Data with encoded categorical variables
    """
    # Make a copy to avoid modifying the original dataframe
    processed_data = data.copy()
    
    for col in columns:
        if col in processed_data.columns and pd.api.types.is_object_dtype(processed_data[col]):
            # Use pandas get_dummies for one-hot encoding
            dummies = pd.get_dummies(processed_data[col], prefix=col, drop_first=True)
            processed_data = pd.concat([processed_data, dummies], axis=1)
            processed_data = processed_data.drop(col, axis=1)
    
    return processed_data
