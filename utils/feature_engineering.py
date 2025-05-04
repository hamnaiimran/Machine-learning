import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA

def create_date_features(data, date_column):
    """
    Create features from date column
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    date_column: str
        Name of the date column
        
    Returns:
    --------
    pd.DataFrame
        Data with date features added
    """
    if data is None or data.empty:
        st.warning("No data provided for feature engineering.")
        return data
        
    # Make a copy to avoid modifying the original dataframe
    processed_data = data.copy()
    
    if date_column not in processed_data.columns:
        st.warning(f"Date column '{date_column}' not found in the data. Available columns: {', '.join(processed_data.columns)}")
        return processed_data
    
    # Make sure date column is datetime type
    try:
        processed_data[date_column] = pd.to_datetime(processed_data[date_column])
    except (ValueError, TypeError) as e:
        st.error(f"Error converting '{date_column}' to datetime. Please ensure the column contains valid date values. Error: {str(e)}")
        return processed_data
    except Exception as e:
        st.error(f"Unexpected error while processing date column: {str(e)}")
        return processed_data
    
    # Extract date features
    try:
        processed_data['year'] = processed_data[date_column].dt.year
        processed_data['month'] = processed_data[date_column].dt.month
        processed_data['day'] = processed_data[date_column].dt.day
        processed_data['day_of_week'] = processed_data[date_column].dt.dayofweek
        processed_data['quarter'] = processed_data[date_column].dt.quarter
    except Exception as e:
        st.error(f"Error extracting date features: {str(e)}")
        return processed_data
    
    return processed_data

def create_lag_features(data, column, lag_periods):
    """
    Create lag features for time series data
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    column: str
        Column to create lag features for
    lag_periods: list
        List of lag periods to create
        
    Returns:
    --------
    pd.DataFrame
        Data with lag features added
    """
    if data is None or data.empty:
        st.warning("No data provided for creating lag features.")
        return data
        
    if not lag_periods:
        st.warning("No lag periods provided.")
        return data
        
    # Make a copy to avoid modifying the original dataframe
    processed_data = data.copy()
    
    if column not in processed_data.columns:
        st.warning(f"Column '{column}' not found in the data. Available columns: {', '.join(processed_data.columns)}")
        return processed_data
    
    try:
        # Create lag features
        for lag in lag_periods:
            if not isinstance(lag, int) or lag <= 0:
                st.warning(f"Invalid lag period: {lag}. Lag periods must be positive integers.")
                continue
            processed_data[f'{column}_lag_{lag}'] = processed_data[column].shift(lag)
        
        # Drop rows with NaN values created by the lag operation
        processed_data = processed_data.dropna()
        
        if processed_data.empty:
            st.warning("All rows were dropped due to NaN values after creating lag features.")
            return data
            
        return processed_data
    except Exception as e:
        st.error(f"Error creating lag features: {str(e)}")
        return data

def create_rolling_features(data, column, windows):
    """
    Create rolling window features for time series data
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    column: str
        Column to create rolling features for
    windows: list
        List of window sizes to create
        
    Returns:
    --------
    pd.DataFrame
        Data with rolling features added
    """
    if data is None or data.empty:
        st.warning("No data provided for creating rolling features.")
        return data
        
    if not windows:
        st.warning("No window sizes provided.")
        return data
        
    # Make a copy to avoid modifying the original dataframe
    processed_data = data.copy()
    
    if column not in processed_data.columns:
        st.warning(f"Column '{column}' not found in the data. Available columns: {', '.join(processed_data.columns)}")
        return processed_data
    
    try:
        # Create rolling features
        for window in windows:
            if not isinstance(window, int) or window <= 0:
                st.warning(f"Invalid window size: {window}. Window sizes must be positive integers.")
                continue
            processed_data[f'{column}_rolling_mean_{window}'] = processed_data[column].rolling(window=window).mean()
            processed_data[f'{column}_rolling_std_{window}'] = processed_data[column].rolling(window=window).std()
        
        # Drop rows with NaN values created by the rolling operation
        processed_data = processed_data.dropna()
        
        if processed_data.empty:
            st.warning("All rows were dropped due to NaN values after creating rolling features.")
            return data
            
        return processed_data
    except Exception as e:
        st.error(f"Error creating rolling features: {str(e)}")
        return data

def select_features(data, target, feature_cols, method='f_regression', k=5):
    """
    Select top k features based on statistical tests
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    target: str
        Target column name
    feature_cols: list
        List of feature column names
    method: str
        Method to use for feature selection ('f_regression' or 'mutual_info')
    k: int
        Number of top features to select
        
    Returns:
    --------
    list, np.array
        List of selected feature names and their scores
    """
    if data is None or data.empty:
        st.warning("No data provided for feature selection.")
        return [], None
        
    if not target or target not in data.columns:
        st.warning(f"Target column '{target}' not found in the data. Available columns: {', '.join(data.columns)}")
        return [], None
        
    if not feature_cols:
        st.warning("No feature columns provided.")
        return [], None
        
    if not isinstance(k, int) or k <= 0:
        st.warning(f"Invalid value for k: {k}. k must be a positive integer.")
        return [], None
        
    if method not in ['f_regression', 'mutual_info']:
        st.warning(f"Invalid method: {method}. Must be either 'f_regression' or 'mutual_info'.")
        return [], None
        
    try:
        # Make a copy to avoid modifying the original dataframe
        X = data[feature_cols].copy()
        y = data[target].copy()
        
        # Choose the appropriate scoring function
        if method == 'f_regression':
            score_func = f_regression
        else:  # mutual_info
            score_func = mutual_info_regression
        
        # Apply the feature selection
        selector = SelectKBest(score_func=score_func, k=min(k, len(feature_cols)))
        selector.fit(X, y)
        
        # Get the selected feature names and scores
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
        feature_scores = selector.scores_
        
        # Create a dataframe of features and their scores
        scores_df = pd.DataFrame({
            'Feature': feature_cols,
            'Score': feature_scores
        })
        scores_df = scores_df.sort_values('Score', ascending=False)
        
        return selected_features, scores_df
    except Exception as e:
        st.error(f"Error selecting features: {str(e)}")
        return [], None

def apply_pca(data, columns, n_components=2):
    """
    Apply PCA for dimensionality reduction
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    columns: list
        List of columns to apply PCA to
    n_components: int
        Number of principal components to keep
        
    Returns:
    --------
    pd.DataFrame, PCA
        Data with PCA features and the PCA object
    """
    if data is None or data.empty:
        st.warning("No data provided for PCA.")
        return data, None
        
    if not columns:
        st.warning("No columns provided for PCA.")
        return data, None
        
    if not isinstance(n_components, int) or n_components <= 0:
        st.warning(f"Invalid number of components: {n_components}. Must be a positive integer.")
        return data, None
        
    try:
        # Make a copy to avoid modifying the original dataframe
        processed_data = data.copy()
        
        # Select numeric columns from the provided list
        numeric_cols = [col for col in columns if col in processed_data.columns 
                        and pd.api.types.is_numeric_dtype(processed_data[col])]
        
        if not numeric_cols:
            st.warning("No numeric columns found for PCA.")
            return processed_data, None
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(numeric_cols)))
        pca_result = pca.fit_transform(processed_data[numeric_cols])
        
        # Create a dataframe of PCA results
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
        )
        
        # Add the PCA components to the original dataframe
        for col in pca_df.columns:
            processed_data[col] = pca_df[col].values
        
        return processed_data, pca
    except Exception as e:
        st.error(f"Error applying PCA: {str(e)}")
        return data, None
