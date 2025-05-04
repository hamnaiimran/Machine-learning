import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def split_data(data, target=None, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    target: str
        Target column name (None for clustering)
    test_size: float
        Proportion of the dataset to include in the test split
    random_state: int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test (for supervised learning)
        or X_train, X_test (for unsupervised learning)
    """
    if target is None or target not in data.columns:
        # For unsupervised learning, just split the data
        X_train, X_test = train_test_split(data, test_size=test_size, random_state=random_state)
        return X_train, X_test
    else:
        # For supervised learning
        X = data.drop(target, axis=1)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model
    
    Parameters:
    -----------
    X_train: pd.DataFrame
        Training features
    y_train: pd.Series
        Training target
        
    Returns:
    --------
    LinearRegression
        Trained linear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_linear_regression(model, X_test, y_test):
    """
    Evaluate a linear regression model
    
    Parameters:
    -----------
    model: LinearRegression
        Trained linear regression model
    X_test: pd.DataFrame
        Testing features
    y_test: pd.Series
        Testing target
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    coefficients = pd.DataFrame({
        'Feature': X_test.columns,
        'Coefficient': model.coef_
    })
    coefficients = coefficients.sort_values('Coefficient', ascending=False)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R-squared': r2,
        'Coefficients': coefficients,
        'Predicted': y_pred,
        'Actual': y_test
    }
    
    return metrics

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model
    
    Parameters:
    -----------
    X_train: pd.DataFrame
        Training features
    y_train: pd.Series
        Training target
        
    Returns:
    --------
    LogisticRegression
        Trained logistic regression model
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_logistic_regression(model, X_test, y_test):
    """
    Evaluate a logistic regression model
    
    Parameters:
    -----------
    model: LogisticRegression
        Trained logistic regression model
    X_test: pd.DataFrame
        Testing features
    y_test: pd.Series
        Testing target
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    try:
        # Get feature importance (coefficients)
        coefficients = pd.DataFrame({
            'Feature': X_test.columns,
            'Coefficient': model.coef_[0]
        })
        coefficients = coefficients.sort_values('Coefficient', ascending=False)
    except:
        coefficients = pd.DataFrame({
            'Feature': X_test.columns,
            'Coefficient': model.coef_.ravel()
        })
        coefficients = coefficients.sort_values('Coefficient', ascending=False)
    
    metrics = {
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
        'Coefficients': coefficients,
        'Predicted': y_pred,
        'Actual': y_test
    }
    
    return metrics

def train_kmeans(X_train, n_clusters=3):
    """
    Train a K-means clustering model
    
    Parameters:
    -----------
    X_train: pd.DataFrame
        Training features
    n_clusters: int
        Number of clusters
        
    Returns:
    --------
    KMeans
        Trained K-means model
    """
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X_train)
    return model

def evaluate_kmeans(model, X_test):
    """
    Evaluate a K-means clustering model
    
    Parameters:
    -----------
    model: KMeans
        Trained K-means model
    X_test: pd.DataFrame
        Testing features
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    clusters = model.predict(X_test)
    silhouette = silhouette_score(X_test, clusters) if len(np.unique(clusters)) > 1 else 0
    
    metrics = {
        'Silhouette Score': silhouette,
        'Inertia': model.inertia_,
        'Cluster Centers': model.cluster_centers_,
        'Predicted Clusters': clusters
    }
    
    return metrics

def get_optimal_clusters(X, max_clusters=10):
    """
    Find the optimal number of clusters using the elbow method
    
    Parameters:
    -----------
    X: pd.DataFrame
        Feature data
    max_clusters: int
        Maximum number of clusters to try
        
    Returns:
    --------
    list, list
        Number of clusters and corresponding inertia values
    """
    inertia = []
    silhouette_scores = []
    k_values = range(2, min(max_clusters+1, len(X)))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score
        clusters = kmeans.predict(X)
        if len(np.unique(clusters)) > 1:  # Silhouette requires at least 2 clusters
            silhouette_scores.append(silhouette_score(X, clusters))
        else:
            silhouette_scores.append(0)
    
    return list(k_values), inertia, silhouette_scores
