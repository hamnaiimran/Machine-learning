import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from plotly.subplots import make_subplots

def plot_missing_values(missing_data):
    """
    Plot missing values in the dataset
    
    Parameters:
    -----------
    missing_data: pd.DataFrame
        DataFrame containing missing value statistics
    """
    if missing_data.empty:
        st.info("No missing values found in the dataset.")
        return
        
    fig = px.bar(
        missing_data, 
        x=missing_data.index, 
        y='Percentage (%)',
        title='Missing Values by Column',
        labels={'index': 'Column', 'value': 'Percentage (%)'},
        color='Percentage (%)',
        color_continuous_scale='blues'
    )
    
    fig.update_layout(xaxis_title='Column', yaxis_title='Missing Values (%)')
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(data):
    """
    Plot correlation matrix for numeric columns
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.shape[1] < 2:
        st.info("Not enough numeric columns to compute correlations.")
        return
    
    # Compute correlation matrix
    corr = numeric_data.corr()
    
    # Create heatmap with Plotly
    fig = px.imshow(
        corr, 
        text_auto=True, 
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix'
    )
    
    fig.update_layout(height=500, width=700)
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(feature_importance_df, title='Feature Importance'):
    """
    Plot feature importance from a model
    
    Parameters:
    -----------
    feature_importance_df: pd.DataFrame
        DataFrame with columns 'Feature' and 'Coefficient'/'Importance'
    title: str
        Plot title
    """
    if feature_importance_df is None or feature_importance_df.empty:
        st.info("No feature importance data available.")
        return
    
    # Sort features by absolute importance
    if 'Coefficient' in feature_importance_df.columns:
        importance_col = 'Coefficient'
    else:
        importance_col = 'Importance'
    
    feature_importance_df['Abs_Importance'] = abs(feature_importance_df[importance_col])
    feature_importance_df = feature_importance_df.sort_values('Abs_Importance', ascending=False)
    
    # Plot feature importance
    fig = px.bar(
        feature_importance_df, 
        x=importance_col, 
        y='Feature',
        title=title,
        orientation='h',
        color=importance_col,
        color_continuous_scale='RdBu'
    )
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def plot_regression_results(y_test, y_pred, title='Actual vs Predicted Values'):
    """
    Plot regression results comparing actual vs predicted values
    
    Parameters:
    -----------
    y_test: array-like
        Actual values
    y_pred: array-like
        Predicted values
    title: str
        Plot title
    """
    # Create a dataframe with actual and predicted values
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    # Create a scatter plot
    fig = px.scatter(
        results_df, 
        x='Actual', 
        y='Predicted',
        title=title,
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        opacity=0.7
    )
    
    # Add 45-degree line
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], 
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    """
    Plot confusion matrix for classification results
    
    Parameters:
    -----------
    conf_matrix: array-like
        Confusion matrix
    title: str
        Plot title
    """
    # Create an annotated heatmap
    fig = px.imshow(
        conf_matrix,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=['Class 0', 'Class 1'],
        y=['Class 0', 'Class 1'],
        color_continuous_scale='blues',
        title=title
    )
    
    fig.update_layout(width=500, height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_clusters(X, clusters, centers=None, title='K-Means Clustering Results'):
    """
    Plot clustering results in 2D
    
    Parameters:
    -----------
    X: pd.DataFrame
        Feature data (should have at least 2 columns for visualization)
    clusters: array-like
        Cluster assignments
    centers: array-like
        Cluster centers
    title: str
        Plot title
    """
    # If X has more than 2 columns, use the first two for visualization
    if X.shape[1] > 2:
        viz_cols = X.columns[:2]
        st.info(f"Using only the first two features ({viz_cols[0]} and {viz_cols[1]}) for visualization.")
        plot_data = X[viz_cols].copy()
    else:
        plot_data = X.copy()
    
    # Add cluster labels to the data
    plot_data['Cluster'] = clusters
    
    # Create a scatter plot
    fig = px.scatter(
        plot_data, 
        x=plot_data.columns[0], 
        y=plot_data.columns[1],
        color='Cluster',
        title=title,
        labels={
            plot_data.columns[0]: plot_data.columns[0],
            plot_data.columns[1]: plot_data.columns[1]
        },
        color_continuous_scale='viridis'
    )
    
    # Add cluster centers if provided
    if centers is not None:
        if len(centers) > 0 and len(centers[0]) >= 2:
            center_x = centers[:, 0]
            center_y = centers[:, 1]
            
            fig.add_trace(
                go.Scatter(
                    x=center_x,
                    y=center_y,
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=12,
                        symbol='x'
                    ),
                    name='Cluster Centers'
                )
            )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_elbow_method(k_values, inertia, silhouette=None):
    """
    Plot the elbow method results for K-means
    
    Parameters:
    -----------
    k_values: array-like
        Number of clusters
    inertia: array-like
        Inertia values for each k
    silhouette: array-like
        Silhouette scores for each k
    """
    # Create subplot with 1 or 2 rows
    if silhouette is not None:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Elbow Method", "Silhouette Score")
        )
        
        # Add inertia trace
        fig.add_trace(
            go.Scatter(
                x=k_values, 
                y=inertia,
                mode='lines+markers',
                name='Inertia'
            ),
            row=1, col=1
        )
        
        # Add silhouette trace
        fig.add_trace(
            go.Scatter(
                x=k_values, 
                y=silhouette,
                mode='lines+markers',
                name='Silhouette Score',
                marker=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Optimal Number of Clusters',
            height=600
        )
        
        fig.update_xaxes(title_text='Number of Clusters (k)', row=1, col=1)
        fig.update_xaxes(title_text='Number of Clusters (k)', row=2, col=1)
        fig.update_yaxes(title_text='Inertia', row=1, col=1)
        fig.update_yaxes(title_text='Silhouette Score', row=2, col=1)
        
    else:
        fig = go.Figure()
        
        # Add inertia trace
        fig.add_trace(
            go.Scatter(
                x=k_values, 
                y=inertia,
                mode='lines+markers',
                name='Inertia'
            )
        )
        
        fig.update_layout(
            title='Elbow Method for Optimal k',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Inertia',
            height=400
        )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_train_test_split(train_size, test_size):
    """
    Visualize the train-test split using a pie chart
    
    Parameters:
    -----------
    train_size: int
        Size of the training set
    test_size: int
        Size of the testing set
    """
    labels = ['Training Set', 'Test Set']
    values = [train_size, test_size]
    colors = ['rgb(0, 128, 255)', 'rgb(255, 128, 0)']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker=dict(colors=colors)
    )])
    
    fig.update_layout(
        title='Train-Test Split',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_stock_data(data, title="Stock Price History"):
    """
    Plot stock price data
    
    Parameters:
    -----------
    data: pd.DataFrame
        Stock data with Date and price columns
    title: str
        Plot title
    """
    if data is None or data.empty:
        st.warning("No data provided for plotting.")
        return
        
    if 'Date' not in data.columns:
        st.warning("Date column not found in stock data. Please ensure your data contains a 'Date' column.")
        return
    
    try:
        # Make sure Date is datetime type
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Select price columns
        price_cols = [col for col in ['Close', 'Open', 'High', 'Low'] if col in data.columns]
        
        if not price_cols:
            st.warning("No price columns (Close, Open, High, Low) found in the data. Please ensure your data contains at least one price column.")
            return
        
        # Create a line plot for closing prices
        main_price = price_cols[0]  # Usually 'Close'
        
        fig = px.line(
            data,
            x='Date',
            y=main_price,
            title=title,
            labels={'x': 'Date', 'y': f'{main_price} Price'}
        )
        
        # Add volume if available
        if 'Volume' in data.columns:
            fig2 = px.bar(
                data,
                x='Date',
                y='Volume',
                title='Trading Volume',
                labels={'x': 'Date', 'y': 'Volume'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"An error occurred while plotting the stock data: {str(e)}")

def plot_pca_explained_variance(pca):
    """
    Plot explained variance ratio for PCA components
    
    Parameters:
    -----------
    pca: PCA
        Fitted PCA object
    """
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Create a dataframe for the plot
    pca_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
        'Explained Variance': explained_variance,
        'Cumulative Variance': cumulative_variance
    })
    
    # Create a subplot with 2 y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for individual explained variance
    fig.add_trace(
        go.Bar(
            x=pca_df['Component'],
            y=pca_df['Explained Variance'],
            name='Explained Variance'
        ),
        secondary_y=False
    )
    
    # Add line chart for cumulative variance
    fig.add_trace(
        go.Scatter(
            x=pca_df['Component'],
            y=pca_df['Cumulative Variance'],
            mode='lines+markers',
            name='Cumulative Variance',
            line=dict(color='red')
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='PCA Explained Variance',
        xaxis_title='Principal Component',
        height=400
    )
    
    fig.update_yaxes(title_text='Explained Variance', secondary_y=False)
    fig.update_yaxes(title_text='Cumulative Variance', secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
