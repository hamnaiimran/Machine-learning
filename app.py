import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import base64
from io import BytesIO

# Import utility modules
from utils.data_loader import load_csv_data, fetch_yahoo_finance_data, get_summary_stats
from utils.preprocessing import (check_missing_values, handle_missing_values, 
                               remove_outliers, normalize_data, encode_categorical)
from utils.feature_engineering import (create_date_features, create_lag_features,
                                      create_rolling_features, select_features, apply_pca)
from utils.model_training import (split_data, train_linear_regression, evaluate_linear_regression,
                                 train_logistic_regression, evaluate_logistic_regression,
                                 train_kmeans, evaluate_kmeans, get_optimal_clusters)
from utils.visualization import (plot_missing_values, plot_correlation_matrix, plot_feature_importance,
                                plot_regression_results, plot_confusion_matrix, plot_clusters,
                                plot_elbow_method, plot_train_test_split, plot_stock_data,
                                plot_pca_explained_variance)

# Set page config
st.set_page_config(
    page_title="Financial ML Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'feature_engineered_data' not in st.session_state:
    st.session_state.feature_engineered_data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'numeric_features' not in st.session_state:
    st.session_state.numeric_features = None

# Define a function to reset the session state
def reset_session_state():
    st.session_state.data = None
    st.session_state.processed_data = None
    st.session_state.feature_engineered_data = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.model = None
    st.session_state.evaluation_results = None
    st.session_state.step = 1
    st.session_state.model_type = None
    st.session_state.data_source = None
    st.session_state.target_column = None
    st.session_state.features = None
    st.session_state.numeric_features = None

# Function to download data as CSV
def download_csv(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Define SVG content for welcome graphic
welcome_svg = """
<svg width="400" height="250" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#f0f2f6"/>
    <text x="50%" y="30%" font-family="Arial" font-size="24" text-anchor="middle" fill="#0068c9">Financial ML Pipeline</text>
    <path d="M50,180 L350,180" stroke="#262730" stroke-width="2"/>
    <path d="M50,120 L350,120" stroke="#dfe4ec" stroke-width="1" stroke-dasharray="5,5"/>
    <path d="M50,150 L350,150" stroke="#dfe4ec" stroke-width="1" stroke-dasharray="5,5"/>
    <path d="M50,90 L350,90" stroke="#dfe4ec" stroke-width="1" stroke-dasharray="5,5"/>
    <path d="M50,60 L350,60" stroke="#dfe4ec" stroke-width="1" stroke-dasharray="5,5"/>
    <path d="M50,180 L50,50" stroke="#262730" stroke-width="2"/>
    <path d="M50,180 L100,130 L150,160 L200,80 L250,120 L300,70 L350,100" stroke="#0068c9" stroke-width="3" fill="none"/>
    <circle cx="100" cy="130" r="4" fill="#0068c9"/>
    <circle cx="150" cy="160" r="4" fill="#0068c9"/>
    <circle cx="200" cy="80" r="4" fill="#0068c9"/>
    <circle cx="250" cy="120" r="4" fill="#0068c9"/>
    <circle cx="300" cy="70" r="4" fill="#0068c9"/>
    <circle cx="350" cy="100" r="4" fill="#0068c9"/>
</svg>
"""

# Define SVG content for ML workflow graphic
ml_workflow_svg = """
<svg width="700" height="200" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#f0f2f6"/>
    <rect x="10" y="80" width="100" height="50" rx="10" fill="#0068c9" opacity="0.8"/>
    <text x="60" y="110" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Data Loading</text>
    <rect x="130" y="80" width="100" height="50" rx="10" fill="#0068c9" opacity="0.8"/>
    <text x="180" y="110" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Preprocessing</text>
    <rect x="250" y="80" width="100" height="50" rx="10" fill="#0068c9" opacity="0.8"/>
    <text x="300" y="110" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Feature Engineering</text>
    <rect x="370" y="80" width="100" height="50" rx="10" fill="#0068c9" opacity="0.8"/>
    <text x="420" y="110" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Model Training</text>
    <rect x="490" y="80" width="100" height="50" rx="10" fill="#0068c9" opacity="0.8"/>
    <text x="540" y="110" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Evaluation</text>
    <rect x="610" y="80" width="80" height="50" rx="10" fill="#0068c9" opacity="0.8"/>
    <text x="650" y="110" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Results</text>
    <path d="M110,105 L130,105" stroke="#262730" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M230,105 L250,105" stroke="#262730" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M350,105 L370,105" stroke="#262730" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M470,105 L490,105" stroke="#262730" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M590,105 L610,105" stroke="#262730" stroke-width="2" marker-end="url(#arrow)"/>
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <path d="M0,0 L0,6 L9,3 Z" fill="#262730"/>
        </marker>
    </defs>
</svg>
"""

# Define main app function
def main():
    # Sidebar for navigation and data input
    with st.sidebar:
        st.title("Financial ML Pipeline")
        
        # Add option to reset the app
        if st.button("Reset App"):
            reset_session_state()
            st.experimental_rerun()
        
        # Data source selection
        st.header("1. Select Data Source")
        data_source = st.radio(
            "Choose data source:",
            options=["Upload Kragle Dataset", "Fetch Yahoo Finance Data"],
            index=0 if st.session_state.data_source == "Upload Kragle Dataset" else 1 if st.session_state.data_source == "Fetch Yahoo Finance Data" else 0
        )
        
        if data_source == "Upload Kragle Dataset":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None and (st.session_state.data is None or st.session_state.data_source != "Upload Kragle Dataset"):
                data = load_csv_data(uploaded_file)
                if data is not None:
                    st.session_state.data = data
                    st.session_state.data_source = "Upload Kragle Dataset"
                    st.session_state.step = 1
        
        elif data_source == "Fetch Yahoo Finance Data":
            ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT):", "AAPL")
            start_date = st.date_input("Start date:", datetime.date(2020, 1, 1))
            end_date = st.date_input("End date:", datetime.date.today())
            
            if st.button("Fetch Data"):
                with st.spinner("Fetching data from Yahoo Finance..."):
                    data = fetch_yahoo_finance_data(ticker, start_date, end_date)
                    if data is not None:
                        st.session_state.data = data
                        st.session_state.data_source = "Fetch Yahoo Finance Data"
                        st.session_state.step = 1
                        st.success(f"Successfully fetched data for {ticker}!")
        
        # Model selection (only shown when data is loaded)
        if st.session_state.data is not None:
            st.header("2. Select Model")
            model_type = st.selectbox(
                "Choose a machine learning model:",
                options=["Linear Regression", "Logistic Regression", "K-Means Clustering"],
                index=0 if st.session_state.model_type == "Linear Regression" else
                       1 if st.session_state.model_type == "Logistic Regression" else
                       2 if st.session_state.model_type == "K-Means Clustering" else 0
            )
            st.session_state.model_type = model_type
        
    # Main content
    st.title("Financial Machine Learning Pipeline")
    
    # Welcome screen when no data is loaded
    if st.session_state.data is None:
        st.markdown(f'<div style="text-align: center;">{welcome_svg}</div>', unsafe_allow_html=True)
        st.markdown("## Welcome to the Financial ML Pipeline! 📈")
        st.markdown("""
        This application guides you through a complete machine learning workflow for financial data analysis.
        
        ### Get Started:
        1. **Select a data source** from the sidebar (Upload a dataset or Fetch Yahoo Finance data)
        2. **Choose a machine learning model** once data is loaded
        3. **Follow the step-by-step ML pipeline** with interactive visualizations
        
        ### Available Models:
        - **Linear Regression**: Predict continuous financial values
        - **Logistic Regression**: Classify financial data into categories
        - **K-Means Clustering**: Discover natural groupings in financial data
        """)
        
        st.markdown(f'<div style="text-align: center;">{ml_workflow_svg}</div>', unsafe_allow_html=True)
        
        st.info("👈 Start by selecting a data source in the sidebar.")
        
        return
    
    # Display step navigation when data is loaded
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        step1_button = st.button("1. Load Data", 
                               use_container_width=True, 
                               type="primary" if st.session_state.step == 1 else "secondary")
    with col2:
        step2_button = st.button("2. Preprocessing", 
                               use_container_width=True, 
                               type="primary" if st.session_state.step == 2 else "secondary")
    with col3:
        step3_button = st.button("3. Feature Engineering", 
                               use_container_width=True, 
                               type="primary" if st.session_state.step == 3 else "secondary")
    with col4:
        step4_button = st.button("4. Train/Test Split", 
                               use_container_width=True, 
                               type="primary" if st.session_state.step == 4 else "secondary")
    with col5:
        step5_button = st.button("5. Model Training", 
                               use_container_width=True, 
                               type="primary" if st.session_state.step == 5 else "secondary")
    with col6:
        step6_button = st.button("6. Evaluation", 
                               use_container_width=True, 
                               type="primary" if st.session_state.step == 6 else "secondary")
    with col7:
        step7_button = st.button("7. Results", 
                               use_container_width=True, 
                               type="primary" if st.session_state.step == 7 else "secondary")
    
    # Update step based on button clicks
    if step1_button:
        st.session_state.step = 1
    elif step2_button:
        if st.session_state.data is not None:
            st.session_state.step = 2
        else:
            st.error("Please load data first.")
    elif step3_button:
        if st.session_state.processed_data is not None:
            st.session_state.step = 3
        else:
            st.error("Please complete preprocessing first.")
    elif step4_button:
        if st.session_state.feature_engineered_data is not None:
            st.session_state.step = 4
        else:
            st.error("Please complete feature engineering first.")
    elif step5_button:
        if st.session_state.X_train is not None:
            st.session_state.step = 5
        else:
            st.error("Please complete train/test split first.")
    elif step6_button:
        if st.session_state.model is not None:
            st.session_state.step = 6
        else:
            st.error("Please train a model first.")
    elif step7_button:
        if st.session_state.evaluation_results is not None:
            st.session_state.step = 7
        else:
            st.error("Please evaluate the model first.")
    
    # Step 1: Load Data
    if st.session_state.step == 1:
        st.header("Step 1: Load Data")
        
        if st.session_state.data is not None:
            st.success(f"Data loaded successfully from {st.session_state.data_source}!")
            
            # Display data info
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head())
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data Shape")
                st.info(f"Rows: {st.session_state.data.shape[0]}, Columns: {st.session_state.data.shape[1]}")
            
            with col2:
                st.subheader("Data Types")
                st.dataframe(pd.DataFrame({'Data Type': st.session_state.data.dtypes}))
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            summary_stats = get_summary_stats(st.session_state.data)
            if summary_stats is not None:
                st.dataframe(summary_stats)
            
            # For stock data, show price chart
            if st.session_state.data_source == "Fetch Yahoo Finance Data" and 'Date' in st.session_state.data.columns:
                st.subheader("Stock Price Chart")
                plot_stock_data(st.session_state.data)
            
            # Proceed to next step button
            if st.button("Proceed to Preprocessing"):
                st.session_state.step = 2
                st.experimental_rerun()
        else:
            st.info("Please select a data source from the sidebar.")
    
    # Step 2: Preprocessing
    elif st.session_state.step == 2:
        st.header("Step 2: Preprocessing")
        
        # Create tabs for different preprocessing tasks
        tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Outliers", "Normalization", "Categorical Encoding"])
        
        with tab1:
            st.subheader("Check for Missing Values")
            missing_data = check_missing_values(st.session_state.data)
            
            if missing_data.empty:
                st.success("No missing values found in the dataset.")
            else:
                st.write("Missing values in the dataset:")
                st.dataframe(missing_data)
                plot_missing_values(missing_data)
                
                # Handling missing values
                st.subheader("Handle Missing Values")
                missing_strategy = st.selectbox(
                    "Select strategy for handling missing values:",
                    options=["mean", "median", "most_frequent", "drop"],
                    index=0
                )
                
                if st.button("Apply Missing Value Handling"):
                    with st.spinner("Handling missing values..."):
                        processed_data = handle_missing_values(st.session_state.data, strategy=missing_strategy)
                        st.session_state.processed_data = processed_data
                        
                        # Check remaining missing values
                        remaining_missing = check_missing_values(processed_data)
                        if remaining_missing.empty:
                            st.success("All missing values have been handled!")
                        else:
                            st.warning("Some missing values still remain. Consider using a different strategy.")
                            st.dataframe(remaining_missing)
        
        with tab2:
            st.subheader("Handle Outliers")
            
            # Select columns for outlier detection
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            outlier_cols = st.multiselect(
                "Select columns to check for outliers:",
                options=numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if outlier_cols:
                outlier_method = st.selectbox(
                    "Select method for outlier detection:",
                    options=["iqr", "zscore"],
                    index=0
                )
                
                outlier_threshold = st.slider(
                    "Select threshold for outlier detection:",
                    min_value=1.0,
                    max_value=5.0,
                    value=1.5,
                    step=0.1
                )
                
                if st.button("Detect and Remove Outliers"):
                    with st.spinner("Detecting and removing outliers..."):
                        # Use processed data if it exists, otherwise use original data
                        input_data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
                        
                        processed_data = remove_outliers(
                            input_data, 
                            columns=outlier_cols, 
                            method=outlier_method,
                            threshold=outlier_threshold
                        )
                        
                        # Calculate number of rows removed
                        rows_removed = len(input_data) - len(processed_data)
                        
                        if rows_removed > 0:
                            st.warning(f"Removed {rows_removed} rows with outliers ({rows_removed/len(input_data):.2%} of data).")
                        else:
                            st.success("No outliers detected with the current threshold.")
                        
                        st.session_state.processed_data = processed_data
            else:
                st.info("Select columns to check for outliers.")
        
        with tab3:
            st.subheader("Normalize Data")
            
            # Select columns for normalization
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            normalize_cols = st.multiselect(
                "Select columns to normalize:",
                options=numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if normalize_cols:
                normalize_method = st.selectbox(
                    "Select normalization method:",
                    options=["standard", "minmax"],
                    index=0
                )
                
                if st.button("Normalize Data"):
                    with st.spinner("Normalizing data..."):
                        # Use processed data if it exists, otherwise use original data
                        input_data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
                        
                        processed_data, scaler = normalize_data(
                            input_data,
                            columns=normalize_cols,
                            method=normalize_method
                        )
                        
                        if scaler is not None:
                            st.success(f"Data normalized using {normalize_method} scaling.")
                            st.session_state.processed_data = processed_data
                        else:
                            st.error("Error normalizing data. Please check your selected columns.")
            else:
                st.info("Select columns to normalize.")
        
        with tab4:
            st.subheader("Encode Categorical Variables")
            
            # Select columns for encoding
            categorical_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols:
                encode_cols = st.multiselect(
                    "Select categorical columns to encode:",
                    options=categorical_cols,
                    default=categorical_cols
                )
                
                if encode_cols and st.button("Encode Categorical Variables"):
                    with st.spinner("Encoding categorical variables..."):
                        # Use processed data if it exists, otherwise use original data
                        input_data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
                        
                        processed_data = encode_categorical(input_data, columns=encode_cols)
                        
                        st.success(f"Encoded {len(encode_cols)} categorical variables.")
                        st.session_state.processed_data = processed_data
            else:
                st.info("No categorical columns found in the dataset.")
        
        # Display current state of processed data
        if st.session_state.processed_data is not None:
            st.subheader("Processed Data Preview")
            st.dataframe(st.session_state.processed_data.head())
            
            if st.checkbox("Show correlation matrix"):
                st.subheader("Correlation Matrix")
                plot_correlation_matrix(st.session_state.processed_data)
            
            # Proceed to next step button
            if st.button("Proceed to Feature Engineering"):
                st.session_state.step = 3
                st.experimental_rerun()
        else:
            st.info("Apply preprocessing steps to prepare the data.")
    
    # Step 3: Feature Engineering
    elif st.session_state.step == 3:
        st.header("Step 3: Feature Engineering")
        
        # Check if processed data exists
        if st.session_state.processed_data is None:
            st.error("Processed data not found. Please complete the preprocessing step first.")
            return
        
        # Create tabs for different feature engineering tasks
        tab1, tab2, tab3, tab4 = st.tabs(["Date Features", "Time Series Features", "Feature Selection", "Dimensionality Reduction"])
        
        with tab1:
            st.subheader("Create Date Features")
            
            # Identify potential date columns
            date_cols = []
            for col in st.session_state.processed_data.columns:
                if 'date' in col.lower() or 'time' in col.lower() or col.lower() == 'date':
                    date_cols.append(col)
            
            if date_cols:
                date_column = st.selectbox(
                    "Select date column:",
                    options=date_cols,
                    index=0
                )
                
                if st.button("Create Date Features"):
                    with st.spinner("Creating date features..."):
                        # Create date features
                        feature_data = create_date_features(st.session_state.processed_data, date_column)
                        
                        st.success("Date features created successfully!")
                        st.session_state.feature_engineered_data = feature_data
                        
                        # Show new columns
                        st.subheader("New Date Features")
                        new_cols = ['year', 'month', 'day', 'day_of_week', 'quarter']
                        if all(col in feature_data.columns for col in new_cols):
                            st.dataframe(feature_data[new_cols].head())
            else:
                st.info("No date columns detected in the dataset.")
        
        with tab2:
            st.subheader("Create Time Series Features")
            
            # Select columns for lag features
            numeric_cols = st.session_state.processed_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                lag_column = st.selectbox(
                    "Select column for time series features:",
                    options=numeric_cols,
                    index=0 if 'close' in [c.lower() for c in numeric_cols] else 0
                )
                
                # Lag features
                st.subheader("Lag Features")
                create_lags = st.checkbox("Create lag features")
                
                if create_lags:
                    lag_periods = st.multiselect(
                        "Select lag periods:",
                        options=[1, 2, 3, 5, 7, 10, 14, 21, 30],
                        default=[1, 5]
                    )
                
                # Rolling features
                st.subheader("Rolling Window Features")
                create_rolling = st.checkbox("Create rolling window features")
                
                if create_rolling:
                    rolling_windows = st.multiselect(
                        "Select rolling window sizes:",
                        options=[2, 3, 5, 7, 10, 14, 21, 30],
                        default=[5, 10]
                    )
                
                if st.button("Create Time Series Features"):
                    with st.spinner("Creating time series features..."):
                        # Start with processed data or previously feature engineered data
                        input_data = st.session_state.feature_engineered_data if st.session_state.feature_engineered_data is not None else st.session_state.processed_data
                        
                        if create_lags and lag_periods:
                            feature_data = create_lag_features(input_data, lag_column, lag_periods)
                            st.success(f"Created lag features for {lag_column}.")
                        else:
                            feature_data = input_data.copy()
                        
                        if create_rolling and rolling_windows:
                            feature_data = create_rolling_features(feature_data, lag_column, rolling_windows)
                            st.success(f"Created rolling window features for {lag_column}.")
                        
                        st.session_state.feature_engineered_data = feature_data
            else:
                st.info("No numeric columns found for time series feature creation.")
        
        with tab3:
            st.subheader("Feature Selection")
            
            # Get data for feature selection
            input_data = st.session_state.feature_engineered_data if st.session_state.feature_engineered_data is not None else st.session_state.processed_data
            
            # Select target variable for supervised learning
            if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                numeric_cols = input_data.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    target_column = st.selectbox(
                        "Select target variable:",
                        options=numeric_cols,
                        index=0
                    )
                    
                    # Save target column to session state
                    st.session_state.target_column = target_column
                    
                    # Select feature columns (exclude target)
                    feature_cols = [col for col in numeric_cols if col != target_column]
                    
                    if feature_cols:
                        selected_features = st.multiselect(
                            "Select features to use:",
                            options=feature_cols,
                            default=feature_cols[:min(10, len(feature_cols))]
                        )
                        
                        st.session_state.features = selected_features
                        
                        # Feature selection methods
                        if len(selected_features) > 1:
                            selection_method = st.selectbox(
                                "Feature selection method:",
                                options=["f_regression", "mutual_info"],
                                index=0
                            )
                            
                            top_k = st.slider(
                                "Number of top features to select:",
                                min_value=1,
                                max_value=len(selected_features),
                                value=min(5, len(selected_features))
                            )
                            
                            if st.button("Run Feature Selection"):
                                with st.spinner("Selecting top features..."):
                                    top_features, feature_scores = select_features(
                                        input_data,
                                        target=target_column,
                                        feature_cols=selected_features,
                                        method=selection_method,
                                        k=top_k
                                    )
                                    
                                    st.success(f"Selected top {len(top_features)} features.")
                                    
                                    # Display feature scores
                                    st.subheader("Feature Importance Scores")
                                    st.dataframe(feature_scores)
                                    
                                    # Save selected features
                                    st.session_state.features = top_features
                                    
                                    # Plot feature importance
                                    st.subheader("Feature Importance")
                                    plot_feature_importance(feature_scores, title=f"Feature Importance ({selection_method})")
                        else:
                            st.warning("Select at least 2 features for feature selection.")
                    else:
                        st.warning("Not enough numeric features available.")
                else:
                    st.warning("No numeric columns found for feature selection.")
            else:
                # For clustering, just select features
                numeric_cols = input_data.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    selected_features = st.multiselect(
                        "Select features for clustering:",
                        options=numeric_cols,
                        default=numeric_cols[:min(5, len(numeric_cols))]
                    )
                    
                    st.session_state.features = selected_features
                    st.session_state.numeric_features = numeric_cols
                else:
                    st.warning("No numeric columns found for clustering.")
        
        with tab4:
            st.subheader("Dimensionality Reduction (PCA)")
            
            # Get data for PCA
            input_data = st.session_state.feature_engineered_data if st.session_state.feature_engineered_data is not None else st.session_state.processed_data
            
            # Select columns for PCA
            numeric_cols = input_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                # Get features from session state or let user select
                if st.session_state.features is not None:
                    default_pca_cols = [col for col in st.session_state.features if col in numeric_cols]
                else:
                    default_pca_cols = numeric_cols[:min(5, len(numeric_cols))]
                
                pca_cols = st.multiselect(
                    "Select columns for PCA:",
                    options=numeric_cols,
                    default=default_pca_cols
                )
                
                if len(pca_cols) > 1:
                    n_components = st.slider(
                        "Number of principal components:",
                        min_value=1,
                        max_value=min(len(pca_cols), 10),
                        value=min(2, len(pca_cols))
                    )
                    
                    if st.button("Apply PCA"):
                        with st.spinner("Applying PCA..."):
                            pca_data, pca = apply_pca(input_data, pca_cols, n_components)
                            
                            if pca is not None:
                                st.success(f"PCA applied successfully with {n_components} components.")
                                st.session_state.feature_engineered_data = pca_data
                                
                                # Plot explained variance
                                st.subheader("PCA Explained Variance")
                                plot_pca_explained_variance(pca)
                            else:
                                st.error("Error applying PCA. Please check your selected columns.")
                else:
                    st.warning("Select at least 2 columns for PCA.")
            else:
                st.info("Not enough numeric columns for PCA (need at least 2).")
        
        # Display current state of feature engineered data
        if st.session_state.feature_engineered_data is not None:
            st.subheader("Feature Engineered Data Preview")
            st.dataframe(st.session_state.feature_engineered_data.head())
            
            # Proceed to next step button
            if st.button("Proceed to Train/Test Split"):
                st.session_state.step = 4
                st.experimental_rerun()
        else:
            # If no feature engineering has been done yet, use processed data
            if st.session_state.processed_data is not None:
                st.info("No feature engineering applied yet. You can continue with the processed data.")
                
                if st.button("Use processed data and proceed to Train/Test Split"):
                    st.session_state.feature_engineered_data = st.session_state.processed_data
                    st.session_state.step = 4
                    st.experimental_rerun()
            else:
                st.error("Processed data not found. Please complete the preprocessing step first.")
    
    # Step 4: Train/Test Split
    elif st.session_state.step == 4:
        st.header("Step 4: Train/Test Split")
        
        # Check if feature engineered data exists
        if st.session_state.feature_engineered_data is None:
            st.error("Feature engineered data not found. Please complete the feature engineering step first.")
            return
        
        # For supervised learning models
        if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
            # Check if target column is selected
            if st.session_state.target_column is None:
                st.error("Target column not selected. Please go back to the Feature Engineering step.")
                return
            
            # Check if features are selected
            if st.session_state.features is None or len(st.session_state.features) == 0:
                st.error("No features selected. Please go back to the Feature Engineering step.")
                return
            
            # Configure train/test split
            test_size = st.slider(
                "Test set size (%):",
                min_value=10,
                max_value=50,
                value=20,
                step=5
            ) / 100
            
            random_state = st.number_input(
                "Random state (for reproducibility):",
                min_value=0,
                max_value=1000,
                value=42,
                step=1
            )
            
            if st.button("Split Data"):
                with st.spinner("Splitting data into training and testing sets..."):
                    # Prepare data for splitting
                    data = st.session_state.feature_engineered_data.copy()
                    
                    # Select only required columns (features + target)
                    selected_cols = st.session_state.features + [st.session_state.target_column]
                    data = data[selected_cols].dropna()
                    
                    # Split data
                    X_train, X_test, y_train, y_test = split_data(
                        data,
                        target=st.session_state.target_column,
                        test_size=test_size,
                        random_state=random_state
                    )
                    
                    # Save to session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    st.success("Data split successfully!")
        
        # For clustering
        else:  # K-Means Clustering
            # Check if features are selected
            if st.session_state.features is None or len(st.session_state.features) == 0:
                st.error("No features selected. Please go back to the Feature Engineering step.")
                return
            
            # Configure train/test split
            test_size = st.slider(
                "Test set size (%):",
                min_value=10,
                max_value=50,
                value=20,
                step=5
            ) / 100
            
            random_state = st.number_input(
                "Random state (for reproducibility):",
                min_value=0,
                max_value=1000,
                value=42,
                step=1
            )
            
            if st.button("Split Data"):
                with st.spinner("Splitting data for clustering..."):
                    # Prepare data for splitting
                    data = st.session_state.feature_engineered_data.copy()
                    
                    # Select only required columns (features)
                    data = data[st.session_state.features].dropna()
                    
                    # Split data for clustering (unsupervised)
                    X_train, X_test = split_data(
                        data,
                        target=None,
                        test_size=test_size,
                        random_state=random_state
                    )
                    
                    # Save to session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = None
                    st.session_state.y_test = None
                    
                    st.success("Data split successfully for clustering!")
        
        # Display train/test split info when available
        if st.session_state.X_train is not None and st.session_state.X_test is not None:
            st.subheader("Train/Test Split Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"Training set: {st.session_state.X_train.shape[0]} samples")
                st.dataframe(st.session_state.X_train.head())
            
            with col2:
                st.info(f"Testing set: {st.session_state.X_test.shape[0]} samples")
                st.dataframe(st.session_state.X_test.head())
            
            # Visualize the split
            st.subheader("Train/Test Split Visualization")
            plot_train_test_split(
                train_size=st.session_state.X_train.shape[0],
                test_size=st.session_state.X_test.shape[0]
            )
            
            # Proceed to next step button
            if st.button("Proceed to Model Training"):
                st.session_state.step = 5
                st.experimental_rerun()
    
    # Step 5: Model Training
    elif st.session_state.step == 5:
        st.header("Step 5: Model Training")
        
        # Check if data is split
        if st.session_state.X_train is None or st.session_state.X_test is None:
            st.error("Train/test data not found. Please complete the train/test split step first.")
            return
        
        # Linear Regression
        if st.session_state.model_type == "Linear Regression":
            st.subheader("Linear Regression Model")
            
            if st.button("Train Linear Regression Model"):
                with st.spinner("Training Linear Regression model..."):
                    model = train_linear_regression(
                        st.session_state.X_train,
                        st.session_state.y_train
                    )
                    
                    # Save model to session state
                    st.session_state.model = model
                    
                    st.success("Linear Regression model trained successfully!")
        
        # Logistic Regression
        elif st.session_state.model_type == "Logistic Regression":
            st.subheader("Logistic Regression Model")
            
            # Check if target is binary
            unique_values = st.session_state.y_train.nunique()
            if unique_values != 2:
                st.warning(f"Logistic Regression expects a binary target. Your target has {unique_values} unique values.")
            
            if st.button("Train Logistic Regression Model"):
                with st.spinner("Training Logistic Regression model..."):
                    model = train_logistic_regression(
                        st.session_state.X_train,
                        st.session_state.y_train
                    )
                    
                    # Save model to session state
                    st.session_state.model = model
                    
                    st.success("Logistic Regression model trained successfully!")
        
        # K-Means Clustering
        else:  # K-Means Clustering
            st.subheader("K-Means Clustering")
            
            # Find optimal number of clusters
            if st.checkbox("Find optimal number of clusters"):
                max_clusters = st.slider(
                    "Maximum number of clusters to try:",
                    min_value=2,
                    max_value=20,
                    value=10,
                    step=1
                )
                
                if st.button("Run Elbow Method"):
                    with st.spinner("Finding optimal number of clusters..."):
                        k_values, inertia, silhouette_scores = get_optimal_clusters(
                            st.session_state.X_train,
                            max_clusters=max_clusters
                        )
                        
                        st.success("Elbow method completed. Analyze the plots below to select the optimal number of clusters.")
                        
                        # Plot elbow method results
                        plot_elbow_method(k_values, inertia, silhouette_scores)
            
            # Train K-Means model
            n_clusters = st.slider(
                "Number of clusters:",
                min_value=2,
                max_value=20,
                value=3,
                step=1
            )
            
            if st.button("Train K-Means Clustering Model"):
                with st.spinner("Training K-Means Clustering model..."):
                    model = train_kmeans(
                        st.session_state.X_train,
                        n_clusters=n_clusters
                    )
                    
                    # Save model to session state
                    st.session_state.model = model
                    
                    st.success(f"K-Means model trained with {n_clusters} clusters!")
        
        # When model is trained
        if st.session_state.model is not None:
            st.info("Model trained successfully. Proceed to evaluation.")
            
            # Proceed to next step button
            if st.button("Proceed to Evaluation"):
                st.session_state.step = 6
                st.experimental_rerun()
    
    # Step 6: Evaluation
    elif st.session_state.step == 6:
        st.header("Step 6: Model Evaluation")
        
        # Check if model is trained
        if st.session_state.model is None:
            st.error("Model not found. Please complete the model training step first.")
            return
        
        # Linear Regression evaluation
        if st.session_state.model_type == "Linear Regression":
            st.subheader("Linear Regression Evaluation")
            
            if st.button("Evaluate Linear Regression Model"):
                with st.spinner("Evaluating Linear Regression model..."):
                    metrics = evaluate_linear_regression(
                        st.session_state.model,
                        st.session_state.X_test,
                        st.session_state.y_test
                    )
                    
                    # Save evaluation results to session state
                    st.session_state.evaluation_results = metrics
                    
                    st.success("Model evaluation completed!")
        
        # Logistic Regression evaluation
        elif st.session_state.model_type == "Logistic Regression":
            st.subheader("Logistic Regression Evaluation")
            
            if st.button("Evaluate Logistic Regression Model"):
                with st.spinner("Evaluating Logistic Regression model..."):
                    metrics = evaluate_logistic_regression(
                        st.session_state.model,
                        st.session_state.X_test,
                        st.session_state.y_test
                    )
                    
                    # Save evaluation results to session state
                    st.session_state.evaluation_results = metrics
                    
                    st.success("Model evaluation completed!")
        
        # K-Means Clustering evaluation
        else:  # K-Means Clustering
            st.subheader("K-Means Clustering Evaluation")
            
            if st.button("Evaluate K-Means Clustering Model"):
                with st.spinner("Evaluating K-Means Clustering model..."):
                    metrics = evaluate_kmeans(
                        st.session_state.model,
                        st.session_state.X_test
                    )
                    
                    # Save evaluation results to session state
                    st.session_state.evaluation_results = metrics
                    
                    st.success("Model evaluation completed!")
        
        # Display evaluation results when available
        if st.session_state.evaluation_results is not None:
            st.subheader("Evaluation Results")
            
            # Linear Regression results
            if st.session_state.model_type == "Linear Regression":
                metrics = st.session_state.evaluation_results
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MSE", f"{metrics['MSE']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                with col3:
                    st.metric("R-squared", f"{metrics['R-squared']:.4f}")
                
                # Feature importance
                st.subheader("Feature Importance")
                plot_feature_importance(metrics['Coefficients'], title="Linear Regression Coefficients")
                
                # Predictions vs Actual
                st.subheader("Predictions vs Actual Values")
                plot_regression_results(metrics['Actual'], metrics['Predicted'])
            
            # Logistic Regression results
            elif st.session_state.model_type == "Logistic Regression":
                metrics = st.session_state.evaluation_results
                
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(metrics['Confusion Matrix'])
                
                # Feature importance
                st.subheader("Feature Importance")
                plot_feature_importance(metrics['Coefficients'], title="Logistic Regression Coefficients")
            
            # K-Means Clustering results
            else:  # K-Means Clustering
                metrics = st.session_state.evaluation_results
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", f"{metrics['Silhouette Score']:.4f}")
                with col2:
                    st.metric("Inertia", f"{metrics['Inertia']:.2f}")
                
                # Cluster visualization
                st.subheader("Cluster Visualization")
                plot_clusters(
                    st.session_state.X_test,
                    metrics['Predicted Clusters'],
                    centers=metrics['Cluster Centers']
                )
            
            # Proceed to next step button
            if st.button("Proceed to Results"):
                st.session_state.step = 7
                st.experimental_rerun()
    
    # Step 7: Results
    elif st.session_state.step == 7:
        st.header("Step 7: Results and Conclusions")
        
        # Check if evaluation is done
        if st.session_state.evaluation_results is None:
            st.error("Evaluation results not found. Please complete the model evaluation step first.")
            return
        
        # Summarize the pipeline steps
        st.subheader("ML Pipeline Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Source:**", st.session_state.data_source)
            st.write("**Model Type:**", st.session_state.model_type)
            
            if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                st.write("**Target Variable:**", st.session_state.target_column)
            
            st.write("**Features:**", ", ".join(st.session_state.features))
            
            # Dataset stats
            st.write("**Dataset Size:**", f"{len(st.session_state.data)} samples")
            st.write("**Training Set Size:**", f"{len(st.session_state.X_train)} samples")
            st.write("**Testing Set Size:**", f"{len(st.session_state.X_test)} samples")
        
        with col2:
            # Model performance metrics
            st.subheader("Model Performance")
            
            if st.session_state.model_type == "Linear Regression":
                metrics = st.session_state.evaluation_results
                st.write("**Mean Squared Error (MSE):**", f"{metrics['MSE']:.4f}")
                st.write("**Root Mean Squared Error (RMSE):**", f"{metrics['RMSE']:.4f}")
                st.write("**R-squared (R²):**", f"{metrics['R-squared']:.4f}")
                
            elif st.session_state.model_type == "Logistic Regression":
                metrics = st.session_state.evaluation_results
                st.write("**Accuracy:**", f"{metrics['Accuracy']:.4f}")
                
            else:  # K-Means Clustering
                metrics = st.session_state.evaluation_results
                st.write("**Silhouette Score:**", f"{metrics['Silhouette Score']:.4f}")
                st.write("**Inertia:**", f"{metrics['Inertia']:.2f}")
                st.write("**Number of Clusters:**", len(metrics['Cluster Centers']))
        
        # Visualizations specific to model type
        st.subheader("Key Visualizations")
        
        if st.session_state.model_type == "Linear Regression":
            # Top coefficients
            st.subheader("Top Model Coefficients")
            metrics = st.session_state.evaluation_results
            plot_feature_importance(metrics['Coefficients'], title="Linear Regression Coefficients")
            
            # Predictions vs Actual
            st.subheader("Predictions vs Actual Values")
            plot_regression_results(metrics['Actual'], metrics['Predicted'])
            
        elif st.session_state.model_type == "Logistic Regression":
            # Feature importance
            st.subheader("Feature Importance")
            metrics = st.session_state.evaluation_results
            plot_feature_importance(metrics['Coefficients'], title="Logistic Regression Coefficients")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(metrics['Confusion Matrix'])
            
        else:  # K-Means Clustering
            # Cluster visualization
            st.subheader("Cluster Visualization")
            metrics = st.session_state.evaluation_results
            plot_clusters(
                st.session_state.X_test,
                metrics['Predicted Clusters'],
                centers=metrics['Cluster Centers']
            )
        
        # Final conclusions
        st.subheader("Conclusions")
        
        if st.session_state.model_type == "Linear Regression":
            r2 = st.session_state.evaluation_results['R-squared']
            if r2 > 0.7:
                st.success(f"The Linear Regression model shows strong predictive power with an R² of {r2:.4f}.")
            elif r2 > 0.5:
                st.info(f"The Linear Regression model shows moderate predictive power with an R² of {r2:.4f}.")
            else:
                st.warning(f"The Linear Regression model shows limited predictive power with an R² of {r2:.4f}. Consider feature engineering or trying different models.")
            
            # Most influential features
            coeffs = st.session_state.evaluation_results['Coefficients']
            top_positive = coeffs.nlargest(3, 'Coefficient')
            top_negative = coeffs.nsmallest(3, 'Coefficient')
            
            st.write("**Most positively influential features:**", ", ".join(top_positive['Feature'].tolist()))
            st.write("**Most negatively influential features:**", ", ".join(top_negative['Feature'].tolist()))
            
        elif st.session_state.model_type == "Logistic Regression":
            accuracy = st.session_state.evaluation_results['Accuracy']
            if accuracy > 0.8:
                st.success(f"The Logistic Regression model shows strong classification performance with an accuracy of {accuracy:.4f}.")
            elif accuracy > 0.6:
                st.info(f"The Logistic Regression model shows moderate classification performance with an accuracy of {accuracy:.4f}.")
            else:
                st.warning(f"The Logistic Regression model shows limited classification performance with an accuracy of {accuracy:.4f}. Consider feature engineering or trying different models.")
            
            # Most influential features
            coeffs = st.session_state.evaluation_results['Coefficients']
            top_features = coeffs.iloc[:3]
            
            st.write("**Most influential features for classification:**", ", ".join(top_features['Feature'].tolist()))
            
        else:  # K-Means Clustering
            silhouette = st.session_state.evaluation_results['Silhouette Score']
            if silhouette > 0.6:
                st.success(f"The K-Means clustering model shows well-separated clusters with a silhouette score of {silhouette:.4f}.")
            elif silhouette > 0.3:
                st.info(f"The K-Means clustering model shows moderately separated clusters with a silhouette score of {silhouette:.4f}.")
            else:
                st.warning(f"The K-Means clustering model shows poorly separated clusters with a silhouette score of {silhouette:.4f}. Consider feature engineering or trying a different number of clusters.")
        
        # Download options
        st.subheader("Download Results")
        
        if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
            # Create results dataframe
            results_df = pd.DataFrame({
                'Actual': st.session_state.evaluation_results['Actual'],
                'Predicted': st.session_state.evaluation_results['Predicted']
            })
            
            st.markdown(download_csv(results_df, "prediction_results.csv"), unsafe_allow_html=True)
            
            # Feature importance
            st.markdown(download_csv(st.session_state.evaluation_results['Coefficients'], "feature_importance.csv"), unsafe_allow_html=True)
            
        else:  # K-Means Clustering
            # Create results dataframe with cluster assignments
            cluster_df = st.session_state.X_test.copy()
            cluster_df['Cluster'] = st.session_state.evaluation_results['Predicted Clusters']
            
            st.markdown(download_csv(cluster_df, "cluster_results.csv"), unsafe_allow_html=True)
        
        # Reset button for starting over
        if st.button("Start Over"):
            reset_session_state()
            st.experimental_rerun()

# Run the app
if __name__ == "__main__":
    main()
