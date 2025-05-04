# Financial Machine Learning Pipeline

## Overview
This interactive Streamlit application provides a step-by-step guided machine learning workflow for financial data analysis. Users can upload their own financial datasets or fetch real-time stock data from Yahoo Finance, preprocess the data, create custom features, train machine learning models, and visualize the results.

## Features
- **Data Sources**: 
  - Upload your own financial datasets (CSV format)
  - Fetch real-time stock market data using Yahoo Finance API
- **Machine Learning Models**:
  - Linear Regression
  - Logistic Regression
  - K-Means Clustering
- **Interactive ML Pipeline**:
  - Step 1: Data Loading
  - Step 2: Preprocessing
  - Step 3: Feature Engineering
  - Step 4: Train/Test Split
  - Step 5: Model Training
  - Step 6: Evaluation
  - Step 7: Results Visualization
- **Visualizations**:
  - Interactive Plotly charts
  - Feature importance visualizations
  - Model performance metrics
  - Cluster visualizations
- **Features**:
  - Step-by-step guided workflow
  - Button-based navigation
  - Success/info notifications
  - Download results

## Installation and Setup
1. Clone this repository
2. Install the required packages:
   ```
   pip install streamlit pandas numpy scikit-learn matplotlib plotly yfinance
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py --server.port 5000
   ```

## How to Use
1. Start by selecting a data source in the sidebar:
   - Upload a financial dataset (CSV format)
   - Fetch stock market data from Yahoo Finance
2. Choose a machine learning model
3. Follow the step-by-step ML pipeline:
   - Preprocess your data (handle missing values, outliers, etc.)
   - Engineer relevant features
   - Split data into training and testing sets
   - Train your selected ML model
   - Evaluate model performance
   - Visualize and interpret results
4. Download results for further analysis

## Requirements
- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- yfinance
