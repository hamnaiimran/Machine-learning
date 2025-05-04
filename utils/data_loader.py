import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import datetime

def load_csv_data(uploaded_file):
    """
    Load data from an uploaded CSV file
    
    Parameters:
    -----------
    uploaded_file: StreamlitUploadedFile
        The uploaded CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the loaded data
    """
    if uploaded_file is None:
        st.warning("No file was uploaded.")
        return None
        
    try:
        data = pd.read_csv(uploaded_file)
        if data.empty:
            st.warning("The uploaded file is empty.")
            return None
        return data
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or contains no data.")
        return None
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please ensure the file is properly formatted.")
        return None
    except UnicodeDecodeError:
        st.error("Error reading the file. Please ensure the file is encoded in UTF-8.")
        return None
    except Exception as e:
        st.error(f"Unexpected error while loading the CSV file: {str(e)}")
        return None

def fetch_yahoo_finance_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    -----------
    ticker: str
        Stock ticker symbol
    start_date: datetime
        Start date for fetching data
    end_date: datetime
        End date for fetching data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the stock data
    """
    if not ticker:
        st.warning("No ticker symbol provided.")
        return None
        
    if not start_date or not end_date:
        st.warning("Please provide both start and end dates.")
        return None
        
    if not isinstance(start_date, datetime.datetime) or not isinstance(end_date, datetime.datetime):
        st.warning("Start and end dates must be datetime objects.")
        return None
        
    if start_date > end_date:
        st.warning("Start date cannot be after end date.")
        return None
        
    if start_date > datetime.datetime.now():
        st.warning("Start date cannot be in the future.")
        return None
        
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Check if data is empty
        if data.empty:
            st.warning(f"No data found for ticker {ticker} in the specified date range ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}).")
            return None
            
        return data
    except yf.errors.YFinanceError as e:
        st.error(f"Error fetching data from Yahoo Finance: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error while fetching data: {str(e)}")
        return None

def get_summary_stats(data):
    """
    Generate summary statistics for the data
    
    Parameters:
    -----------
    data: pd.DataFrame
        The input data
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics of the data
    """
    if data is None or data.empty:
        st.warning("No data provided for summary statistics.")
        return None
        
    try:
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            st.warning("No numeric columns found in the data for summary statistics.")
            return None
            
        try:
            summary = numeric_data.describe()
            if summary.empty:
                st.warning("Could not generate summary statistics for the numeric columns.")
                return None
            return summary
        except Exception as e:
            st.error(f"Error calculating summary statistics: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error processing data for summary statistics: {str(e)}")
        return None
