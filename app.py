# app.py - The Streamlit Web Application

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# --- Configuration ---
# Set the page configuration for the Streamlit app
st.set_page_config(layout="wide")

# Set the start and end dates for data fetching (used across the app)
START_DATE = '2012-01-01'
END_DATE = '2022-12-31'
WINDOW_SIZE = 100 # The number of previous days to use for prediction

# Load the trained model only once
@st.cache_resource
def load_the_model():
    """Load the pre-trained Keras model."""
    try:
        model = load_model('stock_predictions_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model. Make sure 'stock_predictions_model.keras' is in the same folder. Error: {e}")
        return None

# --- Main Streamlit Application ---
st.title('📈 Stock Price Prediction App (LSTM)')
st.markdown("---")

# 1. User Input
stock_symbol = st.text_input('Enter Stock Symbol (e.g., GOOG, AAPL, TSLA)', 'GOOG')

# 2. Data Fetching
try:
    # Get the raw data from Yahoo Finance
    data = yf.download(stock_symbol, start=START_DATE, end=END_DATE)
    data.reset_index(inplace=True)
    
    # Check if data is available
    if data.empty:
        st.warning(f"No data found for the symbol: **{stock_symbol}**. Please check the ticker.")
        st.stop()

except Exception as e:
    st.error(f"Failed to retrieve data for {stock_symbol}. Please check the symbol and your connection. Error: {e}")
    st.stop()


# 3. Data Preprocessing and Splitting
# Drop any null values
data.dropna(inplace=True)

# Split data into 80% train and 20% test
data_train_len = int(len(data) * 0.80)
data_train = pd.DataFrame(data['Close'][0:data_train_len])
data_test = pd.DataFrame(data['Close'][data_train_len:len(data)])

st.subheader(f'Stock Data for {stock_symbol} ({START_DATE} to {END_DATE})')
st.write(data.tail())
st.markdown("---")


# 4. Moving Average Analysis (Visualizations)
col1, col2 = st.columns(2)

# --- Chart 1: Price vs MA 50 ---
with col1:
    ma_50_days = data['Close'].rolling(WINDOW_SIZE).mean()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(data['Date'], ma_50_days, 'r', label='MA 50')
    ax1.plot(data['Date'], data['Close'], 'g', label='Closing Price')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.set_title(f'Price vs Moving Average (50 Days)')
    ax1.legend()
    st.pyplot(fig1)

# --- Chart 2: MA 100 vs MA 200 ---
with col2:
    ma_100_days = data['Close'].rolling(100).mean()
    ma_200_days = data['Close'].rolling(200).mean()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data['Date'], ma_100_days, 'r', label='MA 100')
    ax2.plot(data['Date'], ma_200_days, 'b', label='MA 200')
    ax2.plot(data['Date'], data['Close'], 'g', label='Closing Price')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    ax2.set_title(f'Price vs Moving Averages (100 & 200 Days)')
    ax2.legend()
    st.pyplot(fig2)

st.markdown("---")


# 5. Model Prediction Setup
st.subheader('Original Price vs. Predicted Price')

# Get the last 100 days from the training data and concatenate with the test data
past_100_days = data_train.tail(WINDOW_SIZE)
final_data = pd.concat([past_100_days, data_test], ignore_index=True)

# Scale the data using the *same scaler* that was fit on the training data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train) # Re-fit on only the training data
data_test_scale = scaler.transform(final_data) # Transform the combined data

# Create X_test (input sequences for prediction)
X_test = []
for i in range(WINDOW_SIZE, data_test_scale.shape[0]):
    X_test.append(data_test_scale[i-WINDOW_SIZE:i, 0])

X_test = np.array(X_test)
# Reshape for LSTM input
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Load model and predict
model = load_the_model()
if model:
    Y_predicted_scaled = model.predict(X_test)

    # Inverse Transform the scaled predictions back to actual dollar values
    scale_factor = 1/scaler.scale_[0] # The factor is calculated from the scaler
    Y_predicted = Y_predicted_scaled * scale_factor
    
    # Get the actual closing prices for the prediction period (for comparison)
    Y_actual = data_test.values * scale_factor # The original values of the test set are needed

    # 6. Final Visualization (Predicted vs. Actual)
    # Prepare DataFrame for final plot
    df_results = data_test.copy().reset_index()
    df_results['Actual Price'] = Y_actual.flatten()
    df_results['Predicted Price'] = Y_predicted.flatten()
    df_results = df_results.drop(columns=['Close', 'index'], errors='ignore')
    
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.plot(df_results.index, df_results['Actual Price'], 'g', label='Original Price')
    ax3.plot(df_results.index, df_results['Predicted Price'], 'r', label='Predicted Price')
    ax3.set_xlabel('Time (Test Period)')
    ax3.set_ylabel('Price')
    ax3.set_title('Original Price vs. Predicted Price (Test Set)')
    ax3.legend()
    st.pyplot(fig3)

    st.write("Predicted Prices (First 50):")
    st.dataframe(df_results.head(50))