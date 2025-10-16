#Stock Market Price Predictor

A machine learning web application built with Python that predicts stock market prices using LSTM (Long Short-Term Memory) neural networks. This end-to-end ML project fetches historical stock data and uses deep learning to forecast future stock prices.

🎯 Project Overview


This project creates a neural network capable of handling time series data to predict stock market prices based on historical trends. The application features an interactive web interface built with Streamlit, allowing users to analyze any stock by entering its ticker symbol.

✨ Features


Real-time Stock Data: Fetches historical stock data from Yahoo Finance

LSTM Neural Network: Uses deep learning for accurate time series predictions

Interactive Visualizations: Displays multiple charts including:

Price vs Moving Average (50 days)

Price vs MA50 vs MA100

Price vs MA100 vs MA200

Original Price vs Predicted Price

Web Interface: User-friendly Streamlit application

Custom Stock Analysis: Supports any stock symbol (US and Indian markets)

🛠️ Technologies Used


Python 3.x

TensorFlow/Keras: For building LSTM neural network

yfinance: For fetching stock market data

Pandas & NumPy: For data manipulation and processing

Matplotlib: For creating visualizations

Streamlit: For web application deployment

scikit-learn: For data preprocessing (MinMaxScaler)

