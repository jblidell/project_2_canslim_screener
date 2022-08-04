import pandas as pd
import yfinance as yf
import talib as ta
import numpy as np
import math

from finvizfinance.screener.overview import Overview
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

import streamlit as st

st.set_page_config(page_title="LSTM Predictions")

st.markdown("# Showing our Neural Network using LSTM")
st.sidebar.header("LSTM")
st.write(
    """Here are the results of the LSTM Prediction Model that are updated daily.
    Not only are they consistenly updated, but the parameters are
    customizable""")

criteria = Overview()

# Criteria Set from CANSLIM method
filters_dic = {'Price':'Over $5', 'EPS growthqtr over qtr':'Over 20%', 'EPS growthpast 5 years':'Over 15%', 'InstitutionalOwnership':'Under 90%', 
               'Return on Equity':'Over +15%', '52-Week High/Low':'0-10% below High', 'Shares Outstanding': 'Under 50M', 'Price': 'Over $5', 'Average Volume': 'Over 100K'}

criteria.set_filter(filters_dict=filters_dic)

screened_stocks_df = criteria.screener_view()
tickers = list(screened_stocks_df.Ticker)

all_stock_df = {}

for ticker in tickers:
    stock = yf.Ticker(ticker)
    stock_hist = stock.history(period='5y')
    stock_hist = stock_hist.drop(columns=["Dividends", "Stock Splits"])
    stock_hist['Ticker'] = (ticker)
    col = stock_hist.pop('Ticker')
    stock_hist.insert(loc=0, column='Ticker', value=col)
    stock_hist.dropna()
    all_stock_df[ticker] = pd.DataFrame(stock_hist)
    
    stock_chart = plt.figure()
    plt.title([ticker])
    plt.plot(all_stock_df[ticker]['Close'])
    plt.xlabel('Date')
    plt.ylabel('Prices ($)')
    st.pyplot(stock_chart)

    
    close_prices = all_stock_df[ticker]['Close']
    values = close_prices.values
    training_data_len = math.ceil(len(values)* 0.8)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(values.reshape(-1,1))
    train_data = scaled_data[0: training_data_len, :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    test_data = scaled_data[training_data_len-60: , : ]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size= 1, epochs=3)
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    rmse
    
    data = all_stock_df[ticker].filter(['Close'])
    train = data[:training_data_len]
    validation = data[training_data_len:]
    validation['Predictions'] = predictions
    lstm_chart = plt.figure()
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train)
    plt.plot(validation[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    st.pyplot(lstm_chart)