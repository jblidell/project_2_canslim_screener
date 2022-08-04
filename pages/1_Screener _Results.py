import pandas as pd
from finvizfinance.screener.overview import Overview
import yfinance as yf
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report
import talib as TA
import numpy as np
import hvplot.pandas
import matplotlib.pyplot as plt
from datetime import date
import streamlit as st

st.set_page_config(page_title="Stock Screener Results")

st.markdown("# Currently Using the CANSLIM Method")
st.sidebar.header("Stock Screener")
st.write(
    """Here are the results of the screener that are updated daily.
    Not only are they consistenly updated, but the parameters are
    customizable"""
)
criteria = Overview()
filters_dic = {'Price':'Over $5', 'EPS growthqtr over qtr':'Over 20%', 'EPS growthpast 5 years':'Over 15%', 'InstitutionalOwnership':'Under 90%', 
               'Return on Equity':'Over +15%', '52-Week High/Low':'0-10% below High', 'Shares Outstanding': 'Under 50M', 'Price': 'Over $5', 'Average Volume': 'Over 100K'}
criteria.set_filter(filters_dict=filters_dic)
screened_stocks_df = criteria.screener_view()
st.dataframe(screened_stocks_df)
