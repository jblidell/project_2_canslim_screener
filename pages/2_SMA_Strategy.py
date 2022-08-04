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

st.set_page_config(page_title="SMA Trading Strategy")

st.markdown("# Trading when the 21 Day SMA crosses the 50 Day SMA")
st.sidebar.header("SMA Trading Strategy")
st.write(
    """Here are the results of the SMA Trading Strategy that are updated daily.
    Not only are they consistenly updated, but the parameters are
    customizable""")

criteria = Overview()
# Criteria Set from CANSLIM method
filters_dic = {'Price':'Over $5', 'EPS growthqtr over qtr':'Over 20%', 'EPS growthpast 5 years':'Over 15%', 'InstitutionalOwnership':'Under 90%', 
               'Return on Equity':'Over +15%', '52-Week High/Low':'0-10% below High', 'Shares Outstanding': 'Under 50M', 'Price': 'Over $5', 'Average Volume': 'Over 100K'}

criteria.set_filter(filters_dict=filters_dic)
screened_stocks_df = criteria.screener_view()
ticker = screened_stocks_df["Ticker"]



all_stock_df = {}
classification_report_dic = {}
testing_report_dic = {}
predictions_df_dic = {}

for ticker in ticker:
    stock = yf.Ticker(ticker)
    stock_hist = stock.history(period='5y')
    stock_hist = stock_hist.drop(columns=["Dividends", "Stock Splits", "Open", "High", "Low"])
    stock_hist['Ticker'] = (ticker)
    col = stock_hist.pop('Ticker')
    stock_hist.insert(loc=0, column='Ticker', value=col)
    stock_hist['Actual Returns'] = stock_hist['Close'].pct_change()
    stock_hist.dropna()
    stock_hist["SMA 21 Day"] = TA.SMA(stock_hist['Close'].values, timeperiod=21)
    stock_hist["SMA 50 Day"] = TA.SMA(stock_hist['Close'].values, timeperiod=50)
    stock_hist.dropna()
    stock_hist["Signal"]=0.0
    
    stock_hist.loc[(stock_hist["SMA 21 Day"] > stock_hist["SMA 50 Day"]), "Signal"] = 1.0
    stock_hist.loc[(stock_hist["SMA 21 Day"] < stock_hist["SMA 50 Day"]), "Signal"] = -1.0
        
    stock_hist["Entry/Exit"] = stock_hist["Signal"].diff()
    stock_ticker = stock_hist['Ticker']
    def buy_sell(stock_hist):
        signalBuy = []
        signalSell = []
        position = False 

        for i in range(len(stock_hist)):
            if stock_hist['SMA 21 Day'][i] > stock_hist['SMA 50 Day'][i]:
                if position == False :
                    signalBuy.append(stock_hist['Close'][i])
                    signalSell.append(np.nan)
                    position = True
                else:
                    signalBuy.append(np.nan)
                    signalSell.append(np.nan)
            elif stock_hist['SMA 21 Day'][i] < stock_hist['SMA 50 Day'][i]:
                if position == True:
                    signalBuy.append(np.nan)
                    signalSell.append(stock_hist['Close'][i])
                    position = False
                else:
                    signalBuy.append(np.nan)
                    signalSell.append(np.nan)
            else:
                signalBuy.append(np.nan)
                signalSell.append(np.nan)
        return pd.Series([signalBuy, signalSell])
    stock_hist['Buy_Signal_price'], stock_hist['Sell_Signal_price'] = buy_sell(stock_hist)
    
    fig, ax = plt.subplots(figsize=(20,8))
    ax.plot(stock_hist['Close'] , label = stock_ticker[0] ,linewidth=0.5, color='blue', alpha = 0.9)
    ax.plot(stock_hist['SMA 21 Day'], label = 'SMA 21 Day', alpha = 0.85)
    ax.plot(stock_hist['SMA 50 Day'], label = 'SMA 50 Day' , alpha = 0.85)
    ax.scatter(stock_hist.index , stock_hist['Buy_Signal_price'] , label = 'Buy' , marker = '^', color = 'green',alpha =1 )
    ax.scatter(stock_hist.index , stock_hist['Sell_Signal_price'] , label = 'Sell' , marker = 'v', color = 'red',alpha =1 )
    ax.set_title(stock_ticker[0] + " Price History with buy and sell signals",fontsize=10, backgroundcolor='blue', color='white')
    ax.set_xlabel(f'5 Year Chart of {stock_ticker[0]}' ,fontsize=18)
    ax.set_ylabel('Closing Price' , fontsize=18)
    legend = ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    

    X= stock_hist[['SMA 21 Day', 'SMA 50 Day']].shift().dropna().copy()
    y = stock_hist["Signal"].copy()
    training_begin = X.index.min()
    training_end = X.index.min() + DateOffset(months=12)
    
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]

    # Create testing datasets

    X_test = X.loc[training_end:]
    y_test = y.loc[training_end:]
    # Standerdize the data

    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    # Create classification report

    svm_model = svm.SVC()
    svm_model = svm_model.fit(X_train_scaled, y_train)

    training_signal_predictions = svm_model.predict(X_train_scaled)
    training_report = classification_report(y_train, training_signal_predictions, output_dict=True)
    classification_report_dic[ticker] = pd.DataFrame(training_report)
    
    # Backtest the machine learning algorithm
    
    testing_signal_predictions = svm_model.predict(X_test_scaled)
    testing_report = classification_report(y_test, testing_signal_predictions, output_dict=True)
    testing_report_dic[ticker] = pd.DataFrame(testing_report)
    
    # Compare actual and predicted returns

    predictions_df = pd.DataFrame(index=X_test.index)
    predictions_df["Predicted Signal"] = testing_signal_predictions
    predictions_df["Actual Returns"] = stock_hist["Actual Returns"]
    predictions_df["Trading Algorithm Returns"] = (
        stock_hist["Actual Returns"] * predictions_df["Predicted Signal"]
    )

    
    predictions_df_dic[ticker] = pd.DataFrame(predictions_df).head()
    
    ml_chart = (1 + predictions_df[["Actual Returns", "Trading Algorithm Returns"]]).cumprod()
    st.line_chart(ml_chart, width=20)
    # Set initial capital
    initial_capital = float(100000)

# Set the share size
    share_size = 500

# Buy a 500 share position when the dual moving average crossover Signal equals 1 (SMA50 is greater than SMA100)
# Sell a 500 share position when the dual moving average crossover Signal equals 0 (SMA50 is less than SMA100)
    stock_hist['Position'] = share_size * stock_hist['Signal']

# Determine the points in time where a 500 share position is bought or sold
    stock_hist['Entry/Exit Position'] = stock_hist['Position'].diff()

# Multiply the close price by the number of shares held, or the Position
    stock_hist['Portfolio Holdings'] = stock_hist['Close'] * stock_hist['Position']

# Subtract the amount of either the cost or proceeds of the trade from the initial capital invested
    stock_hist['Portfolio Cash'] = initial_capital - (stock_hist['Close'] * stock_hist['Entry/Exit Position']).cumsum()

# Calculate the total portfolio value by adding the portfolio cash to the portfolio holdings (or investments)
    stock_hist['Portfolio Total'] = stock_hist['Portfolio Cash'] + stock_hist['Portfolio Holdings']

# Calculate the portfolio daily returns
    stock_hist['Portfolio Daily Returns'] = stock_hist['Portfolio Total'].pct_change()

# Calculate the portfolio cumulative returns
    stock_hist['Portfolio Cumulative Returns'] = (1 + stock_hist['Portfolio Daily Returns']).cumprod() - 1

# Print the DataFrame
    print(f"The total return for {ticker} is {(stock_hist['Portfolio Cumulative Returns'][-1]*100)}")
    
    name = str(ticker)
    all_stock_df[name] = stock_hist
