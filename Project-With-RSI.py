#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import scipy
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

def price_info(api_key, window, ticker):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}'  
    response = requests.get(url)
    data = response.json()
    time_series = data['Time Series (Daily)']
    dates = sorted(time_series.keys(), reverse=True)[:window]  # Get the most recent 'window' dates

    result = []
    
    for i, date in enumerate(dates):
        adj_close = float(time_series[date]['5. adjusted close'])
        one_day_return = None if i == 0 else (adj_close / float(time_series[dates[i-1]]['5. adjusted close']) - 1) * 100
        three_day_return = None if i < 3 else (adj_close / float(time_series[dates[i-3]]['5. adjusted close']) - 1) * 100
        five_day_return = None if i < 5 else (adj_close / float(time_series[dates[i-5]]['5. adjusted close']) - 1) * 100
        
        result.append([ticker, date, adj_close, one_day_return, three_day_return, five_day_return])
        
    return result


# In[ ]:


def thresholds(data, target_date):
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=['Ticker', 'Date', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # Calculate the daily price changes
    df['Delta'] = df['Close'].diff()

    # Calculate the gains and losses
    df['Gain'] = df['Delta'].clip(lower=0)
    df['Loss'] = -df['Delta'].clip(upper=0)

    # Calculate the average gains and losses
    window = 14
    df['Avg Gain'] = df['Gain'].rolling(window=window, min_periods=1).mean()
    df['Avg Loss'] = df['Loss'].rolling(window=window, min_periods=1).mean()

    # Calculate the Relative Strength (RS)
    df['RS'] = df['Avg Gain'] / df['Avg Loss']

    # Calculate the Relative Strength Index (RSI)
    df['RSI'] = 100 - (100 / (1 + df['RS']))

    # Extract the RSI for the target date
    rsi_value = df.loc[target_date]['RSI'] if target_date in df.index else None

    return {'RSI': rsi_value}


# In[ ]:


def signals(data, target_date, thresholds):
    rsi = thresholds.get('RSI')
    
    if rsi is not None:
        if rsi <= 30:
            return {'trade': 'Buy'}
        elif rsi >= 70:
            return {'trade': 'Sell'}
    return {'trade': 'Do nothing'}


# In[ ]:


import math
def trades(data, target_date, signal):
    # Initialize an empty list to store the results
    results = []
    
    # Determine the action from the signal
    action = signal.get('trade')
    
    if action in ['Buy', 'Sell']:
        # Find the index for the target date
        target_index = next((i for i, item in enumerate(data) if item[1] == target_date), None)
        
        if target_index is not None and target_index + 5 < len(data):  
            # Get the closing prices for the target and future dates
            close_target = data[target_index][2]
            close_plus_1 = data[target_index + 1][2]  # Price 1 day later
            close_plus_3 = data[target_index + 3][2]  # Price 3 days later
            close_plus_5 = data[target_index + 5][2]  # Price 5 days later
            
            # Calculate returns based on the action
            if action == 'Buy':
                one_day_ret = math.log(close_plus_1 / close_target) if target_index + 1 < len(data) else None
                three_day_ret = math.log(close_plus_3 / close_target) if target_index + 3 < len(data) else None
                five_day_ret = math.log(close_plus_5 / close_target) if target_index + 5 < len(data) else None
            elif action == 'Sell':
                one_day_ret = math.log(close_target / close_plus_1) if target_index + 1 < len(data) else None
                three_day_ret = math.log(close_target / close_plus_3) if target_index + 3 < len(data) else None
                five_day_ret = math.log(close_target / close_plus_5) if target_index + 5 < len(data) else None
            
            # Append the results if the trade is possible
            if one_day_ret is not None and three_day_ret is not None and five_day_ret is not None:
                results.append([one_day_ret, three_day_ret, five_day_ret])
    
    return results


# In[ ]:


def trades1(data, thresholds):
    trade_results = {}

    for daily_data in data:
        target_date = daily_data[1]  # Assuming the date is the second element in the list
        signal = signals(data, target_date, thresholds)  # Get the signal for this date
        
        # If there's no signal or the data is not available, skip this date
        if signal.get('trade') in ["Data not available for the specified date", "Do nothing"]:
            continue
        
        target_index = next((i for i, item in enumerate(data) if item[1] == target_date), None)
        
        if target_index is not None and target_index + 5 < len(data):
            close_target = daily_data[2]
            close_plus_1 = data[target_index + 1][2]  # Price 1 day later
            close_plus_3 = data[target_index + 3][2]  # Price 3 days later
            close_plus_5 = data[target_index + 5][2]  # Price 3 days later

            one_day_ret = math.log(close_plus_1 / close_target) - 1
            three_day_ret = math.log(close_plus_3 / close_target) - 1
            five_day_ret = math.log(close_plus_5 / close_target) - 1

            if signal['trade'] == "Buy":
                # Results for a buy signal
                trade_info = [target_date, one_day_ret, three_day_ret, five_day_ret]
            elif signal['trade'] == "Sell":
                # Results for a sell signal (reversed returns)
                trade_info = [target_date, -one_day_ret, -three_day_ret, -five_day_ret]
            
            ticker = daily_data[0]  # Assuming the ticker is the first element in the list

            # Append to the trade_results under the ticker key
            if ticker in trade_results:
                trade_results[ticker].append(trade_info)
            else:
                trade_results[ticker] = [trade_info]
    
    return trade_results


# In[ ]:


from bs4 import BeautifulSoup

def tickers_nasdaq100():
    headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
    res=requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100",headers=headers)
    main_data=res.json()['data']['data']['rows']
    ticker_list = []

    for i in range(len(main_data)):
        ticker_list.append(main_data[i]['symbol'])
    return ticker_list
# Use the function and print the result
list_of_tickers = tickers_nasdaq100()
print(list_of_tickers)


# In[ ]:


import numpy as np

def testing(returns, k):
    # Ensure returns list is not empty
    if not returns:
        return "No returns data available for testing."
    
    # Flatten the list of returns if it's a list of lists (for 3-day and 5-day returns)
    flat_returns = [item for sublist in returns for item in sublist]
    
    # Convert to numpy array for ease of calculation
    returns_array = np.array(flat_returns)
    
    # Calculate the mean and standard deviation of the returns
    mean_returns = np.mean(returns_array)
    std_returns = np.std(returns_array)
    
    # Calculate the margin of error
    margin_of_error = k * (std_returns / np.sqrt(len(returns_array)))
    
    # Compute the confidence interval
    lower_bound = mean_returns - margin_of_error
    upper_bound = mean_returns + margin_of_error
    
    # Print the confidence interval
    print(f"Confidence Interval at {95 if k==2 else 99}%: ({lower_bound}, {upper_bound})")
    return lower_bound, upper_bound


# In[ ]:


import pandas as pd


# [ticker, date, adjusted closing price, one-day return, 3-day return, 5-day return]

stock_data = price_info('PBBOCARQLUUHKQBA', 50, "META") 

dates = ['2024-02-12','2024-02-13','2024-02-14','2024-02-15','2024-02-16','2024-02-19','2024-02-20',
         '2024-02-21','2024-02-22','2024-02-23','2024-02-26','2024-02-27','2024-02-28','2024-02-29'
        ]


# Initialize a list to store the trade returns for statistical testing
all_trade_returns = []

for date in dates:
    # Compute thresholds for the current date
    current_thresholds = thresholds(stock_data, date)
    
    # Generate signals based on the current date and thresholds
    current_signal = signals(stock_data, date, current_thresholds)
    
    # Execute trades based on the current date and signals, and store the returns
    current_trades = trades(stock_data, date, current_signal)
    
    # Append the trade returns from the current day to the aggregate list
    all_trade_returns.extend(current_trades)

k = 2
testing(all_trade_returns, k)


# In[ ]:


api_key = "PBBOCARQLUUHKQBA"

# Initialize the summary dictionary
summary = {}

# Get the list of tickers
nasdaq_tickers = tickers_nasdaq100()

# Loop through each ticker and process the information
for ticker in nasdaq_tickers:
    # Get price data for the ticker
    price_data = price_info(api_key, 50, ticker)
    
    # Initialize a dictionary to hold all trades for the current ticker
    ticker_trades = {}
    
    # Process each day's price data for the current ticker
    for daily_data in price_data:
        target_date = daily_data[1]  # The date for the current data
        
        # Calculate thresholds for the current date
        current_thresholds = thresholds(price_data, target_date)
        
        # Generate signals based on the current date and thresholds
        current_signal = signals(price_data, target_date, current_thresholds)
        
        # Add the result of trades to the ticker_trades dictionary
        if current_signal['trade'] != "Data not available for the specified date" and current_signal['trade'] != "Do nothing":
            trades_data = trades1(price_data, current_thresholds)
            if trades_data:
                ticker_trades.update(trades_data)
    
    # After processing all dates for the ticker, add the results to the summary dictionary
    summary[ticker] = ticker_trades.get(ticker, [])
    


# In[ ]:


print(summary["AAPL"])


# In[ ]:


import numpy as np
from scipy.stats import t

def testing_cross_section(summary):
    # Extract one-day and three-day returns from the summary dictionary
    L1 = [trade[1] for ticker_trades in summary.values() for trade in ticker_trades if trade[1] is not None]
    L2 = [trade[2] for ticker_trades in summary.values() for trade in ticker_trades if trade[2] is not None]
    L3 = [trade[3] for ticker_trades in summary.values() for trade in ticker_trades if trade[2] is not None]
   
    # Define a function to calculate the confidence interval
    def calculate_confidence_interval(data, confidence=0.99):
        n = len(data)
        mean = np.mean(data)
        sem = scipy.stats.sem(data)
        margin_of_error = sem * t.ppf((1 + confidence) / 2., n-1)  # two-sided t-test
        return mean - margin_of_error, mean, mean + margin_of_error

    # Calculate confidence intervals for L1 and L2
    one_day = calculate_confidence_interval(L1)
    three_day = calculate_confidence_interval(L2)
    five_day = calculate_confidence_interval(L3)
    
    return one_day, three_day, five_day

testing_cross_section(summary)


# In[ ]:


import numpy as np
from scipy.stats import t

def testing_time_series(summary):
    results = {}

    # Define a function to calculate the lower bound of the confidence interval
    def calculate_lower_bound(data, confidence=0.99):
        n = len(data)
        mean = np.mean(data)
        sem = scipy.stats.sem(data)
        margin_of_error = sem * t.ppf((1 + confidence) / 2., n-1)  # two-sided t-test
        return mean - margin_of_error

    # Iterate over each ticker in the summary
    for ticker, trades in summary.items():
        # Extract one-day, three-day and five-day returns for the current ticker
        one_day_returns = [trade[1] for trade in trades if trade[1] is not None]
        three_day_returns = [trade[2] for trade in trades if trade[2] is not None]
        five_day_returns = [trade[3] for trade in trades if trade[2] is not None]
        
        # Calculate the lower bounds for one-day and three-day returns
        if one_day_returns:
            one_day_lower_bound = calculate_lower_bound(one_day_returns)
        else:
            one_day_lower_bound = None  # No data for one-day returns

        if three_day_returns:
            three_day_lower_bound = calculate_lower_bound(three_day_returns)
        else:
            three_day_lower_bound = None  # No data for three-day returns
        
        if five_day_returns:
            five_day_lower_bound = calculate_lower_bound(five_day_returns)
        else:
            five_day_lower_bound = None  # No data for five-day returns

        # Add the results to the dictionary
        results[ticker] = [one_day_lower_bound, three_day_lower_bound]

    return results

testing_time_series(summary)


# In[ ]:


def stock_picks(api_key, date):
    buy_tickers = []
    sell_tickers = []
    
    
    nasdaq_tickers = tickers_nasdaq100()
    
    for ticker in nasdaq_tickers:
       
        data = price_info(api_key, 50, "MSFT")  # Adjust window size as needed
        
        # Compute thresholds for the given date
        threshold_values = thresholds(data, date)
        
        # Compute signals for the given date
        signal = signals(data, date, threshold_values)
        
        # Check if the current close price results in a buy or sell
        # Assuming the signals function returns a dictionary with a 'trade' key
        trade_action = signal.get('trade')
        
        if trade_action == 'Buy':
            buy_tickers.append(ticker)
        elif trade_action == 'Sell':
            sell_tickers.append(ticker)
    
    return buy_tickers, sell_tickers


# In[ ]:


summary


# In[ ]:


# Extract all the dates from the summary dictionary
all_dates = [trade[0] for ticker_trades in summary.values() for trade in ticker_trades]

# Count the occurrences of each date
date_counts = Counter(all_dates)

# Convert to a dictionary {date: # of trades}
date_trades_dict = dict(date_counts)

# Plotting
dates = list(date_trades_dict.keys())
trade_counts = list(date_trades_dict.values())

plt.figure(figsize=(10, 6))
plt.bar(dates, trade_counts, color='skyblue')
plt.xlabel('Date')
plt.ylabel('Number of Trades')
plt.title('Number of Trades per Day')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# Initialize an investment amount
initial_investment = 100

# Define your list of 10 selected tickers
selected_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'NVDA', 'BABA', 'NFLX', 'INTC']  # Example tickers

# Extract all unique dates across the selected stocks
all_dates = sorted({date for ticker, trades in summary.items() if ticker in selected_tickers for trade in trades for date in [trade[0]]})

# Initialize a dictionary to hold the investment paths for each selected stock
investment_paths = {}

for ticker in selected_tickers:
    trades = summary.get(ticker, [])
    # Initialize a series with the initial investment for all dates
    investment_series = pd.Series(initial_investment, index=all_dates)
    # Correctly handling the unpacking of trades with more elements
    for trade in trades:
        trade_date, one_day_return = trade[0], trade[1]
        if one_day_return < 0:
            one_day_return *= -1
        investment_series.loc[trade_date:] *= one_day_return
    
    # Store the series in the dictionary
    investment_paths[ticker] = investment_series

# Plot the investment paths for the selected stocks
plt.figure(figsize=(14, 8))

for ticker in selected_tickers:
    if ticker in investment_paths:  # Check if the ticker has any trades
        plt.plot(investment_paths[ticker], label=ticker)

plt.legend()
plt.title("Investment Evolution for Selected Stocks")
plt.xlabel("Date")
plt.ylabel("Dollar Amount")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# Step 1: Compute the cumulative return series for each stock.
investment_paths = {}
initial_investment = 100
all_dates = sorted(set(trade[0] for trades in summary.values() for trade in trades))

for ticker, trades in summary.items():
    # Initialize the investment series with the initial investment amount for all dates
    investment_series = pd.Series(initial_investment, index=all_dates)
    for trade in trades:
        trade_date, one_day_return = trade[0], trade[1]
        if one_day_return < 0:
            one_day_return *= -1
        investment_series.loc[trade_date:] *= one_day_return
    investment_paths[ticker] = investment_series

# Step 2: Determine the final investment value to calculate BHR for each stock.
BHRs = {}
for ticker, investment_series in investment_paths.items():
    # BHR is the final value of the investment minus the initial value, divided by the initial value
    BHRs[ticker] = (investment_series.iloc[-1] - initial_investment) / initial_investment

# Step 3: Store the BHR for each stock in a list.
BHR_list = list(BHRs.values())

# Step 4: Plot the distribution of BHRs.
plt.figure(figsize=(10, 6))
plt.hist(BHR_list, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Buy-and-Hold Returns (BHR) for Nasdaq-100 Stocks')
plt.xlabel('Buy-and-Hold Return')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


# In[ ]:


# Sort the BHRs dictionary by its values to get the highest and lowest BHRs
sorted_BHRs = sorted(BHRs.items(), key=lambda item: item[1], reverse=True)

# Top 10 BHRs
top_10_BHRs = sorted_BHRs[:10]

# Bottom 10 BHRs
bottom_10_BHRs = sorted_BHRs[-10:]

print("Top 10 Stocks by BHR:")
for ticker, bhr in top_10_BHRs:
    print(f"{ticker}: {bhr:.2%}")

print("\nBottom 10 Stocks by BHR:")
for ticker, bhr in bottom_10_BHRs:
    print(f"{ticker}: {bhr:.2%}")

