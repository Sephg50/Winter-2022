# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:15:57 2022

@author: seph
"""

### Optimization of Moving Average Strategy Windows to Maximize BTC Cumulative Returns ###

import pandas as pd
import numpy as np
from math import exp
import time

start_time = time.time() # <-- imported time to calculate speed efficiency
print('Timing:')

BTC_df = pd.read_csv("BTC_USD_2014-11-03_2021-12-31-CoinDesk.csv",
                         parse_dates = True,
                         index_col = "Date").drop(['Currency', 
                                                   '24h Open (USD)', 
                                                   '24h High (USD)', 
                                                   '24h Low (USD)'], 1).rename(columns = {"Closing Price (USD)" : "BTC"})

BTC_df['daily_rets'] = BTC_df['BTC'].pct_change()
BTC_df['daily_logrets'] = np.log(BTC_df['daily_rets'] + 1)
BTC_df['cum_rets'] = np.cumprod(BTC_df['daily_rets'] + 1) - 1

# 1. Specify a range of windows for EMA and SMA using range()

"""
Create df with all possible SMA and EMA combinations for cumrets calculations in part 2 below:
"""

range_start = 10
SMA_range = 251
EMA_range = 101
increment = 1

for i in range(range_start,SMA_range,increment): 
    BTC_df[f'SMA_{i}'] = BTC_df['BTC'].rolling(i).mean()
    
for i in range(range_start,EMA_range,increment):
    BTC_df[f'modPrice_{i}'] = BTC_df['BTC']
    BTC_df[f'modPrice_{i}'][0:i] = BTC_df[f'SMA_{i}'][0:i]
    BTC_df[f'EMA_{i}'] = BTC_df[f'modPrice_{i}'].ewm(span=i, adjust=False).mean()

print("1. --- %s seconds ---" % (time.time() - start_time))

BTC_df = BTC_df.loc['2016-01-01':'2021-12-31']
    
# 2. Maximize cumulative rets - Long-Short Strategy EMA(i):SMA(j)

"""
Iterate through all combinations of SMA and EMA windows, 
calculating all possible cumulative rets for SMA windows > 100 and EMA windows < SMA windows,
results stored in cumrets_df['cumrets'],
EMA_window_xs and SMA_window_xs used to append window sizes to respective columns in cumrets_df.
"""

start_time = time.time()

cumrets_df = pd.DataFrame({'cumrets' : []})

EMA_window_xs = []
SMA_window_xs = []

for i in range(range_start,EMA_range,increment):
    for j in range(100,SMA_range,increment):
        if i < j:
            BTC_df['Position'] = np.where(BTC_df[f'EMA_{i}'] > BTC_df[f'SMA_{j}'],
                                                    1,
                                                    -1)
            BTC_df['Position'].mask(BTC_df[f'SMA_{j}'].isna(),np.nan,inplace=True)
            BTC_df['Position_logrets'] = BTC_df['daily_logrets'] * BTC_df['Position'].shift(1)
            cumrets_df.loc[f'EMA{i}_SMA{j}'] = exp(np.sum(BTC_df['Position_logrets']))
            EMA_window_xs.append(i)
            SMA_window_xs.append(j)
            
cumrets_df['EMA_window'] = EMA_window_xs
cumrets_df['SMA_window'] = SMA_window_xs

print("2. --- %s seconds ---\n" % (time.time() - start_time))

cumrets_df_sorted = cumrets_df.sort_values(by = 'cumrets', ascending = False)

print('Top 10 Combinations with minimum SMA window of 100:')
print(f' Range start = {range_start}\n EMA range = {EMA_range}\n SMA range = {SMA_range}\n Increment = {increment}')
print(f'{cumrets_df_sorted.head(10)}')