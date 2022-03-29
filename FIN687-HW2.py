# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 17:43:14 2022

@author: seph
"""

### Backtesting Moving Average trading strategies with BTC ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp

BTC_df = pd.read_csv("BTC_USD_2014-11-03_2021-12-31-CoinDesk.csv",
                         parse_dates = True,
                         index_col = "Date").drop(['Currency', 
                                                   '24h Open (USD)', 
                                                   '24h High (USD)', 
                                                   '24h Low (USD)'], 1).rename(columns = {"Closing Price (USD)" : "BTC"})

BTC_df['daily_rets'] = BTC_df['BTC'].pct_change()
BTC_df['daily_logrets'] = np.log(BTC_df['daily_rets'] + 1)
BTC_df['cum_rets'] = np.cumprod(BTC_df['daily_rets'] + 1) - 1

#SMA 100
BTC_df['SMA_100'] = BTC_df['BTC'].rolling(100).mean()

#EMA 20                                           
BTC_df['modPrice'] = BTC_df['BTC']
BTC_df['SMA_20'] = BTC_df['BTC'].rolling(20).mean()
BTC_df['modPrice'][0:20] = BTC_df['SMA_20'][0:20]
BTC_df['EMA_20'] = BTC_df['modPrice'].ewm(span=20, adjust=False).mean()

#VMA 20
BTC_df['VMA_20'] = BTC_df['SMA_20']
BTC_df['CMO_20'] = abs(
    (BTC_df['daily_logrets'].gt(0)).rolling(window=20).sum() 
    - (BTC_df['daily_logrets'].le(0)).rolling(window=20).sum()) / 20
"""
CMO = |number of positive logrets days - number of negative logrets days for the past 20 days|, divided by 20 days
"""

for i in range(len(BTC_df['VMA_20'])):
    if i < 20:
        BTC_df['VMA_20'][i] = BTC_df['SMA_20'][i]
    else:
        BTC_df['VMA_20'][i] = 2/21 * BTC_df['CMO_20'][i] * BTC_df['BTC'][i] + (1-2/21 * BTC_df['CMO_20'][i]) * BTC_df['VMA_20'][i-1]
"""
For each row in the column VMA_20, 
if the row number is less than 20th position then set the value of the row equal to the value of the same row in SMA_20
else use the given formula to calculate the value of VMA_20 in the iterated row
"""        
        
#Trading Position 1: Long-Short EMA(20)-SMA(100)
BTC_df['Position1'] = np.where(BTC_df['EMA_20'] > BTC_df['SMA_100'],
                              1,
                              -1)
BTC_df['Position1'].mask(BTC_df['SMA_100'].isna(),np.nan,inplace=True) #<-- Replaces position values where SMA_100 is nan with nans

BTC_df['Position1_logrets'] = BTC_df['daily_logrets'] * BTC_df['Position1'].shift(1)
position1_cumrets = exp(np.sum(BTC_df['Position1_logrets']))
print(f'Trading Strategy 1: Long-Short EMA(20)-SMA(100)\n Cumulative Returns = {position1_cumrets}\n')

#Trading Position 2: Long Only EMA(20)-SMA(100)
BTC_df['Position2'] = np.where(BTC_df['EMA_20'] > BTC_df['SMA_100'],
                              1,
                              0)
BTC_df['Position2'].mask(BTC_df['SMA_100'].isna(),np.nan,inplace=True)
BTC_df['Position2_logrets'] = BTC_df['daily_logrets'] * BTC_df['Position2'].shift(1)
position2_cumrets = exp(np.sum(BTC_df['Position2_logrets']))
print(f'Trading Strategy 2: Long Only EMA(20)-SMA(100)\n Cumulative Returns = {position2_cumrets}\n')


#Trading Position 3: Long Only VMA(20)-SMA(100)
BTC_df['Position3'] = np.where(BTC_df['VMA_20'] > BTC_df['SMA_100'],
                              1,
                              0)
BTC_df['Position3'].mask(BTC_df['SMA_100'].isna(),np.nan,inplace=True)
BTC_df['Position3_logrets'] = BTC_df['daily_logrets'] * BTC_df['Position3'].shift(1)
position3_cumrets = exp(np.sum(BTC_df['Position3_logrets']))
print(f'Trading Strategy 3: Long Only VMA(20)-SMA(100)\n Cumulative Returns = {position3_cumrets}\n')

passive_cumrets = exp(np.sum(BTC_df['daily_logrets'])) #<-- It appears that only the VMA(20)-SMA(100) Long Only strategy beats the passive rets from holding
print(f'Passive Strategy: HODL\n Cumulative Returns = {passive_cumrets}\n')

#Position 1 Plots
BTC_df['2015-1-1':'2015-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position1'],
    secondary_y = 'Position1', figsize = (12,6)).set_title('Long-Short EMA(20)-SMA(100) 2015')
plt.savefig('Long-Short EMA(20)-SMA(100) 2015.png')

BTC_df['2016-1-1':'2016-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position1'],
    secondary_y = 'Position1', figsize = (12,6)).set_title('Long-Short EMA(20)-SMA(100) 2016')
plt.savefig('Long-Short EMA(20)-SMA(100) 2016.png')

BTC_df['2017-1-1':'2017-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position1'],
    secondary_y = 'Position1', figsize = (12,6)).set_title('Long-Short EMA(20)-SMA(100) 2017')
plt.savefig('Long-Short EMA(20)-SMA(100) 2017.png')

BTC_df['2018-1-1':'2018-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position1'],
    secondary_y = 'Position1', figsize = (12,6)).set_title('Long-Short EMA(20)-SMA(100) 2018')
plt.savefig('Long-Short EMA(20)-SMA(100) 2018.png')

BTC_df['2019-1-1':'2019-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position1'],
    secondary_y = 'Position1', figsize = (12,6)).set_title('Long-Short EMA(20)-SMA(100) 2019')
plt.savefig('Long-Short EMA(20)-SMA(100) 2019.png')

BTC_df['2020-1-1':'2020-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position1'],
    secondary_y = 'Position1', figsize = (12,6)).set_title('Long-Short EMA(20)-SMA(100) 2020')
plt.savefig('Long-Short EMA(20)-SMA(100) 2020.png')

BTC_df['2021-1-1':'2021-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position1'],
    secondary_y = 'Position1', figsize = (12,6)).set_title('Long-Short EMA(20)-SMA(100) 2021')
plt.savefig('Long-Short EMA(20)-SMA(100) 2021.png')

#Position 2 Plots
BTC_df['2015-1-1':'2015-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position2'],
    secondary_y = 'Position2', figsize = (12,6)).set_title('Long Only EMA(20)-SMA(100) 2015')
plt.savefig('Long Only EMA(20)-SMA(100) 2015.png')

BTC_df['2016-1-1':'2016-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position2'],
    secondary_y = 'Position2', figsize = (12,6)).set_title('Long Only EMA(20)-SMA(100) 2016')
plt.savefig('Long Only EMA(20)-SMA(100) 2016.png')

BTC_df['2017-1-1':'2017-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position2'],
    secondary_y = 'Position2', figsize = (12,6)).set_title('Long Only EMA(20)-SMA(100) 2017')
plt.savefig('Long Only EMA(20)-SMA(100) 2017.png')

BTC_df['2018-1-1':'2018-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position2'],
    secondary_y = 'Position2', figsize = (12,6)).set_title('Long Only EMA(20)-SMA(100) 2018')
plt.savefig('Long Only EMA(20)-SMA(100) 2018.png')

BTC_df['2019-1-1':'2019-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position2'],
    secondary_y = 'Position2', figsize = (12,6)).set_title('Long Only EMA(20)-SMA(100) 2019')
plt.savefig('Long Only EMA(20)-SMA(100) 2019.png')

BTC_df['2020-1-1':'2020-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position2'],
    secondary_y = 'Position2', figsize = (12,6)).set_title('Long Only EMA(20)-SMA(100) 2020')
plt.savefig('Long Only EMA(20)-SMA(100) 2020.png')

BTC_df['2021-1-1':'2021-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position2'],
    secondary_y = 'Position2', figsize = (12,6)).set_title('Long Only EMA(20)-SMA(100) 2021')
plt.savefig('Long Only EMA(20)-SMA(100) 2021.png')

#Position 3 Plots
BTC_df['2015-1-1':'2015-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position3'],
    secondary_y = 'Position3', figsize = (12,6)).set_title('Long Only VMA(20)-SMA(100) 2015')
plt.savefig('Long Only VMA(20)-SMA(100) 2015.png')

BTC_df['2016-1-1':'2016-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position3'],
    secondary_y = 'Position3', figsize = (12,6)).set_title('Long Only VMA(20)-SMA(100) 2016')
plt.savefig('Long Only VMA(20)-SMA(100) 2016.png')

BTC_df['2017-1-1':'2017-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position3'],
    secondary_y = 'Position3', figsize = (12,6)).set_title('Long Only VMA(20)-SMA(100) 2017')
plt.savefig('Long Only VMA(20)-SMA(100) 2017.png')

BTC_df['2018-1-1':'2018-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position3'],
    secondary_y = 'Position3', figsize = (12,6)).set_title('Long Only VMA(20)-SMA(100) 2018')
plt.savefig('Long Only VMA(20)-SMA(100) 2018.png')

BTC_df['2019-1-1':'2019-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position3'],
    secondary_y = 'Position3', figsize = (12,6)).set_title('Long Only VMA(20)-SMA(100) 2019')
plt.savefig('Long Only VMA(20)-SMA(100) 2019.png')

BTC_df['2020-1-1':'2020-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position3'],
    secondary_y = 'Position3', figsize = (12,6)).set_title('Long Only VMA(20)-SMA(100) 2020')
plt.savefig('Long Only VMA(20)-SMA(100) 2020.png')

BTC_df['2021-1-1':'2021-12-31'].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position3'],
    secondary_y = 'Position3', figsize = (12,6)).set_title('Long Only VMA(20)-SMA(100) 2021')
plt.savefig('Long Only VMA(20)-SMA(100) 2021.png')


        
