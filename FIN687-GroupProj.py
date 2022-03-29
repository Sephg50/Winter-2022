# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 17:09:09 2022

@author: seph
"""

### Analysis of Short Target NET using Technical Moving Averages, Momentum Strategy, Statistical Summaries, CAPM Regression, Drawdown Analysis ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import dataframe_image as dfi

NET_df = pd.read_csv("NET.csv",
                         parse_dates = True,
                         index_col = "Date").drop(['Open', 
                                                   'High', 
                                                   'Low', 
                                                   'Close',
                                                   'Volume'], 1).rename(columns = {"Adj Close" : "NET"})
                              
## 1. Technical Analysis ##                           
                                                   
NET_df['daily_rets'] = NET_df['NET'].pct_change()
NET_df['daily_logrets'] = np.log(NET_df['daily_rets'] + 1)
NET_df['cum_rets'] = np.cumprod(NET_df['daily_rets'] + 1) - 1

#SMA 100
NET_df['SMA_100'] = NET_df['NET'].rolling(100).mean()

#EMA 20                                           
NET_df['modPrice'] = NET_df['NET']
NET_df['SMA_20'] = NET_df['NET'].rolling(20).mean()
NET_df['modPrice'][0:20] = NET_df['SMA_20'][0:20]
NET_df['EMA_20'] = NET_df['modPrice'].ewm(span=20, adjust=False).mean()

#VMA 20
NET_df['VMA_20'] = NET_df['SMA_20']
NET_df['CMO_20'] = abs(
    (NET_df['daily_logrets'].gt(0)).rolling(window=20).sum() 
    - (NET_df['daily_logrets'].le(0)).rolling(window=20).sum()) / 20

for i in range(len(NET_df['VMA_20'])):
    if i < 20:
        NET_df['VMA_20'][i] = NET_df['SMA_20'][i]
    else:
        NET_df['VMA_20'][i] = 2/21 * NET_df['CMO_20'][i] * NET_df['NET'][i] + (1-2/21 * NET_df['CMO_20'][i]) * NET_df['VMA_20'][i-1]

#Trading Position 1: Long-Short EMA(20)-SMA(100)        
NET_df['Position1'] = np.where(NET_df['EMA_20'] > NET_df['SMA_100'],
                              1,
                              -1)
NET_df['Position1'].mask(NET_df['SMA_100'].isna(),np.nan,inplace=True)

#Trading Position 2: Long-Short VMA(20)-SMA(100)        
NET_df['Position2'] = np.where(NET_df['VMA_20'] > NET_df['SMA_100'],
                              1,
                              -1)
NET_df['Position2'].mask(NET_df['SMA_100'].isna(),np.nan,inplace=True)


NET_df.plot(y = ['SMA_100', 'EMA_20', 'VMA_20'], figsize = (24,12)).set_title('SMA100, EMA20, VMA20 IPO-Present')
plt.savefig('FIN687-GroupProj Moving Averages IPO-Present.png')

# NET_df.plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position1'],
#             secondary_y = 'Position1', figsize = (12,6)).set_title('NET EMA_20-SMA_100 Strategy IPO-Present')
# plt.savefig('FIN687-GroupProj EMA_20-SMA_100 Strategy IPO-Present.png')

# NET_df.plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position2'],
#             secondary_y = 'Position2', figsize = (12,6)).set_title('NET VMA_20-SMA_100 Strategy IPO-Present')
# plt.savefig('FIN687-GroupProj VMA_20-SMA_100 Strategy IPO-Present.png')
    

# NET_df["2021-9-03":"2022-03-04"].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position1'],
#                                       secondary_y = ['Position1'],
#                                       figsize = (12,6)).set_title('NET EMA_20-SMA_100 Strategy Last 6 Months')
# plt.savefig('FIN687-GroupProj EMA_20-SMA_100 Strategy Last 6 Months.png')

# NET_df["2021-9-03":"2022-03-04"].plot(y = ['SMA_100', 'EMA_20', 'VMA_20', 'Position2'],
#                                       secondary_y = ['Position2'],
#                                       figsize = (12,6)).set_title('NET VMA_20-SMA_100 Strategy Last 6 Months')
# plt.savefig('FIN687-GroupProj VMA_20-SMA_100 Strategy Last 6 Months.png')

## 2. Momentum Strategy ##

ff_df = pd.read_csv('F-F_Research_Data_Factors_daily.csv').rename(columns = {"Mkt-RF" : "RMRF"})

ff_df.rename(columns = {ff_df.columns[0]: "Date"},
                    inplace = True)
ff_df['Date'] = pd.to_datetime(ff_df['Date'], format='%Y%m%d')

ff_df = ff_df.set_index('Date')

ff_df = ff_df / 100

NET_df['RF'] = ff_df['RF'] 
NET_df['RF'] = NET_df['RF'].fillna(method='ffill')
NET_df['Excessrets'] = NET_df['daily_rets'] - NET_df['RF']

monthly_df = pd.DataFrame()
monthly_df['Cumrets'] = NET_df['Excessrets'].resample('M').agg(lambda x: (x + 1).prod() - 1)

lookback_period = 4

mom_df = pd.DataFrame()
mom_df[f'{lookback_period}m_cumrets'] = monthly_df['Cumrets'].rolling(window = lookback_period).agg(lambda x: (x + 1).prod() - 1)

delta = 60/61

NET_df = NET_df.iloc[1:] # <-- remove nan return from first day
NET_df['weight'] = 0   
NET_df['weighted_mean'] = 0
NET_df['var'] = 0
NET_df['weighted_var'] = 0
NET_df['Exante'] = 0

for i in range(0,len(NET_df['Excessrets']),1):
    NET_df['weight'].iloc[i:] = (1 - delta) * (delta ** i)
    NET_df['weighted_mean'].iloc[i:] = np.dot(NET_df['weight'][i::-1],
                                           NET_df['Excessrets'][0:i+1])
    NET_df['var'].iloc[i:] = (NET_df['Excessrets'][i] - NET_df['weighted_mean'][i])**2
    NET_df['weighted_var'].iloc[i:] = np.dot(NET_df['weight'][i::-1],
                                              NET_df['var'][0:i+1])
    NET_df['Exante'].iloc[i:] = (365 * NET_df['weighted_var'][i]) ** 0.5
    
mom_df['Exante'] = NET_df['Exante'].resample('M').apply(lambda x: x[-1])

mom_df['MOM_Position'] = np.where(mom_df[f'{lookback_period}m_cumrets'] > 0,
                                                    1,
                                                    -1)
mom_df['MOM_Position'].mask(mom_df[f'{lookback_period}m_cumrets'].isna(),np.nan,inplace=True)

mom_df.plot(y = f'{lookback_period}m_cumrets', figsize = (24,12)).set_title('Quarterly Momentum IPO-Present')
plt.savefig('FIN687-GroupProj Quarterly Momentum IPO-Present.png')

## 3. Statistics ##

NET_MeanAnnual = NET_df['Excessrets'].mean() * 12
NET_stdAnnual = NET_df['Excessrets'].std() * (12**(1/2))
NET_sr = NET_MeanAnnual / NET_stdAnnual
NET_skew = NET_df['Excessrets'].skew()
NET_kurtosis = NET_df['Excessrets'].kurtosis()
NET_VaR = NET_df['Excessrets'].quantile(0.05)


NETstats_df = pd.DataFrame({'NET' : [NET_MeanAnnual, NET_stdAnnual, NET_sr, NET_skew, NET_kurtosis, NET_VaR]},
                           index = ['Annualized Mean Rets', 'Annualized Std', 'Sharpe Ratio', 'Skew', 'Kurtosis', 'VaR'])
dfi.export(NETstats_df, 'FIN687-GroupProj NET Statistics.png')

## 4. Regression ##

NET_df['RMRF'] = ff_df['RMRF']

CAPM_fitted = smf.ols('Excessrets ~ RMRF', NET_df).fit()
ir = CAPM_fitted.params.Intercept / CAPM_fitted.resid.std()
print(CAPM_fitted.summary())

## 5. Drawdown ##

NET_df['Cum_NET'] = (1 + NET_df['Excessrets']).cumprod()
NET_df['HWM'] = 0
NET_df['DD'] = 0
NET_df['MDD'] = 0

for i in range(0,len(NET_df['Cum_NET']),1):
    NET_df['HWM'].iloc[i:] = max(NET_df['Cum_NET'][0:i+1])
    NET_df['DD'].iloc[i:] = (NET_df['Cum_NET'][i] - NET_df['HWM'][i]) / NET_df['HWM'][i]
    NET_df['MDD'].iloc[i:] = min(NET_df['DD'][0:i+1])
    
NET_df.plot(y = ['DD', 'HWM', 'MDD'],
              secondary_y = 'HWM',
              figsize = (24,12)).set_title('DD, HWM, MDD IPO-Present')
plt.savefig('FIN687-GroupProj Monthly DD, HWM, and MDD from IPO-Present.png')
mdd = NET_df["MDD"][-1]
