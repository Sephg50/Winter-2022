# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 09:15:10 2022

@author: seph
"""

### Backtesting BTC Time Series Momentum Strategy using Daily BTC Returns from 2014 to 2021 ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import dataframe_image as dfi

## Step 1: Determine the Trading Direction and Position Size ##

# 1. Read in the bitcoin daily prices and compute daily returns

BTC_df = pd.read_csv('BTC_USD_2014-11-03_2021-12-31-CoinDesk-2.csv',
                     parse_dates = True,
                     index_col = 'Date').drop(['Currency', 
                                                   '24h Open (USD)', 
                                                   '24h High (USD)', 
                                                   '24h Low (USD)'], 1).rename(columns = {"Closing Price (USD)" : "BTC"})
                                               
BTC_df['Rets'] = BTC_df['BTC'].pct_change()

# 2. Merge bitcoin daily returns with risk-free rates and compute daily excess returns

rf_df = pd.read_csv('F-F_Research_Data_Factors_daily.csv').drop(['SMB',
                                                                 'HML',
                                                                 'Mkt-RF'], 1)
rf_df['Date'] = pd.to_datetime(rf_df['Date'], format='%Y%m%d')
rf_df = rf_df.set_index('Date')

BTC_df['RF'] = rf_df['RF'] / 100
BTC_df['RF'] = BTC_df['RF'].fillna(method='ffill')

BTC_df['Excessrets'] = BTC_df['Rets'] - BTC_df['RF']

# 3. Compute the cumulative return for each month based on daily excess returns

monthly_df = pd.DataFrame()
monthly_df['Cumrets'] = BTC_df['Excessrets'].resample('M').agg(lambda x: (x + 1).prod() - 1)

# 4. Compute the rolling 12-month cumulative returns

rollingmonthly_df = pd.DataFrame()
rollingmonthly_df['12m_cumrets'] = monthly_df['Cumrets'].rolling(window = 12).agg(lambda x: (x + 1).prod() - 1) # <-- calc w MONTHLY values

# 5. Compute the ex-ante volatility measured based on daily excess returns

delta = 60/61

BTC_df = BTC_df.iloc[1:] # <-- remove nan return from first day
BTC_df['weight'] = 0   
BTC_df['weighted_mean'] = 0
BTC_df['var'] = 0
BTC_df['weighted_var'] = 0
BTC_df['Exante'] = 0

for i in range(0,len(BTC_df['Excessrets']),1):
    BTC_df['weight'].iloc[i:] = (1 - delta) * (delta ** i)
    BTC_df['weighted_mean'].iloc[i:] = np.dot(BTC_df['weight'][i::-1],
                                           BTC_df['Excessrets'][0:i+1])
    BTC_df['var'].iloc[i:] = (BTC_df['Excessrets'][i] - BTC_df['weighted_mean'][i])**2
    BTC_df['weighted_var'].iloc[i:] = np.dot(BTC_df['weight'][i::-1],
                                              BTC_df['var'][0:i+1])
    BTC_df['Exante'].iloc[i:] = (365 * BTC_df['weighted_var'][i]) ** 0.5

# 6. Create a new dataframe containing the ex-ante volatility measures for the last day of each month

Exante_df = pd.DataFrame()
Exante_df['Exante'] = BTC_df['Exante'].resample('M').apply(lambda x: x[-1])

# 7. Merge the data frame in (6) into the data frame containing the rolling 12-month cumulative returns in (4)

monthly_df['Exante'] = Exante_df['Exante']

# 8. Determine the trigger and direction of the trade: At each month-end, take a long (short) position
#    if the previous 12-month cumulative return is positive (negative). If the previous 12-month cumulative
#    return is zero, then take no position.

monthly_df['Position'] = np.where(rollingmonthly_df['12m_cumrets'] > 0,
                                                    1,
                                                    -1)
# 9. Determine the size of the position: size the position to make the annualized ex-ante volatility 40%

monthly_df['Size'] = 0.4 / monthly_df['Exante']


## Step 2: Construct Monthly Return series and Perform the Risk-Return Analysis ##

# 1. Calculate the time-series monthly return (TSRet) series based on direction of trade and size of position

TSRet_df = pd.DataFrame()

monthly_df['TSRet'] = monthly_df['Position'].shift(1) * monthly_df['Size'].shift(1) * (monthly_df['Cumrets'])
TSRet_df['TSRet'] = monthly_df['Position'].shift(1) * monthly_df['Size'].shift(1) * (monthly_df['Cumrets']) ## monthly BTC excess returns

#Shift 1 for position and size

TSRet_df = TSRet_df["2016-01-31":"2021-12-31"] # <-- sliced here

# 2. Calculate annualized mean, standard deviation, and Sharpe ratio based on the monthly TSRet series between Jan 2016 and Dec 2021. 

TSRet_mean = TSRet_df['TSRet'].mean() * 12
TSRet_std = TSRet_df['TSRet'].std() * (12 ** (1/2))
TSRet_sr = TSRet_mean / TSRet_std

# 3. Read in the monthly factor returns (RMRF, SMB, HML, and MOM) and merge into the 
#    data frame containing the monthly TSRet series constructed in (1) 

ff_df = pd.read_csv("F-F_Research_Data_Factors.CSV",
                    index_col = 'Date',
                    parse_dates = True).rename(columns = {"Mkt-RF" : "RMRF"})

ff_df = ff_df / 100

TSRet_df = pd.merge(TSRet_df,
                 ff_df,
                 on = "Date")

# 4. Do a scatter plot between TSRet and RMRF 

TSRet_df.plot.scatter(x = "TSRet",
                        y = "RMRF", figsize = (18,9)).set_title("TSRet versus RMRF from 2016 to 2021")
plt.savefig('TSRet versus RMRF from 2016 to 2021.png')

# 5. Regress TSRet on RMRF: alpha, beta, and t-statistics

TSRetRMRF_fitted = smf.ols('TSRet ~ RMRF', TSRet_df).fit()
print(TSRetRMRF_fitted.summary())


"""
Checking performance against BTC hodl:
"""

# TSRet_df['BTCRet'] = monthly_df['Cumrets']
# BTCRetRMRF_fitted = smf.ols('BTCRet ~ RMRF', TSRet_df).fit()
# ir_test = BTCRetRMRF_fitted.params.Intercept / BTCRetRMRF_fitted.resid.std()
# print(BTCRetRMRF_fitted.summary())

# 6. Regress TSRet on RMRF, SMB, HML, and MOM: alpha, betas, and t-statistics 
#    Based on the monthly returns (TSRet), compute the annual return series from 2016 to 2021
#    and plot a bar chart (with Year on the X-axis and Annual Return on the Y-axis).

TSRetMultiple_fitted = smf.ols('TSRet ~ RMRF + SMB + HML + MOM', TSRet_df).fit()
print(TSRetMultiple_fitted.summary())

TSRetAnnual_df = pd.DataFrame()
BTCAnnual_df = pd.DataFrame()

TSRetAnnual_df['TSRet'] = TSRet_df['TSRet'].resample('Y').apply(lambda x: (x + 1).prod() - 1) # <-- Annualized TSRet
BTCAnnual_df['BTC'] = BTC_df['Excessrets'].resample('Y').apply(lambda x: (x + 1).prod() - 1) 
BTCAnnual_df = BTCAnnual_df["2016-01-31":"2021-12-31"]

TSRetAnnual_df.plot.bar(figsize = (12,6)
    ).set_title('Annual TSRet from 2016 to 2021')
plt.savefig('TSRet Annual from 2016 to 2021.png',bbox_inches="tight")

# 7. Based on the monthly return series (TSRet), compute the information ratio (IR)based on the CAPM regression: 

ir = TSRetRMRF_fitted.params.Intercept / TSRetRMRF_fitted.resid.std()

# 8. Based on the monthly return series (TSRet), plot the empirical density function by using histogram. 

plt.show()
plt.close()
TSRet_df['TSRet'].plot.hist(figsize = (12,6)).set_title('Empirical Distribution of Monthly TSRet from 2016 to 2021')
plt.savefig('Monthly TSRet Empirical Distribution from 2016 to 2021.png')

# 9. Based on the monthly return series (TSRet), compute the 5% VaR, the skew, and the kurtosis measures.

TSRet_VaR = TSRet_df['TSRet'].quantile(0.05)
TSRet_skew = TSRet_df['TSRet'].skew()
TSRet_kurtosis = TSRet_df['TSRet'].kurtosis()

# 10. Compute Drawdown: 

"""
Drawdown (DD) is an important risk measure for a hedge fund strategy. It measures the 
amount that has been lost since the peak (i.e., the high-water mark (HWM)). The 
percentage DD since the HWM is given by:  
 DDt = (HWMt – CumRett) / HWMt 
The maximum drawdown (MDD) over some past time period is given by: 
 MDDT = maxt<=T DDt 
Use the monthly return series (TSRet) from 2016 to 2021, compute and plot the monthly 
DDs relative to the HWM and the MDD over the period. 
"""

stats_df = pd.DataFrame({'TSMOM' : [TSRet_mean, TSRet_std, TSRet_sr, TSRet_skew, TSRet_kurtosis, TSRet_VaR]},
                           index = ['Annualized Mean Rets', 'Annualized Std', 'Sharpe Ratio', 'Skew', 'Kurtosis', '5% VaR'])
dfi.export(stats_df, 'TS Momentum Statistics from 2016 to 2019.png')

TSRet_df['Cum_TSRet'] = (1 + TSRet_df['TSRet']).cumprod()
TSRet_df['HWM'] = 0
TSRet_df['DD'] = 0
TSRet_df['MDD'] = 0

for i in range(0,len(TSRet_df['Cum_TSRet']),1):
    TSRet_df['HWM'].iloc[i:] = max(TSRet_df['Cum_TSRet'][0:i+1])
    TSRet_df['DD'].iloc[i:] = (TSRet_df['Cum_TSRet'][i] - TSRet_df['HWM'][i]) / TSRet_df['HWM'][i]
    TSRet_df['MDD'].iloc[i:] = min(TSRet_df['DD'][0:i+1])
    
TSRet_df.plot(y = ['DD', 'HWM', 'MDD'],
              secondary_y = 'HWM',
              figsize = (12,6)).set_title('Monthly DD, HWM, and MDD from 2016 to 2021')
plt.savefig('Monthly DD, HWM, and MDD from 2016 to 2021.png')

print(f'Information Ratio = {ir}')

print(f'MDD = {TSRet_df["MDD"][-1]}')
