# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:56:47 2022

@author: seph
"""

### Analysis of investing characteristics of BTC and the performance of actively managed crypto hedge funds ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Part 1: Construct monthly return series for Bitcoin and S&P500 index

BTC_df = pd.read_csv("BTC_USD_2014-11-03_2021-12-31-CoinDesk.csv",
                         parse_dates = True,
                         index_col = "Date")
SP500_df = pd.read_csv("SP500_DailyIndex_2010-2021.csv",
                         parse_dates = True,
                         index_col = "Date")

BTCrets_df = BTC_df['Closing Price (USD)'].resample('M').apply(lambda x: x[-1]).pct_change().to_frame()
BTCrets_df = BTCrets_df.rename(columns = {'Closing Price (USD)' : 'BTC'})

SP500rets_df = SP500_df['Close'].resample('M').apply(lambda x: x[-1]).pct_change().to_frame()
SP500rets_df = SP500rets_df.rename(columns = {'Close' : 'SP500'})

FF3F_df = pd.read_csv("FF3Factors_Monthly.csv",
                      parse_dates = True,
                      index_col = "Date")
FF3F_df = FF3F_df / 100

helper_df = pd.merge(BTCrets_df, 
                     SP500rets_df,
                     on = "Date")
merged_df = pd.merge(helper_df,
                     FF3F_df,
                     on = "Date")

# Part 2: Statistics, plotting, and CAPM analysis

merged_df['BTC excess'] = merged_df['BTC'] - merged_df['RF']
merged_df['SP500 excess'] = merged_df['SP500'] - merged_df['RF']

BTC_MeanAnnual = merged_df['BTC excess'].mean() * 12
SP500_MeanAnnual = merged_df['SP500 excess'].mean() * 12

BTC_stdAnnual = merged_df['BTC excess'].std() * (12**(1/2))
SP500_stdAnnual = merged_df['SP500 excess'].std() * (12**(1/2))

BTC_sr = BTC_MeanAnnual / BTC_stdAnnual
SP500_sr = SP500_MeanAnnual / SP500_stdAnnual

corr_coeff = merged_df['BTC excess'].corr(merged_df['SP500 excess'])

BTC_skew = merged_df['BTC excess'].skew()
SP500_skew = merged_df['SP500 excess'].skew()

BTC_kurtosis = merged_df['BTC excess'].kurtosis()
SP500_kurtosis = merged_df['SP500 excess'].kurtosis()

BTC_VaR = merged_df['BTC excess'].quantile(0.05)
SP500_VaR = merged_df['SP500 excess'].quantile(0.05)

stats_df = pd.DataFrame(
    {'BTC' : [BTC_MeanAnnual, BTC_stdAnnual, BTC_sr, corr_coeff, BTC_skew, BTC_kurtosis, BTC_VaR],
     'SP500' : [SP500_MeanAnnual, SP500_stdAnnual, SP500_sr, corr_coeff, SP500_skew, SP500_kurtosis, SP500_VaR]},
     index = ['Annualized Mean Rets', 'Annualized Std', 'Sharpe Ratio', 'Correlation Coeffient', 'Skew', 'Kurtosis', 'VaR'])

print(f"Question 2.1 \n{stats_df}\n")

merged_df['BTC cumrets'] = (1 + merged_df['BTC excess']).cumprod() - 1
merged_df['SP500 cumrets'] = (1 + merged_df['SP500 excess']).cumprod() - 1

merged_df.plot(y = ['BTC cumrets', 'SP500 cumrets'],
               secondary_y=['SP500 cumrets']).set_title(
                   'Question 2.2: Bitcoin and S&P 500 Cumulative Returns Nov. 2014 - Nov. 2021')

plt.show()
plt.close()
                   
BTC_histogram = merged_df['BTC excess'].plot.hist().set_title(
    'Question 2.3: Empirical Distribution of Bitcoin Excess Returns')


merged_df.plot.scatter(x = "SP500 excess",
                        y = "BTC excess").set_title(
                            'Question 2.4: S&P 500 Monthly Excess Rets versus BTC Monthly Excess Rets')
                            
                            
BTC_SP500_fitted = smf.ols('Q("BTC excess") ~ Q("SP500 excess")', merged_df).fit()
print(f'Question 2.5:\n{BTC_SP500_fitted.summary()}\n')



BTC_FF3F_fitted = smf.ols('Q("BTC excess") ~ Q("Mkt-RF") + SMB + HML', merged_df).fit()         
print(f'Question 2.6:\n{BTC_FF3F_fitted.summary()}')                   

# Part 3: Performance of actively managed crypto hedge funds

HFR_df = pd.read_csv("HFR_CrpytoCurrency_IndexReturns.csv",
                         parse_dates = True,
                         index_col = "Date")

mergedHFR_df = pd.merge(merged_df,
                     HFR_df, 
                     on = "Date")

mergedHFR_df['ret_hfe'] = mergedHFR_df['ret_hf'] - mergedHFR_df['RF']

HFR_MeanAnnual = mergedHFR_df['ret_hfe'].mean() * 12
HFR_stdAnnual = mergedHFR_df['ret_hfe'].std() * (12**(1/2))
HFR_sr = HFR_MeanAnnual / HFR_stdAnnual
HFR_skew = mergedHFR_df['ret_hfe'].skew()
HFR_kurtosis = mergedHFR_df['ret_hfe'].kurtosis()
HFR_VaR = mergedHFR_df['ret_hfe'].quantile(0.05)

HFRstats_df = pd.DataFrame({'HFR' : [HFR_MeanAnnual, HFR_stdAnnual, HFR_sr, HFR_skew, HFR_kurtosis, HFR_VaR]},
                           index = ['Annualized Mean Rets', 'Annualized Std', 'Sharpe Ratio', 'Skew', 'Kurtosis', 'VaR'])

print(f"Question 3.3 \n{HFRstats_df}\n")
                   
HFR_BTC_fitted = smf.ols('ret_hfe ~ Q("BTC excess")', mergedHFR_df).fit()
info_ratio = HFR_BTC_fitted.params.Intercept / HFR_BTC_fitted.resid.std()

print(f'Question 3.4:\n{HFR_BTC_fitted.summary()}\n\nInformation Ratio = {info_ratio}\n')

mergedHFR_df['non-negative BTC excess'] = np.maximum(0, mergedHFR_df['BTC excess'])

HFR_BTC_multiple_fitted = smf.ols('ret_hfe ~ Q("BTC excess") + Q("non-negative BTC excess")', mergedHFR_df).fit()
print(f'Question 3.5:\n{HFR_BTC_multiple_fitted.summary()}')









