# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 18:27:01 2022

@author: seph
"""

### Analysis of Factor Investing and Optimization with 4 Factor Model ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.optimize as optimize
import dataframe_image as dfi

## Question 2 ##

df = pd.read_csv('FactorReturns_Data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m')
df = df.set_index('Date')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.iloc[1:] # <-- remove first row (redundant header) --> DID NOT WORK, INSTEAD MANUALLY REMOVED FROM CSV FILE
df = df / 100

factor_df = pd.DataFrame()

factor_df['RMRF'] = df['Market'] - df['Risk Free']
factor_df['SMB'] = df['Small'] - df['Big']
factor_df['HML'] = df['High B/M'] - df['Low B/M']
factor_df['MOM'] = df['Winner'] - df['Loser']

## Question 2.1 ##

regression_df = pd.DataFrame()
regression_df['RMRF'] = factor_df['RMRF']
regression_df['Short_MOM'] = -df['Loser']
regression_df['Short_Big'] = -df['Big']
regression_df['Short_Low'] = -df['Low B/M'] 

ShortMOM_fitted = smf.ols('Short_MOM ~ RMRF', regression_df).fit()
ShortMOM_alpha = ShortMOM_fitted.params.Intercept * 12
ShortMOM_pvalue = ShortMOM_fitted.pvalues.Intercept

ShortBig_fitted = smf.ols('Short_Big ~ RMRF', regression_df).fit()
ShortBig_alpha = ShortBig_fitted.params.Intercept * 12
ShortBig_pvalue = ShortBig_fitted.pvalues.Intercept

ShortLow_fitted = smf.ols('Short_Low ~ RMRF', regression_df).fit()
ShortLow_alpha = ShortLow_fitted.params.Intercept * 12
ShortLow_pvalue = ShortLow_fitted.pvalues.Intercept

Short_alpha_df = pd.DataFrame({'Annualized Alpha' : [ShortBig_alpha, ShortLow_alpha, ShortMOM_alpha],
                               'P-value' : [ShortBig_pvalue, ShortLow_pvalue, ShortMOM_pvalue]},
                              index = ['Short Big', 'Short Low','Short MOM'])
dfi.export(Short_alpha_df, 'FIN687-Final Short Alpha.png')

SMB_fitted = smf.ols('SMB ~ RMRF', factor_df).fit()
SMB_alpha = SMB_fitted.params.Intercept * 12
SMB_pvalue = SMB_fitted.pvalues.Intercept
SMB_beta = SMB_fitted.params.RMRF

HML_fitted = smf.ols('HML ~ RMRF', factor_df).fit()
HML_alpha = SMB_fitted.params.Intercept * 12
HML_pvalue = HML_fitted.pvalues.Intercept
HML_beta = HML_fitted.params.RMRF

MOM_fitted = smf.ols('MOM ~ RMRF', factor_df).fit()
MOM_alpha = MOM_fitted.params.Intercept * 12
MOM_pvalue = MOM_fitted.pvalues.Intercept
MOM_beta = MOM_fitted.params.RMRF

factor_AB_df = pd.DataFrame({'Annualized Alpha' : [SMB_alpha, HML_alpha, MOM_alpha],
                             'P-value' : [SMB_pvalue, HML_pvalue, MOM_pvalue],
                             'Beta' : [SMB_beta, HML_beta, MOM_beta]},
                              index = ['SMB', 'HML','MOM'])
dfi.export(factor_AB_df, 'FIN687-Final Factor AlphaBeta.png')

corr_df = factor_df.corr()
dfi.export(corr_df, 'FIN687-Corr.png')

## Question 2.2 ##

monthly_df = pd.DataFrame()

for i in range(1,13,1):
    monthly_df[f'{i}'] = factor_df[factor_df.index.month == i].mean()
    
monthly_df = monthly_df.swapaxes("index","columns")

monthly_df.plot(figsize = (18,6)).set_title('Calendar Month Factor Returns from 1927-2009')
plt.savefig('FIN687-Final Calendar Month Factor Returns from 1927-2009.png')

RMRF_max = max(monthly_df['RMRF'])
RMRF_min = min(monthly_df['RMRF'])

SMB_max = max(monthly_df['SMB'])
SMB_min = min(monthly_df['SMB'])

HML_max = max(monthly_df['HML'])
HML_min = min(monthly_df['HML'])

MOM_max = max(monthly_df['MOM'])
MOM_min = min(monthly_df['MOM'])

maxmin_df = pd.DataFrame({'Max' : [RMRF_max, SMB_max, HML_max, MOM_max],
                          'Min' : [RMRF_min, SMB_min, HML_min, MOM_min]},
                         index = ['RMRF', 'SMB', 'HML', 'MOM'])
maxmin_df.plot.bar(figsize = (18,6)).set_title('Calendar Month Factor Maximum and Minimum Returns from 1927-2009')
plt.savefig('FIN687-Final Calendar Month Factor Maximum and Minimum Returns from 1927-2009.png')


## Question 3.1 and 3.2 ##

cum_df = (1 + factor_df).cumprod() - 1

for i in range(len(cum_df.columns)):
    cum_df.plot(y = f'{cum_df.columns[i]}',
                figsize = (18,6)).set_title(f'{cum_df.columns[i]} Cumulative Monthly Returns from 1927-2009')
    plt.savefig(f'FIN687-Final {cum_df.columns[i]} Cumulative Monthly Returns from 1927-2009.png')

cum_df.plot(figsize = (18,6)).set_title('All Factors Cumulative Monthly Returns from 1927-2009')
plt.savefig('FIN687-Final Cumulative Monthly Returns from 1927-2009.png')

factor_recent_df = factor_df['1999-1-01':]

SMB_recent_fitted = smf.ols('SMB ~ RMRF', factor_recent_df).fit()
SMB_recent_alpha = SMB_recent_fitted.params.Intercept * 12
SMB_recent_pvalue = SMB_recent_fitted.pvalues.Intercept
SMB_recent_beta = SMB_recent_fitted.params.RMRF

HML_recent_fitted = smf.ols('HML ~ RMRF', factor_recent_df).fit()
HML_recent_alpha = SMB_recent_fitted.params.Intercept * 12
HML_recent_pvalue = HML_recent_fitted.pvalues.Intercept
HML_recent_beta = HML_recent_fitted.params.RMRF

MOM_recent_fitted = smf.ols('MOM ~ RMRF', factor_recent_df).fit()
MOM_recent_alpha = MOM_recent_fitted.params.Intercept * 12
MOM_recent_pvalue = MOM_recent_fitted.pvalues.Intercept
MOM_recent_beta = MOM_recent_fitted.params.RMRF

factor_recent_AB_df = pd.DataFrame({'Annualized Alpha' : [SMB_recent_alpha, HML_recent_alpha, MOM_recent_alpha],
                             'P-value' : [SMB_recent_pvalue, HML_recent_pvalue, MOM_recent_pvalue],
                             'Beta' : [SMB_recent_beta, HML_recent_beta, MOM_recent_beta]},
                              index = ['SMB_recent', 'HML_recent','MOM_recent'])
dfi.export(factor_recent_AB_df, 'FIN687-Final Factor Recent AlphaBeta.png')

corr_recent_df = factor_recent_df.corr()
dfi.export(corr_df, 'FIN687-Final Corr Recent.png')


## Question 3.3 ##

mean_xs = []
stdev_xs = []
sr_xs = []
var_xs = []
skew_xs = []
kurt_xs = []

## Question 3.4: Optimized 4 Factor Model ##

rf = 0.02 #<-- Assumed risk free rate of 0.02
Mkt_premium = factor_df['RMRF'].mean() * 12
Mkt_std = factor_df['RMRF'].std() * (12 ** (1/2))

SMB_fs_sd = SMB_fitted.resid.std() * (12**(1/2))
HML_fs_sd = HML_fitted.resid.std() * (12**(1/2))
MOM_fs_sd = MOM_fitted.resid.std() * (12**(1/2))

SMB_rets_CAPM = rf + SMB_beta * Mkt_premium
HML_rets_CAPM = rf + HML_beta * Mkt_premium
MOM_rets_CAPM = rf + MOM_beta * Mkt_premium

def SingleIndex(params):
    SMB_weight, HML_weight, MOM_weight = params
    Market_w = (1 - (SMB_weight + HML_weight + MOM_weight))
    newfactor_alpha = SMB_weight*SMB_alpha + HML_weight*HML_alpha + MOM_weight*MOM_alpha #<-- Mkt alpha = 0
    newfactor_beta = SMB_weight*SMB_beta + HML_weight*HML_beta + MOM_weight*MOM_beta + Market_w*1 #<-- Mkt beta = 1
    newfactor_rets = newfactor_alpha + newfactor_beta * Mkt_premium
    newfactor_var = ((newfactor_beta ** 2)*(Mkt_std**2)
                +(SMB_weight**2)*(SMB_fs_sd**2)
                +(HML_weight**2)*(HML_fs_sd**2)
                +(MOM_weight**2)*(MOM_fs_sd**2)
                +(Market_w**2)*(0)) #<-- Market residual sd = 0
    return -(newfactor_rets / (newfactor_var**(1/2)))
                        
initial_guess = [1, 1, 1]
result = optimize.minimize(SingleIndex, initial_guess)
if result.success:
    fitted_params = result.x
    fitted_params = np.append(fitted_params,[(1-fitted_params.sum())])
    ORP_SingleIndex_df = pd.DataFrame(fitted_params, 
                          index = ['SMB', 'HML', 'MOM', 'RMRF'],
                          columns = ['Weights'])
    dfi.export(ORP_SingleIndex_df, 'FIN687-Final ORP.png')
    
else:
    raise ValueError(result.message)

factor_df['4FactorModel'] = ((ORP_SingleIndex_df['Weights'][0] * factor_df['SMB']) 
                              + (ORP_SingleIndex_df['Weights'][1] * factor_df['HML'])
                              + (ORP_SingleIndex_df['Weights'][2] * factor_df['MOM'])
                              + (ORP_SingleIndex_df['Weights'][3] * factor_df['RMRF'])) 
                              
cum_df['4FactorModel'] = (factor_df['4FactorModel'] + 1).cumprod() - 1

for i in range(len(factor_df.columns)):
    mean_xs.append(factor_df[cum_df.columns[i]].mean() * 12)
    stdev_xs.append(factor_df[cum_df.columns[i]].std() * (12 ** (1/2)))
    sr_xs.append(mean_xs[i] / stdev_xs[i])
    var_xs.append((factor_df[cum_df.columns[i]].quantile(0.05)))
    skew_xs.append((factor_df[cum_df.columns[i]].skew()))
    kurt_xs.append((factor_df[cum_df.columns[i]].kurtosis()))

stats_df = pd.DataFrame({'Annualized Mean Rets' : mean_xs,
                         'Annualized Stdev' : stdev_xs,
                         'Sharpe Ratio' : sr_xs,
                         'Skew' : skew_xs,
                         'Kurtosis' : kurt_xs,
                         '5% VaR' : var_xs},
                        index = ['RMRF', 'SMB', 'HML', 'MOM', '4FactorModel'])
stats_df = stats_df.swapaxes("index","columns")

dfi.export(stats_df, 'FIN687-Final Stats.png')

cum_df.plot(y = '4FactorModel',
            figsize = (18,6)).set_title('4 Factor Model Cumulative Monthly Returns from 1927-2009')
plt.savefig('FIN687-Final 4FactorModel Cumulative Monthly Returns from 1927-2009.png')

cum_df.plot(figsize = (18,6)).set_title('4 Factor Model Versus Other Factors Cumulative Monthly Returns from 1927-2009')
plt.savefig('FIN687-Final 4FactorModel vs all Cumulative Monthly Returns from 1927-2009.png')
