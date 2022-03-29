# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:53:20 2022

@author: seph
"""

### Backtesting Risk Parity Strategy with BTC ###

import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi


# 3. Create two benchmark portfolios used for performance comparison
stocks_df = pd.read_csv('CRSP_StockIndex_Monthly.csv',
                          parse_dates = True,
                          index_col = "Date").rename(columns = {"STRet" : "Stocks"})

stocks_df.index = pd.to_datetime(stocks_df.index, format = '%Y-%m-%d').strftime('%Y-%m')

treasury_df = pd.read_csv('CRSP_TreasuriesIndex_Monthly.csv',
                          parse_dates = True,
                          index_col = "Date").rename(columns = {"TRRet" : "Bonds"})
treasury_df.index = pd.to_datetime(treasury_df.index, format = '%Y-%m-%d').strftime('%Y-%m')

rf_df = pd.read_csv('FF3Factors_Monthly.csv',
                          parse_dates = True,
                          index_col = "Date").drop(['SMB',
                                                    'HML',
                                                    'Mkt-RF'], 1)

rf_df = rf_df / 100
rf_df.index = pd.to_datetime(rf_df.index, format = '%Y-%m-%d').strftime('%Y-%m')

helper_df = pd.merge(stocks_df,
                      treasury_df,
                      on = "Date")

benchmark_df = pd.merge(helper_df,
                      rf_df,
                      on = "Date")

benchmark_df = benchmark_df.loc["1927-01":"2019-12"]

benchmark_df['Stocks'] = benchmark_df['Stocks'] - benchmark_df['RF']
benchmark_df['Bonds'] = benchmark_df['Bonds'] - benchmark_df['RF']

benchmark_df['60/40'] = 0.6 * benchmark_df['Stocks'] + 0.4 * benchmark_df['Bonds'] 

benchmark_df['Stocks Rolling Std'] = benchmark_df['Stocks'].rolling(36).std() * (12**(1/2))
benchmark_df['Stocks Rolling Std'] = benchmark_df['Stocks Rolling Std'].shift(1)

benchmark_df['Bonds Rolling Std'] = benchmark_df['Bonds'].rolling(36).std() * (12**(1/2))
benchmark_df['Bonds Rolling Std'] = benchmark_df['Bonds Rolling Std'].shift(1)

# 4. Construct three Risk Parity portfolios

unlevered_df = pd.DataFrame(index = benchmark_df.index)
unlevered_df['Kt'] = 1 / ((1 / benchmark_df['Stocks Rolling Std']) + (1 / benchmark_df['Bonds Rolling Std']))
unlevered_df['% Stock (RP)'] = (1 / benchmark_df['Stocks Rolling Std']) * unlevered_df['Kt']
unlevered_df['% Bonds (RP)'] = (1 / benchmark_df['Bonds Rolling Std']) * unlevered_df['Kt']
unlevered_df['RP Excess Rets'] = ((benchmark_df['Stocks'] * unlevered_df['% Stock (RP)'])
                                  + (benchmark_df['Bonds'] * unlevered_df['% Bonds (RP)']))


leverage1_df = pd.DataFrame(index = benchmark_df.index)
k1 = benchmark_df['Stocks']["1930-01":"2019-12"].std() * (12**(1/2))
leverage1_df['% Stock (RP)'] = (1 / benchmark_df['Stocks Rolling Std']) * k1
leverage1_df['% Bonds (RP)'] = (1 / benchmark_df['Bonds Rolling Std']) * k1
leverage1_df['RP Excess Rets'] = ((benchmark_df['Stocks'] * leverage1_df['% Stock (RP)'])
                                  + (benchmark_df['Bonds'] * leverage1_df['% Bonds (RP)']))

leverage2_df = pd.DataFrame(index = benchmark_df.index)
k2 = benchmark_df['60/40']["1930-01":"2019-12"].std() * (12**(1/2))
leverage2_df['% Stock (RP)'] = (1 / benchmark_df['Stocks Rolling Std']) * k2
leverage2_df['% Bonds (RP)'] = (1 / benchmark_df['Bonds Rolling Std']) * k2
leverage2_df['RP Excess Rets'] = ((benchmark_df['Stocks'] * leverage2_df['% Stock (RP)'])
                                  + (benchmark_df['Bonds'] * leverage2_df['% Bonds (RP)']))

# 5. Report statistics

stock_mean = benchmark_df['Stocks']["1930-01":"2019-12"].mean() * 12
stock_std = benchmark_df['Stocks']["1930-01":"2019-12"].std() * (12 ** (1/2))
stock_sr = stock_mean / stock_std

p6040_mean = benchmark_df['60/40']["1930-01":"2019-12"].mean() * 12
p6040_std = benchmark_df['60/40']["1930-01":"2019-12"].std() * (12 ** (1/2))
p6040_sr = p6040_mean / p6040_std

unlevered_mean = unlevered_df['RP Excess Rets']["1930-01":"2019-12"].mean() * 12
unlevered_std = unlevered_df['RP Excess Rets']["1930-01":"2019-12"].std() * (12 ** (1/2))
unlevered_sr = unlevered_mean / unlevered_std

leverage1_mean = leverage1_df['RP Excess Rets']["1930-01":"2019-12"].mean() * 12
leverage1_std = leverage1_df['RP Excess Rets']["1930-01":"2019-12"].std() * (12 ** (1/2))
leverage1_sr = leverage1_mean / leverage1_std

leverage2_mean = leverage2_df['RP Excess Rets']["1930-01":"2019-12"].mean() * 12
leverage2_std = leverage2_df['RP Excess Rets']["1930-01":"2019-12"].std() * (12 ** (1/2))
leverage2_sr = leverage2_mean / leverage2_std

unlevered_allocateS = unlevered_df['% Stock (RP)']["1930-01":"2019-12"].mean()
unlevered_allocateB = unlevered_df['% Bonds (RP)']["1930-01":"2019-12"].mean()

leverage1_allocateS = leverage1_df['% Stock (RP)']["1930-01":"2019-12"].mean()
leverage1_allocateB = leverage1_df['% Bonds (RP)']["1930-01":"2019-12"].mean()

leverage2_allocateS = leverage2_df['% Stock (RP)']["1930-01":"2019-12"].mean()
leverage2_allocateB = leverage2_df['% Bonds (RP)']["1930-01":"2019-12"].mean()

stats_df = pd.DataFrame(
    {'Annualized Mean Rets' : [stock_mean, p6040_mean, unlevered_mean, leverage1_mean, leverage2_mean],
     'Annualized Std' : [stock_std, p6040_std, unlevered_std, leverage1_std, leverage2_std],
     'Sharpe Ratio' : [stock_sr, p6040_sr, unlevered_sr, leverage1_sr, leverage2_sr],
     'Average Stock Allocation' : [1, 0.6, unlevered_allocateS, leverage1_allocateS, leverage2_allocateS],
     'Average Bond Allocation' : [0, 0.4, unlevered_allocateB, leverage1_allocateB, leverage2_allocateB]},
    index = ['Stock Market Index', '60/40 Portfolio', 'Unlevered RP', 'Levered RP 1', 'Levered RP 2'])

dfi.export(stats_df, 'Statistics 1930-2019.png')

stats_df['Sharpe Ratio'].plot.bar(figsize = (12,6)
    ).set_title('Sharpe Ratio of Returns from Jan 1930 to Dec 2019')

allocated = plt.text(
    -0.8, -0.45, 
    'Average Allocations of RP Portfolios\n'
    f' Unlevered: Stocks = {unlevered_allocateS:{.4}}, Bonds = {unlevered_allocateB:{.4}}\n'
    f' Levered 1 RP: Stocks = {leverage1_allocateS:{.4}}, Bonds = {leverage1_allocateB:{.4}}\n'
    f' Levered 2 RP: Stocks = {leverage2_allocateS:{.4}}, Bonds = {leverage2_allocateB:{.4}}')
plt.savefig('Sharpe Ratios 1930-2019.png', bbox_inches = "tight")
plt.close()

# 6. Loops

"""
Transformed the same process above into for loops, iterating through each 10-year subperiod:
"""

stock_mean_xs = []
stock_std_xs = []
stock_sr_xs = []

p6040_mean_xs = []
p6040_std_xs = []
p6040_sr_xs = []

unlevered_mean_xs = []
unlevered_std_xs = []
unlevered_sr_xs = []

leverage1_mean_xs = []
leverage1_std_xs = []
leverage1_sr_xs = []

leverage2_mean_xs = []
leverage2_std_xs = []
leverage2_sr_xs = []

unlevered_allocateS_xs = []
unlevered_allocateB_xs = []

leverage1_allocateS_xs = []
leverage1_allocateB_xs = []

leverage2_allocateS_xs = []
leverage2_allocateB_xs = []

for i in range(36,len(benchmark_df),120):
    stock_mean_xs.append((benchmark_df['Stocks'][i:i+120]).mean() * 12)
    stock_std_xs.append((benchmark_df['Stocks'][i:i+120]).std() * (12 ** (1/2)))
    
    p6040_mean_xs.append((benchmark_df['60/40'][i:i+120]).mean() * 12)
    p6040_std_xs.append((benchmark_df['60/40'][i:i+120]).std() * (12 ** (1/2)))
    
    unlevered_mean_xs.append((unlevered_df['RP Excess Rets'][i:i+120]).mean() * 12)
    unlevered_std_xs.append((unlevered_df['RP Excess Rets'][i:i+120]).std() * (12 ** (1/2)))    
    unlevered_allocateS_xs.append((unlevered_df['% Stock (RP)'][i:i+120]).mean())
    unlevered_allocateB_xs.append((unlevered_df['% Bonds (RP)'][i:i+120]).mean())
    
    leverage1_mean_xs.append((leverage1_df['RP Excess Rets'][i:i+120]).mean() * 12)
    leverage1_std_xs.append((leverage1_df['RP Excess Rets'][i:i+120]).std() * (12 ** (1/2)))
    leverage1_allocateS_xs.append((leverage1_df['% Stock (RP)'][i:i+120]).mean())
    leverage1_allocateB_xs.append((leverage1_df['% Bonds (RP)'][i:i+120]).mean())

    leverage2_mean_xs.append((leverage2_df['RP Excess Rets'][i:i+120]).mean() * 12)
    leverage2_std_xs.append((leverage2_df['RP Excess Rets'][i:i+120]).std() * (12 ** (1/2)))    
    leverage2_allocateS_xs.append((leverage2_df['% Stock (RP)'][i:i+120]).mean())
    leverage2_allocateB_xs.append((leverage2_df['% Bonds (RP)'][i:i+120]).mean())
    
for mean,std in zip(stock_mean_xs, stock_std_xs):
    stock_sr_xs.append(mean/std)
  
for mean,std in zip(p6040_mean_xs, p6040_std_xs):
    p6040_sr_xs.append(mean/std)
    
for mean,std in zip(unlevered_mean_xs, unlevered_std_xs):
    unlevered_sr_xs.append(mean/std)
    
for mean,std in zip(leverage1_mean_xs, leverage1_std_xs):
    leverage1_sr_xs.append(mean/std)
    
for mean,std in zip(leverage2_mean_xs, leverage2_std_xs):
    leverage2_sr_xs.append(mean/std)

stats_df_list = []

for i in range(0,9,1):
    stats_df_list.append(pd.DataFrame(
    {'Annualized Mean Rets' : [stock_mean_xs[i], p6040_mean_xs[i], unlevered_mean_xs[i], leverage1_mean_xs[i], leverage2_mean_xs[i]],
     'Annualized Std' : [stock_std_xs[i], p6040_std_xs[i], unlevered_std_xs[i], leverage1_std_xs[i], leverage2_std_xs[i]],
     'Sharpe Ratio' : [stock_sr_xs[i], p6040_sr_xs[i], unlevered_sr_xs[i], leverage1_sr_xs[i], leverage2_sr_xs[i]],
     'Average Stock Allocation' : [1, 0.6, unlevered_allocateS_xs[i], leverage1_allocateS_xs[i], leverage2_allocateS_xs[i]],
     'Average Bond Allocation' : [0, 0.4, unlevered_allocateB_xs[i], leverage1_allocateB_xs[i], leverage2_allocateB_xs[i]]},
    index = ['Stock Market Index', '60/40 Portfolio', 'Unlevered RP', 'Levered RP 1', 'Levered RP 2']))
    
# 6. Bar Charts
start_date = 1930
end_date = 1939
for i in range(0,9,1):
    dfi.export(stats_df_list[i], f'Statistics {start_date}-{end_date}.png')
    stats_df_list[i]['Sharpe Ratio'].plot.bar(figsize = (12,6)
        ).set_title(f'Sharpe Ratio of Returns from Jan {start_date} to Dec {end_date}')
    plt.savefig(f'Sharpe Ratios {start_date}-{end_date}.png', bbox_inches = "tight")
    plt.close()
    start_date += 10
    end_date += 10
    
# Performance Comparison

stats_df.plot.scatter(x = "Annualized Std",
                        y = "Annualized Mean Rets", figsize = (18,9)).set_title("Returns versus Risk 1930-2019")
plt.savefig('Returns versus Risk 1930-2019.png')

sr_df = pd.DataFrame(
    {'Stock Market Index' : stock_sr_xs,
     '60/40 Portfolio' : p6040_sr_xs,
     'Unlevered RP' : unlevered_sr_xs,
     'Levered RP 1' : leverage1_sr_xs,
     'Levered RP 2' : leverage2_sr_xs},
    index = [1939,1949,1959,1969,1979,1989,1999,2009,2019])

sr_df.plot(figsize = (12,6)).set_title('Portfolio Performance Comparison (Sharpe Ratios) 1930-2019')
plt.savefig('Portfolio Performance Comparison 1930-2019.png')