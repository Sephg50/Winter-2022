# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:53:00 2022

@author: seph
"""

### Calculation of Rolling Betas with Rolling Regressions ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.rolling import RollingOLS

df = pd.read_csv('TSLA_Data.csv',
                 index_col = 'Date',
                 parse_dates = True).rename(columns = {"Ret" : "TSLA"})

for col in df:
    df[f'{col}_logrets'] = np.log(df[f'{col}'] + 1)
    
df['TSLA_elogrets'] = df['TSLA_logrets'] - df['RF_logrets']    
df['MKT_elogrets'] = df['MKT_logrets'] - df['RF_logrets']    

df['TSLA_rollingstd'] = df['TSLA_logrets'].rolling(window = 253).std()
df['MKT_rollingstd'] = df['MKT_logrets'].rolling(window = 253).std()

df['TSLA_3day_logrets'] = df['TSLA_logrets'].rolling(3).sum()
df['MKT_3day_logrets'] = df['MKT_logrets'].rolling(3).sum()
df['RF_3day_logrets'] = df[ 'RF_logrets'].rolling(3).sum()

df['TSLA_3day_elogrets'] = df['TSLA_3day_logrets'] - df['RF_3day_logrets']    
df['MKT_3day_elogrets'] = df['MKT_3day_logrets'] - df['RF_3day_logrets']  

df['rolling_corr'] = df['TSLA_3day_elogrets'].rolling(window=253).corr(df['MKT_3day_elogrets'])

df['rolling_beta'] = df['rolling_corr'] * (df['TSLA_rollingstd'] / df['MKT_rollingstd'])

df['rr_beta'] = RollingOLS(df['TSLA_elogrets'],df['MKT_elogrets'],window=253).fit().params

df.plot(y = ['rolling_beta', 'rr_beta'],
              figsize = (12,6)).set_title('Rolling Betas')
plt.savefig('Rolling Betas from 2011 to 2021.png')

output_df = df[['TSLA_rollingstd','MKT_rollingstd','rolling_corr','rolling_beta','rr_beta']] # <-- data starts on 2011-06 
output_df.to_csv('FIN687-HW5-Output.csv')
