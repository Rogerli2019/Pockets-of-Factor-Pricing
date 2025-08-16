#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 15:29:57 2025

@author: lyh2019
"""

import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


# # Specify the path to your CSV file
# # Explicitly specify data types for each column
# dtypes = {
#     'DivInit': 'float64',
#     'DivOmit': 'float64',
#     'ExchSwitch': 'float64',
#     'FirmAge': 'float64',
#     'IndIPO': 'float64',
#     'Spinoff': 'float64'
# }

# csv_file_path = "/Users/lyh2019/Desktop/Asset Pricing/Factor Investment and Empirical Finance/signed_predictors_all_wide.csv"

# dask_df = dd.read_csv(csv_file_path, dtype=dtypes)
# dask_df['yyyymm'] = dd.to_datetime(dask_df['yyyymm'], format='%Y%m')
# columns_list = dask_df.columns.tolist()
# print(columns_list)

# list_signal = ['BMdec','Beta','Mom12m','Size','AssetGrowth','OperProf']
# dask_df_select = dask_df.loc[:, ['permno', 'yyyymm']+list_signal]
# pandas_df = dask_df_select.compute()
# pandas_df['yyyymm'] = pandas_df['yyyymm'].dt.year*100 + pandas_df['yyyymm'].dt.month

# del dask_df
# del dask_df_select


# # username:lyh2016
# # password:Xwlyh960604#2025
# # import wrds
# # conn=wrds.Connection(wrds_username='lyh2016')

# crsp_m = pd.read_csv('/Users/lyh2019/Desktop/Paper 2/crsp_m.csv')

# # change variable format to int
# crsp_m[['permco','permno','shrcd','exchcd']]=crsp_m[['permco','permno','shrcd','exchcd']].astype(int)

# ### Line up date to be end of month
# ### jdate is the correct date
# crsp_m['date']=pd.to_datetime(crsp_m['date'])
# crsp_m['jdate']=crsp_m['date']+pd.offsets.MonthEnd(0)


# # add delisting return
# dlret = pd.read_csv('/Users/lyh2019/Desktop/Paper 2/dlret.csv')

# dlret.permno=dlret.permno.astype(int)
# dlret.dlstcd=dlret.dlstcd.astype(int)
# dlret['dlstdt']=pd.to_datetime(dlret['dlstdt'])

# ## Monthend aligns to the last day of each month
# dlret['jdate']=dlret['dlstdt']+pd.offsets.MonthEnd(0)

# # Adjust for delisting returns by p100 - empirical asset pricing book
# dlstcd_vlues = [500, 520, 574, 580, 584] + list(range(551, 574 + 1))

# # replace 500 related in 30%
# dlret['dlret'] = np.where((dlret['dlret'].isnull())&(dlret['dlstcd'].isin(dlstcd_vlues)),-0.3,dlret['dlret'])

# # replace other with 100%
# dlret['dlret'] = np.where((dlret['dlret'].isnull()),-1,dlret['dlret'])

# crsp = pd.merge(crsp_m, dlret, how='left',on=['permno','jdate'])
# crsp['dlret']=crsp['dlret'].fillna(0)
# crsp['ret']=crsp['ret'].fillna(0)
# crsp['retadj']=(1+crsp['ret'])*(1+crsp['dlret'])-1 # this is used in further analysis


# crsp = crsp[crsp['shrcd']==(10|11)] # restrict to share code 10 or 11
# crsp['mktCap'] = np.abs(crsp['prc']) * crsp['shrout']/1000
# crsp = crsp.sort_values(['permno', 'jdate'])
# crsp['prev_mktCap'] = crsp.groupby('permno')['mktCap'].shift(1)

# # cut the previous 20% bound
# thresholds = crsp[crsp['exchcd']==2].groupby('date')['prev_mktCap'].quantile(0.2).reset_index()
# crsp = pd.merge(crsp, thresholds, on='date', suffixes=('', '_threshold'))
# crsp_filtered = crsp[crsp['prev_mktCap'] >= crsp['prev_mktCap_threshold']]


# crsp_filtered = crsp_filtered[['permno','jdate','prev_mktCap','retadj']]
# crsp_filtered['yyyymm'] = crsp_filtered['jdate'].dt.year*100 + crsp_filtered['jdate'].dt.month
# crsp_filtered.rename(columns={'prev_mktCap': 'mktCap'}, inplace=True)
# crsp_filtered.rename(columns={'retadj': 'ret'}, inplace=True)
# crsp_filtered = crsp_filtered[['permno','yyyymm','mktCap','ret']]

# df = crsp_filtered.merge(pandas_df, on = ['permno','yyyymm'],how = 'left')

# del pandas_df
# del crsp
# del crsp_filtered
# del crsp_m
# del dlret

# # Columns to process
# list_signal = ['BMdec', 'Beta', 'Mom12m', 'Size', 'AssetGrowth', 'OperProf']

# # 1. Fill missing values with monthly cross-sectional mean
# for col in list_signal:
#     df[col] = df.groupby('yyyymm')[col].transform(lambda x: x.fillna(x.mean()))
    
# for col in list_signal:
#     df[col] = df.groupby('yyyymm')[col].transform(
#         lambda x:
#           (x.rank(method='max') - 1)   # Ranks start at 1, convert to 0-based
#             / (len(x)-1)              # Divide by (n-1) to scale to [0, 1]
#             - 0.5                       # Shift to [-0.5, 0.5]
#     )


import wrds
conn=wrds.Connection(wrds_username='lyh2016')
# Xwlyh960604#2025

ff = conn.get_table(library='ff', table='fivefactors_monthly')
ff['date']=pd.to_datetime(ff['date'])
ff['dateff']=pd.to_datetime(ff['dateff'])
ff.set_index('date', inplace=True)  # Set 'jdate' as index
ff['yyyymm'] = ff['dateff'].dt.year*100 + ff['dateff'].dt.month

# # Make sure to initialize the 'ret_decos' column in df
# df['ret_decos'] = None

# # Loop through each row in df
# for idx in df.index:
#     # Select the current 'yyyymm' from df
#     current_yyyymm = df.loc[idx, 'yyyymm']

#     # Use .loc to find the matching row in ff
#     ff_row = ff.loc[ff['yyyymm'] == current_yyyymm]

#     # If a matching row is found, calculate 'ret_decos'
#     if not ff_row.empty:
#         df.loc[idx, 'ret_decos'] = (
#             df.loc[idx, 'ret'] - ff_row['mktrf'].values[0] - ff_row['rf'].values[0]
#         )

# df.to_parquet("/Users/lyh2019/Desktop/Paper 2/ff6_cs_20.parquet")

###############################################################################
# Cross-sectional WLS
def epanechnikov_weights(tau, t_vals, bandwidth):
    u = (t_vals - tau) / bandwidth
    weights = 1.5 * (1 - u**2) * ((u > -1) & (u < 0))  # one-sided kernel: u ∈ (-1, 0)
    return weights

df = pd.read_parquet("/Users/lyh2019/Desktop/Paper 2/ff6_cs_20.parquet")
###############################################################################
# step 1. run the WLS regression to gain coefficients - used for forecast later
predictors = ['BMdec', 'Beta', 'Mom12m', 'Size', 'AssetGrowth', 'OperProf']
results = []

for date, group in df.groupby('yyyymm'):
    row = {'yyyymm': date}
    for var in predictors:
        try:
            formula = f'ret_decos ~ {var}'
            model = smf.wls(formula=formula, data=group, weights=group['mktCap']).fit()
            row[f'{var}'] = model.params[var]
            row[f'const_{var}'] = model.params['Intercept']
        except Exception:
            row[f'{var}'] = None
            row[f'const_{var}'] = None
    results.append(row)

coef_df = pd.DataFrame(results).sort_values('yyyymm').reset_index(drop=True)

###############################################################################
# step 2. Make out-sample forecast
M = 60  # Rolling window size
g = 2 # follow Timmermann
# Ensure data is sorted by yyyymm
df = df.sort_values('yyyymm')
coef_df = coef_df.sort_values('yyyymm').reset_index(drop=True)
months = coef_df['yyyymm'].tolist()

predictions = []
for i in range(M, len(months) - 1):
    month_t = months[i]
    month_t1 = months[i + 1]

    for var in predictors:
        # Get coefficient time series
        t_vals = np.arange(i - M, i)  # time indices for kernel
        tau = i  # current time

        # Extract historical intercepts and slopes
        past_slopes = coef_df.loc[i - M:i - 1, var].astype(float).to_numpy()
        past_intercepts = coef_df.loc[i - M:i - 1, f'const_{var}'].astype(float).to_numpy()

        # Compute one-sided kernel weights
        w = epanechnikov_weights(tau, t_vals, bandwidth=M)
        if w.sum() == 0:
            continue  # skip if no weight

        # Normalize weights
        w = w / w.sum()

        # Kernel-weighted average
        avg_slope = np.sum(w * past_slopes)
        avg_intercept = np.sum(w * past_intercepts)/(g+1) # g-prior


        # Get firm-level data for month t+1
        df_t1 = df[df['yyyymm'] == month_t1].copy()
        df_t1['pred_ret_decos'] = avg_intercept + avg_slope * df_t1[var]
        df_t1['predictor'] = var
        # df_t1['forecast_month'] = month_t1

        predictions.append(df_t1[['permno', 'yyyymm', 'predictor', 'pred_ret_decos']])

###############################################################################
# step 3. Compute SED measure
# Combine all predictions into one DataFrame
forecast_df = pd.concat(predictions)

# Merge with actual data to get ret_decos and mktCap
df_actual = df[['permno', 'yyyymm', 'ret_decos', 'mktCap']]
merged = forecast_df.merge(df_actual, on=['permno', 'yyyymm'], how='left')

# Compute SED components
merged['squared_actual'] = merged['ret_decos'] ** 2
merged['squared_error'] = (merged['ret_decos'] - merged['pred_ret_decos']) ** 2
merged['gain'] = merged['squared_actual'] - merged['squared_error']
merged['weighted_gain'] = merged['gain'] * merged['mktCap']

# Compute weighted average SED per predictor per month
sed_df = (
    merged.groupby(['yyyymm', 'predictor'])
          .apply(lambda g: g['weighted_gain'].sum() / g['mktCap'].sum())
          .unstack()
          .reset_index()
)

###############################################################################
# one-side kernal regression
from statsmodels.regression.linear_model import WLS
from statsmodels.tools.tools import add_constant


bandwidth = 12
pocket_flags = pd.DataFrame({'yyyymm': sed_df['yyyymm']})
predictors = sed_df.columns.drop('yyyymm')
date_index = np.arange(len(sed_df))

for pred in predictors:
    sed_series = sed_df[pred].values.copy()
    flags = []

    for t in range(len(sed_df)):
        if t < bandwidth:
            flags.append(np.nan)
            continue

        y = sed_series[t - bandwidth:t]
        x = date_index[t - bandwidth:t]
        w = epanechnikov_weights(t, x, bandwidth)

        if w.sum() == 0 or np.any(np.isnan(y)):
            flags.append(np.nan)
            continue

        X = np.column_stack([np.ones_like(x), x])  # intercept + time
        model = WLS(y, X, weights=w).fit()
        gamma_0 = model.params[0]
        t_stat_gamma_0 = model.tvalues[0]

        # Only assign in-pocket flag if γ₀ > 0 and significant
        if (gamma_0 > 0) and (abs(t_stat_gamma_0) > 1.645):
            flags.append(1)
        else:
            flags.append(0)

    pocket_flags[f'{pred}_in_pocket'] = flags


import matplotlib.pyplot as plt
import seaborn as sns

# Prep data for heatmap
pocket_plot = pocket_flags.copy()
pocket_plot = pocket_plot.set_index('yyyymm')
pocket_plot.columns = [col.replace('_in_pocket', '') for col in pocket_plot.columns]

# Transpose so predictors are on y-axis
pocket_matrix = pocket_plot.T

# Make sure data is float for seaborn (NaN, 0, 1)
pocket_matrix = pocket_matrix.astype(float)

# Plot
plt.figure(figsize=(14, 6))
sns.heatmap(pocket_matrix, cmap="Greys", cbar=False, linewidths=0.1, linecolor='lightgray')

plt.title('Pocket Detection Heatmap')
plt.xlabel('Time (yyyymm)')
plt.ylabel('Predictor')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


# List of '_in_pocket' columns
pocket_cols = [
    'AssetGrowth_in_pocket',
    'BMdec_in_pocket',
    'Beta_in_pocket',
    'Mom12m_in_pocket',
    'OperProf_in_pocket',
    'Size_in_pocket'
]

# Count zeros
zero_counts = (pocket_flags[pocket_cols] == 0).sum()

# Count ones
one_counts = (pocket_flags[pocket_cols] == 1).sum()

# Count NaNs
nan_counts = pocket_flags[pocket_cols].isna().sum()

# Combine counts into a single DataFrame
counts_df = pd.DataFrame({
    'Zeros': zero_counts,
    'Ones': one_counts,
    'NaNs': nan_counts
})

# Display the result
print(counts_df)






###############################################################################
# Step 1: Extract year
pocket_year = pocket_flags.copy()
pocket_year['year'] = pocket_year['yyyymm'] // 100

# Step 2: Average in-pocket flags by year
pocket_year_avg = (
    pocket_year.drop(columns='yyyymm')
    .groupby('year')
    .mean()
    .rename(columns=lambda x: x.replace('_in_pocket', ''))
)

# Step 3: Transpose for heatmap (predictors on y-axis)
pocket_heatmap_year = pocket_year_avg.T

# Step 4: Plot heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(pocket_heatmap_year, cmap='Blues', linewidths=0.1, linecolor='gray',
            vmin=0, vmax=1, cbar_kws={'label': 'Fraction In-Pocket'})

plt.title('Annual In-Pocket Frequency by Predictor')
plt.xlabel('Year')
plt.ylabel('Predictor')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

###############################################################################

import matplotlib.pyplot as plt

# Set up the figure and axes
predictors = sed_df.columns.drop('yyyymm')
n = len(predictors)

fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(14, 2.5 * n), sharex=True)

# Convert yyyymm to datetime for nicer x-axis
sed_df_plot = sed_df.copy()
sed_df_plot['date'] = pd.to_datetime(sed_df_plot['yyyymm'].astype(str), format='%Y%m')

# Plot each predictor's SED
for i, pred in enumerate(predictors):
    ax = axes[i]
    ax.plot(sed_df_plot['date'], sed_df_plot[pred], label=pred)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel(pred)
    ax.grid(True, linestyle='--', alpha=0.5)
    if i == 0:
        ax.set_title('SED Measures Over Time (One Panel Per Predictor)')

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.show()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
import statsmodels.api as sm

ar_coeffs = {}

# Drop missing values just in case
sed_data = sed_df.drop(columns='yyyymm').copy()

for col in sed_data.columns:
    y = sed_data[col].dropna().values

    # Lag y by 1 to build AR(1)
    y_lag = y[:-1]
    y_now = y[1:]

    X = sm.add_constant(y_lag)  # add intercept
    model = sm.OLS(y_now, X).fit()

    intercept, phi1 = model.params
    ar_coeffs[col] = {
        'phi1': phi1,
        'intercept': intercept,
        't_stat_phi1': model.tvalues[1],
        'r_squared': model.rsquared
    }
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# Step 1: Merge factor returns and pocket flags
ff_merged = ff.copy()
pockets = pocket_flags.copy()

ff_merged = ff_merged.sort_values('yyyymm')
pockets = pockets.sort_values('yyyymm')
merged = ff_merged.merge(pockets, on='yyyymm', how='inner')

# Step 2: Define mapping between factors and predictors
mapping = {
    'mktrf': 'Beta',
    'smb': 'Size',
    'hml': 'BMdec',
    'rmw': 'OperProf',
    'cma': 'AssetGrowth',
    'umd': 'Mom12m'
}

# Step 3: Compute Sharpe ratios cleanly
results = []

for factor, predictor in mapping.items():
    flag_col = f"{predictor}_in_pocket"
    
    df_tmp = merged[[factor, flag_col]].dropna()  # drop rows where either is NaN
    
    in_pocket = df_tmp[df_tmp[flag_col] == 1][factor]
    out_pocket = df_tmp[df_tmp[flag_col] == 0][factor]
    all_data = df_tmp[factor]
    
    # Annualized Sharpe ratios
    sharpe_all = all_data.mean() / all_data.std() * np.sqrt(12)
    sharpe_in = in_pocket.mean() / in_pocket.std() * np.sqrt(12) if len(in_pocket) > 1 else np.nan
    sharpe_out = out_pocket.mean() / out_pocket.std() * np.sqrt(12) if len(out_pocket) > 1 else np.nan

    results.append({
        'factor': factor,
        'matched_predictor': predictor,
        'sharpe_all': sharpe_all,
        'sharpe_in_pocket': sharpe_in,
        'sharpe_out_of_pocket': sharpe_out
    })

sharpe_df = pd.DataFrame(results).round(4)
sharpe_df
