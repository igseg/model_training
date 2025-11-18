import pandas as pd
import numpy  as np
import re
from glob import glob
from scipy.stats import skew, t
import matplotlib.pyplot as plt
import warnings
import sys
from tools import (implied_volatility,
                    black_scholes_vega)
from windowed_data_to_trainable import load_trainable_df

def dm_test(e1, e2, h=1, power=2):
    """
    Diebold-Mariano test for equal forecast accuracy.

    e1, e2: forecast errors of model 1 and model 2
    h: forecast horizon (1 for one-step)
    power: loss power (2 = squared errors, 1 = absolute)
    """
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)

    # Loss differential
    d = np.abs(e1)**power - np.abs(e2)**power
    T = len(d)

    # Newey–West estimator with lag h-1
    lag = h - 1
    gamma = np.zeros(lag+1)

    for l in range(lag+1):
        gamma[l] = np.sum((d[l:] - d.mean()) * (d[:T-l] - d.mean())) / T

    S = gamma[0] + 2 * np.sum(gamma[1:])

    dm_stat = d.mean() / np.sqrt(S / T)

    # two-sided p-value
    p_value = 2 * (1 - t.cdf(abs(dm_stat), df=T-1))

    return dm_stat, p_value

def calc_loss(df, model):
    loss_vec = ((df[f'estimated_price_{model}'] - df['call_price']) / df['vega'])**2
    return np.mean(loss_vec), loss_vec

if __name__ == '__main__':
    models = ['heston_1', 'cgmy_1', 'variance_gamma_1', 'gaussian_jumps_1', 'cgmy4_sv_1']
    # models = ['cgmy4_sv_1', 'heston_1']
    rows = ['Full Sample', 'otm', 'atm', 'itm']

    results = np.empty((len(rows), len(models)))
    errors = []

    for j, model in enumerate(models):
    # data_rn = glob(f'/home/ignacio/Documents/SFU/finance_lab/cluster/daily/Data/Estimates_{model}/*')
        data_rn = glob(f'/media/ignacio/TOSHIBA EXT/daily_copy_may_22_2025/Data/Estimates_{model}/*')
        data_rn = sorted(data_rn)

        df_estimates_m = pd.read_csv(data_rn[0])
        for file in data_rn[1:]:
            df_estimates_m = pd.concat([df_estimates_m,pd.read_csv(file)])

        df_estimates_m = df_estimates_m.reset_index(drop=True)
        df_estimates_m.timestamp = pd.to_datetime(df_estimates_m.timestamp)
        df_estimates_m['ttm'] = np.round(df_estimates_m['ttm'], 3)
        # n_tot = df_estimates_m.shape[0]
        try:
            df_estimates_m = df_estimates_m.drop(columns='m')
        except KeyError:
            pass
        df_estimates_m = df_estimates_m.rename(columns={'estimated_price': f'estimated_price_{model}'})
        if j==0:
            df_estimates = df_estimates_m[['timestamp', 'k', 'ttm', f'estimated_price_{model}', 'call_price', 'vega']]
        else:
            df_estimates = pd.merge(
                    left=df_estimates,
                    right=df_estimates_m[['timestamp', 'k', 'ttm', f'estimated_price_{model}']],
                    how='inner',
                    left_on=['timestamp', 'k', 'ttm'],
                    right_on=['timestamp', 'k', 'ttm'])


    df_estimates_pre_merge = df_estimates.copy()
    ## df has the moneyness
    df = load_trainable_df('../Data/calls_filtered.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize() + pd.Timedelta(hours=16) - pd.Timedelta(hours=24)
    df['ttm'] = np.round(df['ttm'], 3)
    print(df.shape[0])
    # df_estimates = pd.merge(df_estimates, df['m'], left_index=True, right_index=True)
    df_estimates = pd.merge(
                    left=df_estimates,
                    right=df[['timestamp', 'k', 'm', 'ttm']],
                    how='inner',
                    left_on=['timestamp', 'k', 'ttm'],
                    right_on=['timestamp', 'k', 'ttm'])


    errors = []
    print(f'num. obs: {df_estimates.shape[0]}')
    for j, model in enumerate(models):
        df1 = df_estimates[df_estimates['m'] < -0.01]
        df2 = df_estimates[np.logical_and(-0.01 < df_estimates['m'], df_estimates['m'] < 0.01)]
        df3 = df_estimates[0.01 < df_estimates['m']]
        for i, dataframe in enumerate([df_estimates, df1, df2, df3]):
            if i==3:
                results[i,j], errors_m = calc_loss(dataframe, model)
                errors.append(errors_m)
            else:
                results[i,j], _ = calc_loss(dataframe, model)

    print(pd.DataFrame(results, columns=models, index=rows).T.to_latex())

    errors = np.array(errors)
    results = {}
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            stat, pval = dm_test(errors[:, i], errors[:, j], h=1, power=2)
            results[(i, j)] = (stat, pval)

    for (i, j), (stat, pval) in results.items():
        print(f"Model {i} vs Model {j}: DM = {stat:.3f}, p = {pval:.3f}")
