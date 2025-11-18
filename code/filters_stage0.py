import pandas as pd
import numpy  as np
from glob import glob
import re
from time import time

files = glob('../Data/*s.csv')
save_path = '../Data/'

print(files)

columns_to_read = ['symbol',
 'timestamp',
 'local_timestamp',
 'type',
 'strike_price',
 'expiration',
 'open_interest',
 'last_price',
 'bid_price',
 'bid_amount',
 'bid_iv',
 'ask_price',
 'ask_amount',
 'ask_iv',
 'mark_price',
 'mark_iv',
 'underlying_index',
 'underlying_price',
 'delta',
 'gamma',
 'vega',
 'theta',
 'rho']

def filter_num_contracts(tmp_df, min_num_contracts=10):
        idxs = tmp_df.expiration.value_counts()[tmp_df.expiration.value_counts() > min_num_contracts].index.values
        idxs = pd.to_datetime(idxs).date
        cond = tmp_df.expiration.apply(lambda x: x in idxs).values
        tmp_df = tmp_df[cond]
        return tmp_df
def compute_m(tmp_df):
    idx = (tmp_df['underlying_price'] - tmp_df['strike_price']).abs().idxmin()  # Find closest strike
    ttm = pd.to_timedelta(tmp_df['ttm']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    iv_atm = tmp_df.loc[idx, 'mark_iv']  # Get IV at ATM
    tmp_df['m'] = np.log(tmp_df['strike_price'] / tmp_df['underlying_price']) / (np.sqrt(ttm) * (iv_atm/100))
    return tmp_df

for i, name in enumerate(['puts', 'calls']):
    df = pd.read_csv(files[i])
    df.ttm = pd.to_timedelta(df.ttm)
    df.timestamp = pd.to_datetime(df.timestamp).dt.date
    df.expiration = pd.to_datetime(df.expiration).dt.date
    n = len(df)


    print(f'Number of remaining observations for {name}: {len(df)}')
    ## Filter (i)
    df = df[df.bid_price > 0]
    df = df[df.ask_price > 0]
    df = df[df.mark_iv   > 0]
    print(f'Number of remaining observations for {name}: {len(df)}')
    print(f'Number of dropped observations for {name}: {n-len(df)}')
    n = len(df)

    ## Filter (ii)
    df = df[(df.ask_price / df.bid_price) < 5]
    print(f'Number of remaining observations for {name}: {len(df)}')
    print(f'Number of dropped observations for {name}: {n-len(df)}')
    n = len(df)

    ## Filter (iii)
    df = df.groupby('timestamp', group_keys=False).apply(compute_m)
    mu = df.m.mean()
    sigma = df.m.std()
    # cond = np.logical_and(df.m > mu - sigma, df.m < mu + sigma)
    cond = np.logical_and(df.m > -5, df.m < 5)
    df = df[cond]
    df = df.reset_index(drop=True)
    print(f'Number of remaining observations for {name}: {len(df)}')
    print(f'Number of dropped observations for {name}: {n-len(df)}')
    n = len(df)

    ## Filter (iv)
    df = df[df.ttm.dt.total_seconds() / (3600 * 24) > 7]
    print(f'Number of remaining observations for {name}: {len(df)}')
    print(f'Number of dropped observations for {name}: {n-len(df)}')
    n = len(df)

    ## Filter(v)
    df = df.groupby('timestamp', group_keys=False).apply(filter_num_contracts)
    print(f'Number of remaining observations for {name}: {len(df)}')
    print(f'Number of dropped observations for {name}: {n-len(df)}')
    n = len(df)

    df.to_csv(save_path + name + '_filtered.csv', index=False)
    print('=' * 70)
