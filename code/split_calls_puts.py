import pandas as pd
import numpy  as np
from glob import glob
import re
from time import time

files = glob('/media/ignacio/3b28df90-2e02-4c09-b580-8da764c01346/data/daily/*')
save_path = '../Data/'

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

tardis_timestamp_to_dt = lambda x:  datetime.fromtimestamp(x/1e6)
def tardis_times_to_dt(df, column):
    df[column] = df[column].apply(tardis_timestamp_to_dt)
    return df

def process_file(data_tardis):

    # data_tardis = data_tardis.dropna(subset=["last_price"])
    data_tardis.expiration = pd.to_datetime(data_tardis.expiration, unit='us')
    data_tardis.timestamp = pd.to_datetime(data_tardis.timestamp, unit='us')
    data_tardis = filter_add_ttm(data_tardis)
    data_tardis_calls, data_tardis_puts = filter_split_call_put(data_tardis)
    # df_0dte_calls = df_window(df_0dte_calls)
    # df_0dte_puts = df_window(df_0dte_puts)

    return data_tardis_calls, data_tardis_puts

def tardis_times_to_dt(df, column):
    df[column] = df[column].apply(tardis_timestamp_to_dt)
    return df

def filter_add_ttm(df, ttm_name='ttm', time_name='timestamp', maturity_name='expiration'):
    """
    adds time to maturity
    Times must be as datetime objects
    """
    df[ttm_name] = df[maturity_name] - df[time_name]
    return df

def filter_split_call_put(df, col_name='type', call_name='call', put_name='put'):
    return df[df[col_name]==call_name], df[df[col_name]==put_name]

for i, file in enumerate(files):
    tmp_df = pd.read_csv(file, usecols=columns_to_read)
    if i == 0:
        df = tmp_df.copy()
    else:
        df = pd.concat([df,tmp_df])

df = df.reset_index(drop=True)

df_calls, df_puts = process_file(df)

df_calls.to_csv(save_path + f'calls.csv', index=False)
df_puts.to_csv(save_path + f'puts.csv', index=False)
