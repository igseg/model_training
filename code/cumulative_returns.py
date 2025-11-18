import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

## For puts I am missing the trainable df with puts

def add_iv_atm(tmp_df):
    idx = (tmp_df['index_price'] - tmp_df['k']).abs().idxmin()  # Find closest strike
    iv_atm = tmp_df.loc[idx, 'bsiv']  # Get IV at ATM
    tmp_df['iv_atm'] = iv_atm
    return tmp_df

def strike_price_strategy(tmp_df, i=0):
    tmp_df['k_trading'] = np.exp(tmp_df['ttm'] * tmp_df['iv_atm'] * i * sigma) * tmp_df['index_price']
    idx = (tmp_df['k_trading'] - tmp_df['k']).abs().idxmin()  # Find closest strike
    tmp_df['k_trading'] = tmp_df.loc[idx, 'k']  # Get IV at ATM
    tmp_df = tmp_df[tmp_df['k_trading'] == tmp_df['k']]
    return tmp_df

file = '../Data/trainable_df_1h.csv'
df_quotes = pd.read_csv(file)
df_quotes['timestamp'] = pd.to_datetime(df_quotes['timestamp'])

data_path = f'/media/ignacio/TOSHIBA EXT/data/df_calls.csv'

df = pd.read_csv(data_path, converters={"ttm": pd.to_timedelta}, usecols=['price','timestamp', 'ttm', 'strike', 'index_price', 'iv', 'amount'])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df['ttm'] = df['ttm'].apply(lambda x: x.total_seconds()/ (365.25 * 24 * 3600))
# df.timestamp = df.timestamp.dt.floor('h')
df = df[['timestamp', 'index_price', 'iv']]

sigma = 0.058 # For 0dte Calls

df_merged = pd.merge_asof(df_quotes, df.sort_values('timestamp'), on='timestamp')
df_merged = df_merged.groupby('timestamp', group_keys=False).apply(add_iv_atm)
df_merged = df_merged.groupby('timestamp', group_keys=False).apply(strike_price_strategy).reset_index(drop=True)

df_merged.ttm = np.round(df_merged.ttm * 365 * 24).astype(int) ## Frequency dependent!

## Add underlying price at expiry time
df_merged = df_merged.assign(ST=np.zeros(len(df_merged)))
for idx in df_merged.index:
    try:
        df_merged.loc[idx,'ST'] = df_merged.loc[idx + df_merged.loc[idx, 'ttm'], 'index_price']
    except KeyError:
        continue
df_strategy = df_merged[df_merged.ST != 0]

## Option revenue
df_strategy['revenue'] = df_strategy['ST'] - df_strategy['k']
index = df_strategy[df_strategy['revenue'] <= 0].index
df_strategy.loc[index,'revenue'] = 0

## cumulative Returns
y = ((df_strategy['call_price'] - df_strategy['revenue'])/df_strategy['call_price']).cumsum()
