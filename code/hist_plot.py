import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

def compute_m(tmp_df):
    idx = (tmp_df['index_price'] - tmp_df['strike']).abs().idxmin()  # Find closest strike
    iv_atm = tmp_df.loc[idx, 'iv']  # Get IV at ATM
    tmp_df['m'] = np.log(tmp_df['strike'] / tmp_df['index_price']) / (np.sqrt(tmp_df['ttm']) * (iv_atm / 100))
    return tmp_df

data_path = f'/media/ignacio/TOSHIBA EXT/data/df_calls.csv'

df = pd.read_csv(data_path, converters={"ttm": pd.to_timedelta}, usecols=['price','timestamp', 'ttm', 'strike', 'index_price', 'iv', 'amount'])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df['ttm'] = df['ttm'].apply(lambda x: x.total_seconds()/ (365.25 * 24 * 3600))
df.timestamp = df.timestamp.dt.floor('h')

t0 = time()

df = df.groupby('timestamp', group_keys=False).apply(compute_m)

print(time() - t0)

df['hour'] = df.timestamp.dt.hour
df['volume_usd'] = df['amount'] * df['price']
df = df[~df.m.isna()]

plt.hist(df['m'], bins = 70, range=(-0.1*100,0.1*100), weights=df['amount'] * df['price'], edgecolor='black', alpha=0.7)
plt.xlabel('Moneyness')
plt.ylabel('Volume (USD)')
plt.grid()
plt.tight_layout()
plt.title('Trade volume across moneyness')
# plt.savefig('../Figures/hist_2d_moneyness.pdf')
plt.show()
plt.figure(figsize=(10, 6))
hist, xedges, yedges, img = plt.hist2d(df['m'], df['hour'], weights=df['volume_usd'], bins=(50, 24), cmap='viridis', range=[[-0.1 * 100, 0.1 * 100], [0, 24]])
plt.colorbar(label='Volume (USD)')
plt.xlabel('Moneyness')
plt.ylabel('Hour')
plt.title('Trade volume across hour and moneyness')
# plt.savefig('../Figures/hist_3d_moneyness.pdf')

df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
print(df['m'].describe())
