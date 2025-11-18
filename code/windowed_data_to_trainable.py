import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from tools import (implied_volatility,
                    black_scholes_vega)

window = '1day'
files = ['../Data/calls_filtered.csv', '../Data/puts_filtered.csv']
save_file = f'../Data/trainable_df_{window}.csv'

def load_trainable_df(file):
    usecols=['strike_price', 'timestamp', 'bid_price', 'ask_price', 'ttm', 'underlying_price', 'm', 'type']
    df = pd.read_csv(file, usecols=usecols)
    df['mid_price'] = np.mean([df['bid_price'],df['ask_price']],axis=0)
    df = df.drop(columns=['bid_price', 'ask_price'])
    df['ttm'] = pd.to_timedelta(df['ttm']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    df['timestamp'] = df.timestamp.astype('category')
    df = df.rename(columns={'strike_price':'k', 'underlying_price': 'S0'})
    df['mid_price'] = df['mid_price'] * df['S0']
    df = df.reset_index(drop=True)
    df['T'] = df.timestamp.cat.codes
    df['bsiv'] = df.apply(lambda row: implied_volatility(row['mid_price'], row['S0'], row['k'], row['ttm'], 0, option_type=row['type']), axis=1)
    df['vega'] = df.apply(lambda row: black_scholes_vega(row['S0'], row['k'], row['ttm'], 0, row['bsiv']), axis=1)
    return df

if __name__ == '__main__':
    df1 = load_trainable_df(files[0])
    print('Missing values:')
    print(df1.isna().sum())

    # df1 = df1.dropna(subset=["bsiv"])
    # df1.to_csv(save_file, index=False)

    df2 = load_trainable_df(files[1])
    print('Missing values:')
    print(df2.isna().sum())

    df = pd.concat([df1, df2])
    df = df.sort_values(by="T")
    df = df.reset_index(drop=True)
    df = df.dropna(subset=["bsiv"])

    df.to_csv(save_file, index=False)

    df = pd.read_csv('../Data/trainable_df_1day.csv')
    df.timestamp = pd.to_datetime(df.timestamp)
    timestamp =pd.Timestamp('2020-01-01 00:00:00')
    day = (df.timestamp - timestamp).dt.total_seconds()/(24 * 3600)
    plt.plot(day.unique())
    plt.show()
