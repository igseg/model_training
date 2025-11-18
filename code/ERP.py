import statsmodels.api as sm
import pandas as pd
import numpy  as np
import re
from glob import glob
from scipy.stats import skew
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

from generalized_tools import path_generating, generate_params_cos_1, hes_phi, compensator

# model_name = 'cgmy4_sv_1'
model_name = 'heston_1_step'

import numpy as np

def first_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def second_derivative(f, x, h=1e-5):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)


###### Physical measure

data_btc = pd.read_csv('./sdf/btc_timeseries.csv')
data_btc.timestamp = [pd.to_datetime(date).date() for date in data_btc.timestamp]

## Fit ARCH model

am = arch_model(data_btc.returns * 100, p=1, o=1, q=1, mean='AR', lags=1, dist="StudentsT")
res = am.fit(update_freq=5, disp="off")

data_btc['sigmat'] = res.conditional_volatility / 100

###### Risk neutral measure

path_estimates, path_parameter = path_generating(model_name)
path_estimates, path_parameter = path_estimates[2:] + '*', path_parameter[2:] + '*'

only_calls = True

if only_calls:
    files_rn = glob('/media/ignacio/TOSHIBA EXT/daily_copy_may_22_2025' + path_parameter)
    data_rn = glob('/media/ignacio/TOSHIBA EXT/daily_copy_may_22_2025' + path_estimates)
else:
    files_rn = glob('/home/ignacio/Documents/SFU/finance_lab/cluster/daily' + path_parameter)
    data_rn = glob('/home/ignacio/Documents/SFU/finance_lab/cluster/daily' + path_estimates)

files_rn = sorted(files_rn)
data_rn = sorted(data_rn)

# rn_volatility_1 = lambda u0, lambda_, u_bar, delta_t: np.sqrt(u0 * np.exp(-lambda_ * delta_t) + u_bar * (1 - np.exp(-lambda_ * delta_t)))

## Load phi

tau = 1/365
r = 0
type = ''
k = 1

sigmas_rn = []
dates = []
n_params = len(np.load(files_rn[0]))
P = np.zeros([len(files_rn), n_params])
for i in range(len(files_rn)):
    params = np.load(files_rn[i])
    date = pd.to_datetime(pd.read_csv(data_rn[i]).loc[0, 'date']).date()
    dates.append(date)
    P[i,:] = params
    ## Get S0
    idx = np.where(data_btc.timestamp==date)[0][0]
    S0 = data_btc.loc[idx].index_price

    params_cos = generate_params_cos_1(params, r, tau, S0, k, type, model_name)

    if model_name == 'heston_1_step' or model_name == 'heston_2_step':
        def phi(omega):
            return (hes_phi(omega, *params_cos[:-1]))
    else:
        def phi(omega):
            return compensator(omega, params_cos, model_name)

    mu_1 = (first_derivative(phi,0) * 1j**1).real
    mu_2 = (second_derivative(phi,0) * 1j**2).real

    sigma_rn = np.sqrt(mu_2 - mu_1**2) / np.sqrt(365)
    sigmas_rn.append(sigma_rn)
###### Merge both measures

# print(len(sigmas_rn))
# print(len(dates))
df_rn = pd.DataFrame(data={'sigma_rn': sigmas_rn}, index=dates)
df_rn = pd.merge(df_rn, data_btc[['timestamp', 'returns', 'sigmat']], how='inner', left_index=True, right_on='timestamp')
rows_with_nan = df_rn[df_rn.isna().any(axis=1)]
print(f'Num. nans: {rows_with_nan.shape[0]}')
mask = ~df_rn.isna().any(axis=1)
df_rn = df_rn[mask]
P = P[mask]

###### Regressions

VRP = (df_rn.sigma_rn.values - df_rn.sigmat.values)[1:]
VoV = np.diff(np.log(df_rn.sigma_rn.values))

print(np.sum(np.isnan(VRP)))
print(np.sum(np.isnan(VoV)))
# print(VRP)
# print(VoV)

VRP2 = VRP**2
VoV2 = VoV**2

y = df_rn.returns.values[1:]


# x = np.column_stack((VRP, VoV, VRP2, VoV2, P[1:,0], P[1:,1], P[1:,2], P[1:,3], P[1:,4]))
x = np.column_stack((VRP, VoV, VRP2, VoV2))
x_with_const = sm.add_constant(x)  # Adds a column of 1s

model = sm.OLS(y, x_with_const).fit()

print(model.summary())
