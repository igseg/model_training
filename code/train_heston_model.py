import pandas as pd
import numpy as np
from numba import jit
from time import time
from scipy.optimize import minimize
from tools import (COS_method_call,
                    implied_volatility)


def obj_fun_states(state, r, df, loss_f):
    """
    Notation following Andersen et. al. 2017
    """
    n_obs = df.shape[0]
    u0, lambda_, rho, eta, u_bar = state
    objective = 0
    for i, idx in enumerate(df.index):
        tau = df.loc[idx, 'ttm']
        S0  = df.loc[idx, 'S0']
        k   = df.loc[idx, 'k']
        c_obs = df.loc[idx, 'call_price']
        bsiv_obs = df.loc[idx, 'bsiv']
        vega = df.loc[idx, 'vega']

        cumulants = _calc_cumulants_heston(S0, tau, k, u0, lambda_, rho, eta, r, u_bar)
        phi_num = lambda x: hes_phi(x, u0, lambda_, rho, eta, r, u_bar, tau, S0, k)
        discount = np.exp(-r * tau)

        ### calc loss
        option_price = COS_method_call(phi_num, discount=discount, K=k, N=128, L=10, c=cumulants, method='call')
        if np.isnan(option_price):
            raise ValueError("Option price is NaN")
        if loss_f == 'price':
            # loss = _loss_option_prices(phi_num, discount, k, cumulants, c_obs)
            loss = np.pow(option_price - c_obs,2)
        if loss_f == 'price_vega':
            # loss = _loss_option_prices_vega(phi_num, discount, k, cumulants, c_obs, vega)
            loss = np.pow((option_price - c_obs)/vega,2)
        if loss_f == 'iv':
            # loss = _loss_imp_vol(phi_num, discount,S0, tau, r, k, cumulants, bsiv_obs)
            try:
                bsiv = implied_volatility(option_price, S=S0, K=k, T=tau, r=r, option_type="call")
                # bsiv = (implied_volatility(option_price, S=S0, K=k, T=tau, r=r, option_type="call")).x
            except ValueError:
                bsiv = 0
            loss = np.pow(bsiv - bsiv_obs,2)

        ###
        if loss > 1e-6:
            objective += loss.real
        # print(loss)
        # print(c_obs)
    objective = np.sqrt(objective / n_obs)
    return objective

def _calc_cumulants_heston(S0, tau, k, u0, lambda_, rho, eta, r, u_bar):

    c = np.zeros(6)
    c[0] = cumulant_0_hes(r, tau, lambda_, u_bar, u0)
    c[1] = cumulant_1_hes(r, tau, lambda_, u_bar, u0, eta, rho)
    return c

def cumulant_0_hes(r, tau, lambda_, u_bar, u0):
    """Compute the first cumulant (c1) based on the given formula."""
    return (r * tau + (1 - np.exp(-lambda_ * tau)) * (u_bar - u0) / (2 * lambda_) - 0.5 * u_bar * tau)

def cumulant_1_hes(r, tau, lambda_, u_bar, u0, eta, rho):
    """Compute the second cumulant (c2) based on the given formula."""
    term1 = (eta * tau * lambda_ * np.exp(-lambda_ * tau) * (u0 - u_bar) * (8 * lambda_ * rho - 4 * eta)) / (8 * lambda_**3)
    term2 = (lambda_ * rho * (1 - np.exp(-lambda_ * tau)) * (16 * u_bar - 8 * u0)) / (8 * lambda_**3)
    term3 = (2 * u_bar * lambda_ * tau * (-4 * lambda_ * rho + eta**2 + 4 * lambda_**2)) / (8 * lambda_**3)
    term4 = (eta**2 * ((u_bar - 2 * u0) * np.exp(-2 * lambda_ * tau) + u_bar * (6 * np.exp(-lambda_ * tau) - 7) + 2 * u0)) / (8 * lambda_**3)
    term5 = (8 * lambda_**2 * (u0 - u_bar) * (1 - np.exp(-lambda_ * tau))) / (8 * lambda_**3)

    return term1 + term2 + term3 + term4 + term5

@jit
def hes_levy(omega, u0, lambda_, rho, eta, mu, u_bar, tau):
    D = np.sqrt((lambda_ - 1j * rho * eta * omega) ** 2 + (omega ** 2 + 1j * omega) * eta ** 2)
    G = (lambda_ - 1j * rho * eta * omega - D) / (lambda_ - 1j * rho * eta * omega + D)

    term1 = 1j * omega * mu * tau
    term2 = (u0 / eta ** 2) * ((1 - np.exp(-D * tau)) / (1 - G * np.exp(-D * tau))) * (lambda_ - 1j * rho * eta * omega - D)
    term3 = (lambda_ * u_bar / eta ** 2) * (tau * (lambda_ - 1j * rho * eta * omega - D) - 2 * np.log((1 - G * np.exp(-D * tau)) / (1 - G)))

    phi = np.exp(term1 + term2) * np.exp(term3)
    return phi

# @jit
def hes_phi(omega, u0, lambda_, rho, eta, mu, u_bar, tau, S0, K):
    x = np.log(S0/K)
    return hes_levy(omega, u0, lambda_, rho, eta, mu, u_bar, tau) * np.exp(1j * omega * x)

def test_cos_method():
    r = 0
    lambda_, rho, eta, mu, u_bar, tau, S0, k = (1.5768, -0.5711, 0.5751, 0, 0.0398, 1, 100, 100)
    u0 = 0.0175

    c = _calc_cumulants_heston(S0, tau, k, u0, lambda_, rho, eta, r, u_bar)
    phi_num = lambda x: hes_phi(x, u0, lambda_, rho, eta, r, u_bar, tau, S0, k)

    phi_sp = phi_num
    discount = 1

    option_price = COS_method_call(phi_sp, discount=discount, K=k, N=128, L=10, c=c, method='call')

    return (np.abs(option_price - 5.785155) < 0.001)

def add_iv_estimated(best_state, tmp_df, r):
    n_obs = tmp_df.shape[0]
    u0, lambda_, rho, eta, u_bar = best_state
    objective = 0
    for i, idx in enumerate(tmp_df.index):
        tau = tmp_df.loc[idx, 'ttm']
        S0  = tmp_df.loc[idx, 'S0']
        k   = tmp_df.loc[idx, 'k']
        c_obs = tmp_df.loc[idx, 'call_price']
        bsiv_obs = tmp_df.loc[idx, 'bsiv']
        vega = tmp_df.loc[idx, 'vega']

        cumulants = _calc_cumulants_heston(S0, tau, k, u0, lambda_, rho, eta, r, u_bar)
        phi_num = lambda x: hes_phi(x, u0, lambda_, rho, eta, r, u_bar, tau, S0, k)
        discount = np.exp(-r * tau)

        option_price = COS_method_call(phi_num, discount=discount, K=k, N=128, L=10, c=cumulants, method='call')
        tmp_df.loc[idx, 'estimated_price'] = option_price
        iv_est = implied_volatility(option_price, S=S0, K=k, T=tau, r=r, option_type="call")
        tmp_df.loc[idx, 'iv_estimated'] = iv_est
    return tmp_df


if __name__ == '__main__':
    ## Assess cos_method prices correctly
    if test_cos_method() == False:
        raise ValueError("cos method is not working as intended.")

    ## Load Data
    file = '../Data/trainable_df_1h.csv'
    df = pd.read_csv(file)
    df = df.assign(estimated_price=np.zeros(len(df)))
    df = df.assign(iv_estimated=np.zeros(len(df)))
    print(df.head())
    ## Parameters
    r=0
    loss_f = 'price_vega'
    tol=1e-3
    t0 = time()
    T = len(df['T'].unique())
    pad = int(np.log10(T)) + 1
    path_estimates = '../Data/Estimates/'
    path_parameter = '../Data/Parameters/'

    params_opt = {'method':'Nelder-Mead',
                  'tol':1e-4,
                  'bounds':[[0, np.inf], [-np.inf, np.inf], [-1, 1], [0, np.inf], [0, np.inf]]
                  # each interval corresponds to the variable: u0, lambda_, rho, eta, u_bar
                 }

    ## Define the meshgrid of starting points

    states = np.zeros((T,5))

    # Define different value ranges for each dimension
    values_0 = np.array([0.01])
    # values_1 = np.linspace(0.01, 1, 3)
    values_1 = np.array([0.01])
    values_2 = np.linspace(-1, 1, 2)
    values_3 = np.linspace(0.5, 1.5, 2)
    values_4 = np.array([0.01])

    mesh = np.meshgrid(values_0, values_1, values_2, values_3, values_4, indexing='ij')
    mesh_points = np.stack(mesh, axis=-1)
    results = []
    t0 = time()
    for i in range(T):

        best_state = np.array([np.nan]*5)
        best_result = np.nan
        first_run = True
        iteration = 0
        tmp_df = df[df['T']==i]

        print('='*70)
        for index in np.ndindex(mesh_points.shape[:-1]):
            iteration+=1
            state = mesh_points[index]
            obj_func_1_opt = lambda x: obj_fun_states(x, r, df=tmp_df, loss_f=loss_f)
            try:
                results_step_1 = minimize(obj_func_1_opt, state, **params_opt)
            except ZeroDivisionError:
                # print('aborted')
                continue
            # print(results_step_1.fun)
            if first_run:
                first_run = False
                best_state = results_step_1.x
                best_result = results_step_1.fun
                if best_result < tol:
                        break
            else:
                if results_step_1.fun < best_result:
                    best_state = results_step_1.x
                    best_result = results_step_1.fun
                    if best_result < tol:
                        break

        states[i] = best_state
        results.append(best_result)
        time_elapsed = time() - t0
        time_per_iter = (time() - t0)/(60 * i+1)
        print(f'Iteration {i+1} of {T+1}')
        print(f'Time per iteration: {time_per_iter:.2f}m')
        print(f'Time remaining: {(T-i) * (time_per_iter / 60):.2f} hours')

        tmp_df = add_iv_estimated(best_state, tmp_df,r)

        file_estimates = path_estimates + 'heston_' + str(i).zfill(pad) + '.csv'
        file_parameter = path_parameter + 'heston_' + str(i).zfill(pad) + '.npy'

        np.save(file_parameter, best_state)
        tmp_df.to_csv(file_estimates, index=False)

        ## Calculate R^2 IV
        #
        # y = tmp_df.bsiv.values
        # y_hat = tmp_df.iv_estimated.values
        # ss_tot = np.sqrt(np.mean((y - np.mean(y))**2))
        # ss_res = y - np.sqrt(np.mean((y - y_hat)**2))
        # r2 = 1 - ss_res / ss_tot
        # print(f'The R² value for the implied volatility is: {r2:.2f}')
