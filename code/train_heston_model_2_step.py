import pandas as pd
import numpy as np
from numba import jit
from time import time
from scipy.optimize import minimize
from tools import (COS_method_call,
                    implied_volatility)
from train_heston_model import _calc_cumulants_heston, hes_levy, hes_phi, test_cos_method

def obj_fun_1(state, params, r, df, loss_f):
    """
    Notation following Andersen et. al. 2017
    """
    n_obs = df.shape[0]
    u0 = state[0]
    lambda_, rho, eta, u_bar = params
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


def obj_fun_2(states, params, r, df, loss_f):
    """
    Notation following Andersen et. al. 2017
    """
    n_obs = df.shape[0]
    lambda_, rho, eta, u_bar = params
    objective = 0
    for j, t in enumerate(df['T'].unique()):
        tmp_df = df[df['T']==t]
        u0 = states[j]
        for i, idx in enumerate(tmp_df.index):
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
                loss = np.pow(option_price - c_obs,2)
            if loss_f == 'price_vega':
                loss = np.pow((option_price - c_obs)/vega,2)
            if loss_f == 'iv':
                try:
                    bsiv = implied_volatility(option_price, S=S0, K=k, T=tau, r=r, option_type="call")
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

def add_iv_estimated_2_step(states, params, tmp_df,r):
    lambda_, rho, eta, u_bar = params
    for j, t in enumerate(tmp_df['T'].unique()):
        df_bis = tmp_df[tmp_df['T']==t]
        u0 = states[j]
        for i, idx in enumerate(df_bis.index):
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

def get_best_estimates(optimizer, best_obj, params):
    if np.isnan(best_obj):
        if np.isnan(optimizer.fun):
            raise ValueError("Optimizer is not working as intended.")
        return optimizer.x, optimizer.fun

    if optimizer.fun < best_obj:
        return optimizer.x, optimizer.fun
    else:
        return params, best_obj

if __name__ == '__main__':

    ## Assess cos_method prices correctly
    if test_cos_method() == False:
        raise ValueError("cos method is not working as intended.")
    
    ## Load Data
    file = '../Data/trainable_df_1h.csv'
    df = pd.read_csv(file)
    df = df.assign(estimated_price=np.zeros(len(df)))
    df = df.assign(iv_estimated=np.zeros(len(df)))
    
    df.timestamp = pd.to_datetime(df.timestamp) - pd.Timedelta(hours=8)
    
    df['date'] = df.timestamp.dt.date
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
    
    params_opt_states = {'method':'Nelder-Mead',
                  'tol':1e-4,
                  'bounds':[[0, np.inf]]
                  # each interval corresponds to the variable: u0, lambda_, rho, eta, u_bar
                 }
    
    params_opt_params = {'method':'Nelder-Mead',
                  'tol':1e-4,
                  'bounds':[[-np.inf, np.inf], [-1, 1], [0, np.inf], [0, np.inf]]
                  # each interval corresponds to the variable: u0, lambda_, rho, eta, u_bar
                 }

    # Define different value ranges for each dimension
    state_0 = np.array([0.01])
    
    # values_1 = np.linspace(0.01, 1, 3)
    values_1 = np.array([0.01])
    # values_2 = np.linspace(-1, 1, 2)
    # values_3 = np.linspace(0.5, 1.5, 2)
    values_2 = np.array([0.0])
    values_3 = np.array([1.0])
    values_4 = np.array([0.01])
    
    mesh = np.meshgrid(values_1, values_2, values_3, values_4, indexing='ij')
    mesh_points_parameters = np.stack(mesh, axis=-1)
    results = []
    t0 = time()
    
    for i, day in enumerate(df['date'].unique()):
        tmp_day_df = df[df['date']==day]
        T = tmp_day_df['T'].unique()
        states = np.zeros(len(T)) + 0.1 ## 1 x T as there is 1 hidden state at each time
        parameters = np.zeros(4) + 0.1 ## 1 x 4 as there are 4 params
        best_obj = np.nan
        prev_best_obj = 0
        for _ in range(5):
            ## train parameters
            first_run = True
            for index in np.ndindex(mesh_points_parameters.shape[:-1]):
                params_0 = mesh_points_parameters[index]
                obj_func_2_opt = lambda x: obj_fun_2(states, x, r, df=tmp_day_df, loss_f=loss_f)
                try:
                    results_step_1 = minimize(obj_func_2_opt, params_0, **params_opt_params)
                    aborted=False
                except (ZeroDivisionError, ValueError):
                    # params = results_step_1.x
                    # best_obj = results_step_1.fun
                    # print('aborted')
                    aborted=True
                    continue
                try:
                    params, best_obj = get_best_estimates(results_step_1, best_obj, params)
                except NameError:
                    params = results_step_1.x
                    best_obj = results_step_1.fun
            ## if no convergence
            if aborted:
                params = parameters
            ## stopping criteria:
    
            print(best_obj - prev_best_obj)
            if np.abs(best_obj - prev_best_obj) < tol:
                break
            else:
                prev_best_obj = best_obj
            
            ## Train states
            for j, t in enumerate(T):
                tmp_hour_df = tmp_day_df[tmp_day_df['T']==t]
                state = np.array([states[j]])
                obj_func_1_opt = lambda x: obj_fun_1(x, params, r, df=tmp_hour_df, loss_f=loss_f)
                results_step_1 = minimize(obj_func_1_opt, state, **params_opt_states)
                if aborted:
                    best_obj = results_step_1.fun
                states[j] = results_step_1.x[0]
        print(f'Loss function: {best_obj:.4f}')
        print(f'Parameters: {params}')
        print(f'States: {states}')
        print('='*70)
    
        ## Save best states and params
        tmp_day_df = add_iv_estimated_2_step(states, params, tmp_day_df,r)
    
        file_estimates = path_estimates + 'heston_2_step_' + str(i).zfill(pad) + '.csv'
        file_parameter = path_parameter + 'heston_2_step_' + str(i).zfill(pad) + '.npz'
    
        np.savez(file_parameter, params=params, states=states)
        tmp_day_df.to_csv(file_estimates, index=False)