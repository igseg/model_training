import numpy as np
import scipy.stats as stats
from sympy import exp, Symbol, log, lambdify, pprint, simplify, gamma, sqrt, cosh, sinh, coth
from numba import jit

import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar, bisect
from scipy.optimize import minimize
from sympy import Symbol
from joblib import Parallel, delayed
import pdb
import cobyqa
from scipy.optimize import Bounds, LinearConstraint
np.seterr(divide='ignore')
from math import fabs, erf, erfc, exp
import warnings
# Suppress specific warning
warnings.filterwarnings("ignore", category=RuntimeWarning)
from time import time

@jit
def chi_k(c, d, k, a, b):
    """Computes χ_k(c, d)"""
    factor = 1 / (1 + (k * np.pi / (b - a)) ** 2)
    term1 = np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    term2 = (k * np.pi / (b - a)) * (np.sin(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c))
    return factor * (term1 + term2)

@jit
def psi_k(c, d, k, a, b):
    """Computes ψ_k(c, d)"""
    if k != 0:
        return (np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))) * (b - a) / (k * np.pi)
    else:
        return d - c

def option_valuation(phi,
                      a,
                      b,
                      N,
                      discount,
                      K,
                    option):
    assert option in ['call', 'put']
    exponential_component = lambda factor: np.exp(-1j * factor * a)
    comb = lambda factor: (phi(factor) * exponential_component(factor)).real

    # Vk
    if option == 'call':
        Vk = lambda k: 2/(b-a) * K * (chi_k(c=0, d=b, k=k, a=a, b=b) - psi_k(c=0, d=b, k=k, a=a, b=b))
    if option == 'put':
        Vk = lambda k: 2/(b-a) * K * (-chi_k(c=a, d=0, k=k, a=a, b=b) + psi_k(c=a, d=0, k=k, a=a, b=b))

    # estimation
    v = 0
    for k in range(N):

        #define factor
        factor = k * np.pi / (b-a)
        # print(factor)
        #calc sum
        if k == 0:
            v += comb(factor) * Vk(k) * 0.5
        else:
            v += comb(factor) * Vk(k)
        # print(v)

    # NPV
    v = discount * v

    # print(f'Estimated {option} price in the interval [{a:.2f},{b:.2f}] is: {v:.5f}')
    return v

def parity_call_valuation(phi,
                          a,
                          b,
                          N,
                          discount,
                          K,
                          S0=100):
    """
    Prices the call option trough the put call parity.
    """

    v = option_valuation(phi=phi,
                             a=a,
                             b=b,
                             N=N,
                             discount=discount,
                             K=K,
                        option='put')

    v = v - K * discount + S0

    print(f'Estimated Call price in the interval [{a:.2f},{b:.2f}] is: {v:.2f}')
    return v

def COS_method_call(phi_sp, discount, K, N=1024, L=10, c=None, method='parity'):

    if 'c' not in locals():
        c = np.zeros(6)
        f_derivative = log(phi_sp)
        for i in range(6):
            f_derivative = f_derivative.diff(t)
            c[i] = f_derivative.subs(t, 0) * 1j**-(i+1)
            pprint(simplify(f_derivative))

    # c[5] = 0
    # a = c[0] - L * np.sqrt(c[1] + np.sqrt(c[3]+ np.sqrt(c[5])))
    # b = c[0] + L * np.sqrt(c[1] + np.sqrt(c[3]+ np.sqrt(c[5])))

    a = c[0] - 12 * np.sqrt(np.abs(c[1]))
    b = c[0] + 12 * np.sqrt(np.abs(c[1]))

    # Complex component
    # print(a,b)
    # phi = lambdify((t), phi_sp, modules=['numpy'])
    # phi = phi_sp
    if method == 'call':
        return option_valuation(phi=phi_sp,
                                     a=a,
                                     b=b,
                                     N=N,
                                     discount=discount,
                                     K=K,
                               option='call')
    if method == 'parity':
        return parity_call_valuation(phi=phi_sp,
                                     a=a,
                                     b=b,
                                     N=N,
                                     discount=discount,
                                     K=K)

@jit
def black_scholes_price_numba(S, K, T, r, sigma, option_type="call"):
    """Computes Black-Scholes option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # pdb.set_trace()
    if option_type == "call":
        return S * ndtr_numba(d1) - K * np.exp(-r * T) * ndtr_numba(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * ndtr_numba(-d2) - S * ndtr_numba(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

def implied_volatility(C_mkt, S, K, T, r, option_type="call"):
    """Computes implied volatility using the Brent method."""
    func = lambda sigma: black_scholes_price_numba(S, K, T, r, sigma, option_type) - C_mkt
    try:
        return brentq(func, 1e-6, 5.0)  # Solving for sigma in a reasonable range
    except ValueError:
        return np.nan
    # return newton(func, x0=1e-6, x1=5.0)  # Solving for sigma in a reasonable range
    # return bisect(func,1e-5, 5.0)  # Solving for sigma in a reasonable range
    # return minimize_scalar(func, bounds=(1e-6, 5.0), method='bounded')

def black_scholes_vega(S, K, T, r, sigma):
    """
    Computes the Black-Scholes Vega (sensitivity of option price to volatility).

    Parameters:
    S : float  -> Current stock price
    K : float  -> Strike price
    T : float  -> Time to expiration (in years)
    r : float  -> Risk-free interest rate (as a decimal)
    sigma : float  -> Volatility of the underlying asset (as a decimal)

    Returns:
    float -> Vega (change in option price per 1% change in volatility)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega

def _loss_option_prices(phi_sp, discount, k, c, c_obs):
    option_price = COS_method_call(phi_sp, discount=discount, K=k, N=128, L=10, c=c, method='call')
    # print(option_price)
    return np.pow(option_price - c_obs,2)

def _loss_option_prices_vega(phi_sp, discount, k, c, c_obs, vega):
    option_price = COS_method_call(phi_sp, discount=discount, K=k, N=128, L=10, c=c, method='call')
    # print(option_price)
    return np.pow((option_price - c_obs)/vega,2)

def _loss_imp_vol(phi_sp, discount,S0,tau,r, k, c, bsiv_obs):
    option_price = COS_method_call(phi_sp, discount=discount, K=k, N=128, L=10, c=c, method='call')
    try:
        bsiv = implied_volatility(option_price, S=S0, K=k, T=tau, r=r, option_type="call")
        # bsiv = (implied_volatility(option_price, S=S0, K=k, T=tau, r=r, option_type="call")).x
    except ValueError:
        bsiv = 1
    return np.pow(bsiv - bsiv_obs,2)

NPY_SQRT1_2 = 1.0/ np.sqrt(2)
@jit(cache=True, fastmath=True)
def ndtr_numba(a):

    if (np.isnan(a)):
        return np.nan
    x = a * NPY_SQRT1_2;
    z = fabs(x)
    if (z < NPY_SQRT1_2):
        y = 0.5 + 0.5 * erf(x)
    else:
        y = 0.5 * erfc(z)
        if (x > 0):
            y = 1.0 - y
    return y

def get_best_estimates(optimizer, best_obj, params):
    if np.isnan(best_obj):
        if np.isnan(optimizer.fun):
            raise ValueError("Optimizer is not working as intended.")
        return optimizer.x, optimizer.fun

    if optimizer.fun < best_obj:
        return optimizer.x, optimizer.fun
    else:
        return params, best_obj

@jit
def gaussian_jumps_alpha(u, V, c, mu, sigma, tau):
    term1 = - u * V * 0.5
    term2 = u**2 * V * 0.5
    mu_sim = np.exp(mu + sigma**2 * 0.5) - 1
    term3 = c * (np.exp(u * mu + u**2 * sigma**2 *0.5) - 1 - u * mu_sim)
    return tau * (term1 + term2 + term3)

@jit
def gaussian_jumps_phi(u, V, c, mu, sigma, tau):
    return np.exp(gaussian_jumps_alpha(u, V, c, mu, sigma, tau))

@jit
def variance_gamma_phi_compensated(omega, sigma, nu, theta, tau, S0, k, r):
    x = np.log(S0/k)
    w = - np.log(phi_vg(-1j, sigma, nu, theta, tau, r))/tau
    w = np.log(1 - theta * nu - sigma**2 * nu * 0.5)/nu
    return  variance_gamma_phi(omega, sigma, nu, theta, tau, S0, k, r) *  np.exp(1j * omega * w * tau) * np.exp(1j * omega * x)
@jit
def variance_gamma_phi(omega, sigma, nu, theta, tau, S0, k, r) :
    # return phi_vg(omega, sigma, vega, theta, tau, r) * np.exp(1j * omega * r * tau)
    # w = np.log(1 - theta * nu - sigma**2 * nu * 0.5)/nu
    # x = np.log(S0/k)
    return phi_vg(omega, sigma, nu, theta, tau, r) * np.exp(1j * omega * r * tau)
    # return phi_vg(omega, sigma, nu, theta, tau, r) * np.exp(1j * omega * (x + (r + w)*tau))

@jit
def phi_vg(omega, sigma, nu, theta, tau, r):
    return (1 - 1j * theta * nu * omega + 0.5 * sigma**2 * nu * omega**2)**(-tau/nu)
    # return (1 - 1j * omega * theta * vega + 0.5 * sigma**2 * vega * omega**2)**(-tau/vega)
