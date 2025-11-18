import numpy as np
from numba import jit, njit, prange
from train_heston_model_2_step import hes_phi
from tools import chi_k, psi_k, gaussian_jumps_phi, variance_gamma_phi_compensated, phi_vg, gaussian_jumps_alpha
from scipy.special import gamma
## Pricing functions

def COS_method_call_general(params_cos, discount, K, model_name, N=None, L=10, c=None):
    # if 'c' not in locals():
    #     c = np.zeros(6)
    #     f_derivative = log(phi_sp)
    #     for i in range(6):
    #         f_derivative = f_derivative.diff(t)
    #         c[i] = f_derivative.subs(t, 0) * 1j**-(i+1)
    #         pprint(simplify(f_derivative))
    if 'heston' in model_name:
        a = c[0] - 12 * np.sqrt(np.abs(c[1]))
        b = c[0] + 12 * np.sqrt(np.abs(c[1]))
    else:
        a = c[0] - L * np.sqrt(c[1])
        b = c[0] + L * np.sqrt(c[1])

    return option_valuation_general(params_cos=params_cos,
                                     a=a,
                                     b=b,
                                     N=N,
                                     discount=discount,
                                     K=K,
                               option=params_cos[-1],
                               model_name=model_name)

def option_valuation_general(params_cos,
                          a,
                          b,
                          N,
                          discount,
                          K,
                        option,
                        model_name):
    assert option in ['call', 'put']
    if model_name == 'heston_1_step' or model_name == 'heston_2_step':
        def phi(omega):
            return hes_phi(omega, *params_cos[:-1])
            # def hes_phi(omega, u0, lambda_, rho, eta, mu, u_bar, tau, S0, K):
    if model_name == 'gaussian_jumps_1':
        # def phi(omega):
        #     return gaussian_jumps_phi(omega, *params_cos)
        def phi(omega):
            return compensator(omega, params_cos, model_name)
                #  V, c, mu, sigma, tau
    if model_name == 'variance_gamma_1':
        # def phi(omega):
        #     return variance_gamma_phi_compensated(omega, *params_cos)
        #         # sigma, vega, theta, tau, S0, k, r
        def phi(omega):
            return compensator(omega, params_cos, model_name)

    if model_name == 'cgmy_1':
        def phi(omega):
            # C, G, M, Y, sigma, tau, S0, k, r = params_cos
            return compensator(omega, params_cos, model_name)

    if model_name == 'cgmy_sv_1':
        def phi(omega):
            # C, G, M, Yp, Yn, eta, kappa, eta, lambd, tau, S0, k, r = params_cos
            return compensator(omega, params_cos, model_name)

    if model_name == 'cgmy4_sv_1':
        def phi(omega):
            # C, G, M, Y, kappa, eta, lambd, tau, S0, k, r = params_cos
            return compensator(omega, params_cos, model_name)


    def exponential_component(factor):
        return np.exp(-1j * factor * a)
    # exponential_component = lambda factor: np.exp(-1j * factor * a)
    def comb(factor):
        return (phi(factor) * exponential_component(factor)).real
    # comb = lambda factor: (phi(factor) * exponential_component(factor)).real

    # Vk
    if option == 'call':
        Vk = lambda k: 2/(b-a) * K * (chi_k(c=0, d=b, k=k, a=a, b=b) - psi_k(c=0, d=b, k=k, a=a, b=b))
    if option == 'put':
        Vk = lambda k: 2/(b-a) * K * (-chi_k(c=a, d=0, k=k, a=a, b=b) + psi_k(c=a, d=0, k=k, a=a, b=b))

    # estimation
    v = 0

    factors = np.arange(N) * np.pi / (b - a)
    combs = comb(factors)            # Vectorized comb
    Vks =  np.array([Vk(k) for k in range(N)], dtype=np.float32)           # Example vectorized Vk
    Vks[0] *= 0.5                    # Adjust first element
    v = np.inner(Vks,combs)
    # for k in range(N):
    #     factor = k * np.pi / (b-a)
    #     # print(phi(factor))
    #     if k == 0:
    #         v += comb(factor) * Vk(k) * 0.5
    #     else:
    #         v += comb(factor) * Vk(k)
    v = discount * v
    return v

## tests

def test_cos_method(model_name):
    if model_name == 'heston_2_step':
        r = 0
        lambda_, rho, eta, mu, u_bar, tau, S0, k = (1.5768, -0.5711, 0.5751, 0, 0.0398, 1, 100, 100)
        u0 = 0.0175

        states = np.array([u0])
        params = [lambda_, rho, eta, u_bar]
        c = _calc_cumulants(tau, r, states, params, model_name)

        discount = 1
        params_cos = [u0, lambda_, rho, eta, mu, u_bar, tau, S0, k, 'call']

        option_price = COS_method_call_general(params_cos, discount=discount, K=k, N=128, L=10, c=c, model_name=model_name)

        return (np.abs(option_price - 5.785155) < 0.001)

    if model_name == 'heston_1_step':
        r = 0
        lambda_, rho, eta, mu, u_bar, tau, S0, k = (1.5768, -0.5711, 0.5751, 0, 0.0398, 1, 100, 100)
        u0 = 0.0175

        states = np.array([u0, lambda_, rho, eta, u_bar])
        params = [1,2,3,4]
        c = _calc_cumulants(tau, r, states, params, model_name)

        discount = 1
        params_cos = [u0, lambda_, rho, eta, mu, u_bar, tau, S0, k, 'call']

        # option_price = COS_method_call_general(params_cos, discount=discount, K=k, N=128, L=10, c=c, model_name=model_name)
        for N in range(6,300,10):
            option_price = COS_method_call_general(params_cos, discount=discount, K=k, N=N, L=10, c=c, model_name=model_name)
            print(f'N={N}')
            print(option_price)
            print('--'*10)
        # print((np.abs(option_price - ground_truth[i]) < 0.001))
        return (np.abs(option_price - 5.785155) < 0.001)

    if model_name == 'gaussian_jumps_1':
        return True

    if model_name == 'variance_gamma_1':
        ## params
        tau, k, S0, r, sigma, theta, vega = (1, 90, 100, 0.1, 0.12, -0.14, 0.2)


        states = np.array([sigma, theta, vega])
        params = [1,2,3,4]
        c = _calc_cumulants(tau, r, states, params, model_name)

        discount = np.exp(-r * tau)
        params_cos = [sigma, vega, theta, tau, S0, k, r, 'call']

        option_price = COS_method_call_general(params_cos, discount=discount, K=k, N=128, L=10, c=c, model_name=model_name)
        print(option_price)

    if model_name == 'cgmy_1':
        ground_truth = [19.812948843, 49.790905469]
        for i, Y in enumerate([0.5, 1.5]):
            C, G, M, Y, sigma, tau, S0, k, r = (1, 5, 5, Y, 0, 1, 100, 100, 0.1)

            states = np.array([C, G, M, Y, sigma])
            params = [1,2,3,4]
            c = _calc_cumulants(tau, r, states, params, model_name)

            discount = np.exp(-r * tau)
            params_cos = [C, G, M, Y, sigma, tau, S0, k, r, 'call']

            option_price = COS_method_call_general(params_cos, discount=discount, K=k, N=128, L=10, c=c, model_name=model_name)
            print(option_price)
            # print((np.abs(option_price - ground_truth[i]) < 0.001))

        return (np.abs(option_price - ground_truth[i]) < 0.001)

    if model_name == 'cgmy_sv_1':
        Y = 1.5
        # C, G, M, Y, sigma, tau, S0, k, r = (1, 5, 5, Y, 0, 1, 100, 100, 0.1)
        C, G, M, Yp, Yn, eta, kappa, eta, lambd, tau, S0, k, r = (1, 5, 5, Y, Y, 1, 0.0398, 0.5751, 1.5768, 1, 100, 100, 0.1)
        states = np.array([C, G, M, Y, 0])
        params = [1,2,3,4]
        c = _calc_cumulants(tau, r, states, params, 'cgmy_1')

        discount = np.exp(-r * tau)
        params_cos = [C, G, M, Yp, Yn, eta, kappa, eta, lambd, tau, S0, k, r, 'call']

        for N in range(6,430,5):
            option_price = COS_method_call_general(params_cos, discount=discount, K=k, N=N, L=10, c=c, model_name=model_name)
            print(option_price)
        # print((np.abs(option_price - ground_truth[i]) < 0.001))

        return True

    if model_name == 'cgmy4_sv_1':
        Y = 1.5
        # C, G, M, Y, sigma, tau, S0, k, r = (1, 5, 5, Y, 0, 1, 100, 100, 0.1)
        # C, G, M, Y, kappa, eta, lambd, tau, S0, k, r = (1, 5, 5, Y, 0.0398, 0.5751, 1.5768, 1, 100, 100, 0.1)
        C, G, M, Y, kappa, eta, lambd, tau, S0, k, r = (0.1225224, 0.01331071, 0.10705439, 0.46225137, 0.10221384, 0.71459981, 0.46536522, 1, 100, 100, 0.1)

        states = np.array([C, G, M, Y, 0])
        states = np.array([1, 5, 5, 1.5, 0])
        params = [1,2,3,4]
        c = _calc_cumulants(tau, r, states, params, 'cgmy_1')

        discount = np.exp(-r * tau)

        types = ['call', 'put']

        for type in types:
            params_cos = [C, G, M, Y, kappa, eta, lambd, tau, S0, k, r, type]

            # option_price = COS_method_call_general(params_cos, discount=discount, K=k, N=150, L=10, c=c, model_name=model_name)
            # print(option_price)
            for N in range(6,300,10):
                option_price = COS_method_call_general(params_cos, discount=discount, K=k, N=N, L=10, c=c, model_name=model_name)
                print(f'N={N}')
                print(option_price)
                print('--'*10)
            # print((np.abs(option_price - ground_truth[i]) < 0.001))
            print('=' * 30)
        return True

## Initialization

def path_generating(model_name):
    if model_name == 'heston_1_step':
        path_estimates = '../Data/Estimates_heston_1/'
        path_parameter = '../Data/Parameters_heston_1/'

    if model_name == 'heston_2_step':
        path_estimates = '../Data/Estimates_heston_2/'
        path_parameter = '../Data/Parameters_heston_2/'

    if model_name == 'gaussian_jumps_1':
        path_estimates = '../Data/Estimates_gaussian_jumps_1/'
        path_parameter = '../Data/Parameters_gaussian_jumps_1/'

    if model_name == 'variance_gamma_1':
        path_estimates = '../Data/Estimates_variance_gamma_1/'
        path_parameter = '../Data/Parameters_variance_gamma_1/'

    if model_name == 'cgmy_1':
        path_estimates = '../Data/Estimates_cgmy_1/'
        path_parameter = '../Data/Parameters_cgmy_1/'

    if model_name == 'cgmy_sv_1':
        path_estimates = '../Data/Estimates_cgmy_sv_1/'
        path_parameter = '../Data/Parameters_cgmy_sv_1/'

    if model_name == 'cgmy4_sv_1':
        path_estimates = '../Data/Estimates_cgmy4_sv_1/'
        path_parameter = '../Data/Parameters_cgmy4_sv_1/'

    return path_estimates, path_parameter

def setting_generating(model_name):
    if model_name == 'heston_1_step':
        params_opt = {'method':'Nelder-Mead',
                      'options': {'fatol': 1e-4},  # Stop when parameter or function changes are below 0.0001
                      'bounds':[[0, np.inf], [-np.inf, np.inf], [-1, 1], [0.001, np.inf], [0, np.inf]]
                  # each interval corresponds to the variable: u0, lambda_, rho, eta, u_bar
                 }
        return params_opt

    if model_name == 'heston_2_step':
        params_opt_states = {'method':'Nelder-Mead',
              'options': {'fatol': 1e-4},  # Stop when parameter or function changes are below 0.0001
              'tol':1e-4,
              'bounds':[[0, np.inf]]
              # each interval corresponds to the variable: u0
             }

        params_opt_params = {'method':'Nelder-Mead',
              'options': {'fatol': 1e-4},  # Stop when parameter or function changes are below 0.0001
              'tol':1e-4,
              'bounds':[[-np.inf, np.inf], [-1, 1], [0.001, np.inf], [0, np.inf]]
              # each interval corresponds to the variable: lambda_, rho, eta, u_bar
             }
        return [params_opt_states, params_opt_params]

    if model_name == 'gaussian_jumps_1':
        params_opt = {'method':'Nelder-Mead',
                      'options': {'fatol': 1e-4},  # Stop when parameter or function changes are below 0.0001
                      'bounds':[[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [0, np.inf]]
                  # each interval corresponds to the variable: V, c, mu, sigma
                 }
        return params_opt

    if model_name == 'variance_gamma_1':
        params_opt = {'method':'Nelder-Mead',
                      'options': {'fatol': 1e-4},  # Stop when parameter or function changes are below 0.0001
                      'bounds':[[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]
                  # each interval corresponds to the variable: sigma, vega, theta
                 }
        return params_opt

    if model_name == 'cgmy_1':
        params_opt = {'method':'Nelder-Mead',
                      'options': {'fatol': 1e-4},  # Stop when parameter or function changes are below 0.0001
                      'bounds':[[0, np.inf], [0, np.inf], [0, np.inf], [-np.inf, 1.90], [0, np.inf]]
                  # each interval corresponds to the variable:  C, G, M, Y, sigma
                 }
        return params_opt

    if model_name == 'cgmy_sv_1':
        params_opt = {'method':'Nelder-Mead',
                      'options': {'fatol': 1e-4},  # Stop when parameter or function changes are below 0.0001
                      'bounds':[[0, np.inf], [0, np.inf], [0, np.inf], [0.001, 1.90], [0.001, 1.90], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf]]
                  # each interval corresponds to the variable:  C, G, M, Yp, Yn, eta, kappa, eta, lambd
                 }
        return params_opt

    if model_name == 'cgmy4_sv_1':
        params_opt = {'method':'Nelder-Mead',
                      'options': {'fatol': 1e-4},  # Stop when parameter or function changes are below 0.0001
                      'bounds':[[0, np.inf], [0, np.inf], [0, np.inf], [0.001, 1.90], [0, np.inf], [0, np.inf], [0, np.inf]]
                  # each interval corresponds to the variable:  C, G, M, Yp, Yn, eta, kappa, eta, lambd
                 }
        return params_opt

def x0_generating(model_name):
    if model_name == 'heston_1_step':
        values_0 = np.array([0.01])
        # values_1 = np.linspace(0.01, 1, 3)
        values_1 = np.array([0.01])
        values_2 = np.array([0.0])
        values_3 = np.linspace(-0.5, 1.5, 3)
        # values_3 = np.array([0.01])
        values_4 = np.array([0.01])
        # u0, lambda_, rho, eta, u_bar

        mesh = np.meshgrid(values_0, values_1, values_2, values_3, values_4, indexing='ij')
        mesh_points = np.stack(mesh, axis=-1)

        return mesh_points

    if model_name == 'heston_2_step':
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

        return state_0, mesh_points_parameters

    if model_name == 'gaussian_jumps_1':
        values_0 = np.array([0.01])
        # values_0 = np.linspace(-1, 1, 3)
        # values_1 = np.array([0.01])
        values_1 = np.linspace(-1, 1, 3)
        # values_2 = np.array([0.01])
        values_2 = np.linspace(-1, 1, 3)
        values_3 = np.array([0.01])
        # values_3 = np.linspace(0.01, 1, 2)
        # V, c, mu, sigma

        mesh = np.meshgrid(values_0, values_1, values_2, values_3, indexing='ij')
        mesh_points = np.stack(mesh, axis=-1)

        return mesh_points

    if model_name == 'variance_gamma_1':
        values_0 = np.array([0.01])
        # values_0 = np.linspace(0.01, 5, 3)
        values_1 = np.array([0.01])
        # values_1 = np.linspace(0.01, 5, 3)
        # values_2 = np.array([0.00])
        values_2 = np.linspace(-1, 1, 3)
        # sigma, vega, theta

        mesh = np.meshgrid(values_0, values_1, values_2, indexing='ij')
        mesh_points = np.stack(mesh, axis=-1)

        return mesh_points

    if model_name == 'cgmy_1':
        # values_0 = np.linspace(0.01, 1, 2)
        values_0 = np.array([1])
        # values_1 = np.linspace(0.01, 1, 2)
        values_1 = np.array([1])
        # values_2 = np.linspace(0.01, 1, 2)
        values_2 = np.array([1])
        values_3 = np.linspace(-0.6, 0.6, 2)
        # values_3 = np.array([0.01])
        # values_4 = np.linspace(0.1, 0.5, 2)
        values_4 = np.array([0.1])
        # C, G, M, Y, sigma

        mesh = np.meshgrid(values_0, values_1, values_2, values_3, values_4, indexing='ij')
        mesh_points = np.stack(mesh, axis=-1)

        return mesh_points

    if model_name == 'cgmy_sv_1':
        # values_0 = np.linspace(0.01, 1, 2)
        values_0 = np.array([0.1])
        values_1 = np.linspace(0.01, 1, 2)
        values_2 = np.linspace(0.01, 1, 2)
        # values_3 = np.linspace(0.1, 0.5, 2)
        values_3 = np.array([0.5])
        values_4 = np.array([0.5])
        # values_4 = np.linspace(0.1, 0.5, 2)
        # values_4 = np.array([0.1])
        values_5 = np.linspace(0.1, 0.5, 2)
        # values_6 = np.linspace(0.1, 0.5, 2)
        values_6 = np.array([0.1])
        values_7 = np.array([0.5])
        values_8 = np.array([0.5])
        # values_7 = np.linspace(0.1, 0.5, 2)
        # values_8 = np.linspace(0.1, 0.5, 2)
        # C, G, M, Yp, Yn, eta, kappa, eta, lambd = states


        mesh = np.meshgrid(values_0, values_1, values_2, values_3, values_4,
                           values_5 , values_6 , values_7 , values_8, indexing='ij')
        mesh_points = np.stack(mesh, axis=-1)

        return mesh_points

    if model_name == 'cgmy4_sv_1':
        # values_0 = np.linspace(0.01, 2, 2)
        values_0 = np.array([1])
        values_1 = np.array([5])
        values_2 = np.array([5])
        values_3 = np.linspace(0.1, 1.5, 2)
        # values_3 = np.array([0.5])
        # values_6 = np.linspace(0.1, 0.5, 2)
        values_4 = np.array([0.1])
        values_5 = np.array([0.5])
        values_6 = np.array([0.5])
        # values_7 = np.linspace(0.1, 0.5, 2)
        # values_8 = np.linspace(0.1, 0.5, 2)
        # C, G, M, Y, kappa, eta, lambd = states

        mesh = np.meshgrid(values_0, values_1, values_2, values_3, values_4,
                           values_5 , values_6 , indexing='ij')
        mesh_points = np.stack(mesh, axis=-1)

        return mesh_points

## params_cos

def generate_params_cos_1(state, r, tau, S0, k, type, model_name):
    if model_name == 'heston_1_step':
        params_cos = [state[0], state[1], state[2], state[3], r, state[4], tau, S0, k, type]
    if model_name == 'gaussian_jumps_1':
        #  V, c, mu, sigma, tau, S0, k, r
        params_cos = [*state , tau, S0, k, r, type]
    if model_name == 'variance_gamma_1':
        # sigma, vega, theta, tau, S0, k, r
        params_cos = [*state, tau, S0, k, r, type]
    if model_name == 'cgmy_1':
        # C, G, M, Y, sigma, tau, S0, k, r
        params_cos = [*state, tau, S0, k, r, type]
    if model_name == 'cgmy_sv_1':
        # C, G, M, Yp, Yn, eta, kappa, eta, lambd
        params_cos = [*state, tau, S0, k, r, type]
    if model_name == 'cgmy4_sv_1':
        # C, G, M, Y, kappa, eta, lambd
        params_cos = [*state, tau, S0, k, r, type]
    return params_cos

## Cumulant functions

# @jit
# @jit(nopython=True)
def _calc_cumulants(tau, r, states, params, model_name):
    if model_name == 'heston_1_step':
        u0, lambda_, rho, eta, u_bar = states
        c = np.zeros(6)
        c[0] = cumulant_0_hes(r, tau, lambda_, u_bar, u0)
        c[1] = cumulant_1_hes(r, tau, lambda_, u_bar, u0, eta, rho)
        return c
    if model_name == 'heston_2_step':
        u0 = states[0]
        lambda_, rho, eta, u_bar = params
        c = np.zeros(6)
        c[0] = cumulant_0_hes(r, tau, lambda_, u_bar, u0)
        c[1] = cumulant_1_hes(r, tau, lambda_, u_bar, u0, eta, rho)
        return c
    if model_name == 'gaussian_jumps_1':
        V, ct, mu, sigma = states
        c = np.zeros(6)
        c[0] = cumulant_0_gaussian_jumps(V, ct, mu, sigma, tau)
        c[1] = cumulant_1_gaussian_jumps(V, ct, mu, sigma, tau)
        return c
    if model_name == 'variance_gamma_1':
        sigma, vega, theta = states
        c = np.zeros(6)
        c[0] = cumulant_0_variance_gamma(sigma, vega, theta, tau, r)
        c[1] = cumulant_1_variance_gamma(sigma, vega, theta, tau, r)
        return c
    if model_name == 'cgmy_1':
        C, G, M, Y, sigma = states
        c = np.zeros(6)
        c[0] = cumulant_0_cgmy(C, G, M, Y, sigma, tau, r)
        c[1] = cumulant_1_cgmy(C, G, M, Y, sigma, tau, r)
        return c
    if model_name == 'cgmy_sv_1':
        # C, G, M, Yp, Yn, eta, kappa, eta, lambd, tau, S0, k, r = params_cos
        C, G, M, Yp, Yn, eta, kappa, eta, lambd = states
        c = np.zeros(6)
        c[0] = cumulant_0_cgmy(C, G, M, Yp, eta, tau, r)
        c[1] = cumulant_1_cgmy(C, G, M, Yp, eta, tau, r)
        return c
    if model_name == 'cgmy4_sv_1':
        C, G, M, Y, kappa, eta, lambd = states
        c = np.zeros(6)
        c[0] = cumulant_0_cgmy(C, G, M, Y, 1, tau, r)
        c[1] = cumulant_1_cgmy(C, G, M, Y, 1, tau, r)
        return c

@jit
def cumulant_0_hes(r, tau, lambda_, u_bar, u0):
    """Compute the first cumulant (c1) based on the given formula."""
    return (r * tau + (1 - np.exp(-lambda_ * tau)) * (u_bar - u0) / (2 * lambda_) - 0.5 * u_bar * tau)
@jit
def cumulant_1_hes(r, tau, lambda_, u_bar, u0, eta, rho):
    """Compute the second cumulant (c2) based on the given formula."""
    term1 = (eta * tau * lambda_ * np.exp(-lambda_ * tau) * (u0 - u_bar) * (8 * lambda_ * rho - 4 * eta)) / (8 * lambda_**3)
    term2 = (lambda_ * rho * (1 - np.exp(-lambda_ * tau)) * (16 * u_bar - 8 * u0)) / (8 * lambda_**3)
    term3 = (2 * u_bar * lambda_ * tau * (-4 * lambda_ * rho + eta**2 + 4 * lambda_**2)) / (8 * lambda_**3)
    term4 = (eta**2 * ((u_bar - 2 * u0) * np.exp(-2 * lambda_ * tau) + u_bar * (6 * np.exp(-lambda_ * tau) - 7) + 2 * u0)) / (8 * lambda_**3)
    term5 = (8 * lambda_**2 * (u0 - u_bar) * (1 - np.exp(-lambda_ * tau))) / (8 * lambda_**3)

    return term1 + term2 + term3 + term4 + term5

@jit
def cumulant_0_gaussian_jumps(V, c, mu, sigma, tau):
    return tau * (-0.5 * V + c * (mu - (np.exp(mu + 0.5 * sigma**2) - 1)))

@jit
def cumulant_1_gaussian_jumps(V, c, mu, sigma, tau):
    return tau * (V + c * sigma**2 * np.exp(mu + 0.5 * sigma**2))

@jit
def cumulant_0_variance_gamma(sigma, vega, theta, tau, r):
    return (r + theta) * tau
@jit
def cumulant_1_variance_gamma(sigma, vega, theta, tau, r):
    return (sigma**2 + vega * theta**2) * tau

# @jit
# @jit(nopython=True)
def cumulant_0_cgmy(C, G, M, Y, sigma, tau, r):
    return r * tau + C * tau * gamma(1 - Y) * (M**(Y-1) - G**(Y-1))
# @jit
# @jit(nopython=True)
def cumulant_1_cgmy(C, G, M, Y, sigma, tau, r):
    return sigma**2 *tau + C * tau * gamma(2 - Y) * (M**(Y-2) + G**(Y-2))

@jit
def cf_clock(u, t, y0, kappa, eta, lambd):
    gamma = np.sqrt(kappa**2 - 2 * lambd**2 * 1j * u)

    # Compute B(t, u)
    numerator_B = 2j * u
    denominator_B = kappa + gamma * (1 / np.tanh(gamma * t / 2))
    B = numerator_B / denominator_B

    # Compute A(t, u)
    exp_term = np.exp((kappa**2 * eta * t) / (lambd**2))
    cosh_term = np.cosh(gamma * t / 2)
    sinh_term = np.sinh(gamma * t / 2)
    denom = cosh_term + (kappa / gamma) * sinh_term
    power = (2 * kappa * eta) / (lambd**2)
    A = exp_term / (denom**power)

    # Final characteristic function
    phi = A * np.exp(B * y0)
    return phi

def compensator(omega, params_cos, model_name):
    if model_name == 'variance_gamma_1':
        sigma, nu, theta, tau, S0, k, r, type = params_cos
        x = np.log(S0/k)
        psi = lambda u: np.log(phi_vg(u, sigma, nu, theta, tau, r)) / tau
    if model_name == 'gaussian_jumps_1':
        V, c, mu, sigma, tau, S0, k, r, type = params_cos
        x = np.log(S0/k)
        psi = lambda u: gaussian_jumps_alpha(u, V, c, mu, sigma, tau)/tau
    if model_name == 'cgmy_1':
        C, G, M, Y, sigma, tau, S0, k, r, type = params_cos
        x = np.log(S0/k)
        if np.abs(Y) > 0.001:
            psi = lambda u: C * gamma(-Y) * ( (M - 1j * u)**Y - M**Y + (G + 1j * u)**Y - G**Y ) - sigma**2 * u**2 * 0.5
        else:
            psi = lambda u: - sigma**2 * u**2 * 0.5
    if model_name == 'cgmy_sv_1':
        C, G, M, Yp, Yn, eta, kappa, eta, lambd, tau, S0, k, r, type = params_cos
        x = np.log(S0/k)
        psi_cgmy_6 = lambda u: 1 * (gamma(-Yp) * ((M - 1j * u)**Yp - M**Yp) + eta * gamma(-Yn) * ((G + 1j * u)**Yn - G**Yn)) ## Cp = 1 as it is in y(0), page 354 carr 2003.
        psi = lambda u: np.log(cf_clock(-1j * psi_cgmy_6(u), tau, C, kappa, eta, lambd)) / tau

    if model_name == 'cgmy4_sv_1':
        C, G, M, Y, kappa, eta, lambd, tau, S0, k, r, type = params_cos
        x = np.log(S0/k)
        psi_cgmy_6 = lambda u: 1 * (gamma(-Y) * ((M - 1j * u)**Y - M**Y) + gamma(-Y) * ((G + 1j * u)**Y - G**Y))
        psi = lambda u: np.log(cf_clock(-1j * psi_cgmy_6(u), tau, C, kappa, eta, lambd)) / tau

    return np.exp(1j * omega * x + 1j * omega * tau * (r - psi(-1j)) + tau * psi(omega))

def COS_method_call_general_slow(params_cos, discount, K, model_name, N=None, L=10, c=None):
    # if 'c' not in locals():
    #     c = np.zeros(6)
    #     f_derivative = log(phi_sp)
    #     for i in range(6):
    #         f_derivative = f_derivative.diff(t)
    #         c[i] = f_derivative.subs(t, 0) * 1j**-(i+1)
    #         pprint(simplify(f_derivative))
    if 'heston' in model_name:
        a = c[0] - 12 * np.sqrt(np.abs(c[1]))
        b = c[0] + 12 * np.sqrt(np.abs(c[1]))
    else:
        a = c[0] - L * np.sqrt(c[1])
        b = c[0] + L * np.sqrt(c[1])

    return option_valuation_general_slow(params_cos=params_cos,
                                     a=a,
                                     b=b,
                                     N=N,
                                     discount=discount,
                                     K=K,
                               option=params_cos[-1],
                               model_name=model_name)

def option_valuation_general_slow(params_cos,
                          a,
                          b,
                          N,
                          discount,
                          K,
                        option,
                        model_name):
    assert option in ['call', 'put']
    if model_name == 'heston_1_step' or model_name == 'heston_2_step':
        def phi(omega):
            return hes_phi(omega, *params_cos[:-1])
            # def hes_phi(omega, u0, lambda_, rho, eta, mu, u_bar, tau, S0, K):
    if model_name == 'gaussian_jumps_1':
        # def phi(omega):
        #     return gaussian_jumps_phi(omega, *params_cos)
        def phi(omega):
            return compensator(omega, params_cos, model_name)
                #  V, c, mu, sigma, tau
    if model_name == 'variance_gamma_1':
        # def phi(omega):
        #     return variance_gamma_phi_compensated(omega, *params_cos)
        #         # sigma, vega, theta, tau, S0, k, r
        def phi(omega):
            return compensator(omega, params_cos, model_name)

    if model_name == 'cgmy_1':
        def phi(omega):
            # C, G, M, Y, sigma, tau, S0, k, r = params_cos
            return compensator(omega, params_cos, model_name)

    if model_name == 'cgmy_sv_1':
        def phi(omega):
            # C, G, M, Yp, Yn, eta, kappa, eta, lambd, tau, S0, k, r = params_cos
            return compensator(omega, params_cos, model_name)

    if model_name == 'cgmy4_sv_1':
        def phi(omega):
            # C, G, M, Y, kappa, eta, lambd, tau, S0, k, r = params_cos
            return compensator(omega, params_cos, model_name)


    def exponential_component(factor):
        return np.exp(-1j * factor * a)
    # exponential_component = lambda factor: np.exp(-1j * factor * a)
    def comb(factor):
        return (phi(factor) * exponential_component(factor)).real
    # comb = lambda factor: (phi(factor) * exponential_component(factor)).real

    # Vk
    if option == 'call':
        Vk = lambda k: 2/(b-a) * K * (chi_k(c=0, d=b, k=k, a=a, b=b) - psi_k(c=0, d=b, k=k, a=a, b=b))
    if option == 'put':
        Vk = lambda k: 2/(b-a) * K * (-chi_k(c=a, d=0, k=k, a=a, b=b) + psi_k(c=a, d=0, k=k, a=a, b=b))

    # estimation

    vcomb = np.vectorize(comb)
    factors = np.arange(N) * np.pi / (b - a)
    combs = vcomb(factors)            # Vectorized comb
    Vks =  np.array([Vk(k) for k in range(N)], dtype=np.float64)           # Example vectorized Vk
    Vks[0] *= 0.5                    # Adjust first element
    v = np.inner(Vks,combs)

    # v = 0
    # for k in range(N):
    #     factor = k * np.pi / (b-a)
    #     # print(phi(factor))
    #     if k == 0:
    #         v += comb(factor) * Vk(k) * 0.5
    #     else:
    #         v += comb(factor) * Vk(k)
    v = discount * v
    return v
