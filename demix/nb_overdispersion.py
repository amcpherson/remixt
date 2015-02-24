


import numpy as np
from scipy.stats import nbinom
from scipy.optimize import minimize
from scipy.special import digamma


def nll_nbinom(param_t, x, l):
    
    gamma_t = param_t[:-1]
    r_t = param_t[-1]
    
    mu = l * np.array([gamma_t]).T
    nb_p = mu / (r_t + mu)

    ll = nbinom.logpmf(x, r_t, 1-nb_p)

    nll = -ll.sum()

    assert not np.isnan(nll)

    return nll


def nll_nbinom_der(param_t, x, l):
    
    gamma_t = param_t[:-1]
    r_t = param_t[-1]
    
    gamma_t = gamma_t.reshape(gamma_t.shape + (1,))
    
    der_gamma = np.sum(- r_t * l / (r_t + l * gamma_t) + x / gamma_t - x * l / (r_t + l * gamma_t), axis=1)
    der_r = np.sum(digamma(x + r_t) - digamma(r_t) + np.log(r_t) + 1 - np.log(r_t + l * gamma_t) - (r_t + x) / (r_t + l * gamma_t))
    
    der = -np.concatenate([der_gamma, [der_r]])

    assert not np.any(np.isnan(der))

    return der


def infer_disperion(x, l):
    
    x = x.reshape((len(x) / 2, 2))
    l = l.reshape((len(l) / 2, 2))

    gamma0 = (x / l).mean(axis=1)
    r0 = 10.

    x = x[~np.isnan(gamma0)]
    l = l[~np.isnan(gamma0)]
    gamma0 = gamma0[~np.isnan(gamma0)]
    
    param0 = np.concatenate([gamma0, [r0]])

    result = minimize(nll_nbinom, param0, method='L-BFGS-B', jac=nll_nbinom_der, args=(x, l), bounds=((1e-6, 1e14),)*len(param0))

    if not result.success:
        raise Exception(result.message)

    return result.x[-1]


