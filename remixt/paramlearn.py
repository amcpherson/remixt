import numpy as np
import scipy.optimize


def _sum_adjacent(x):
    return x.reshape((len(x) / 2, 2)).sum(axis=1)


def nll_negbin(param, negbin, x, l):
    """ Negative log likelihood assuming adjacent pairs have equal
    expected read depth
    """
    negbin.r = param[-1]
    g = np.repeat(param[:-1], 2)

    mu = l * g

    return -negbin.log_likelihood(x, mu).sum()


def nll_negbin_partial_param(param, negbin, x, l):
    """ Partial derivative of the negative log likelihood assuming
    adjacent pairs have equal expected read depth
    """

    negbin.r = param[-1]
    g = np.repeat(param[:-1], 2)

    mu = l * g

    ll_partial_mu = negbin.log_likelihood_partial_mu(x, mu)
    ll_partial_g = _sum_adjacent(ll_partial_mu * l)
    ll_partial_r = negbin.log_likelihood_partial_r(x, mu).sum()

    return -np.concatenate([ll_partial_g, [ll_partial_r]])


import statsmodels.tools.numdiff
def assert_grad_correct(func, grad, x0, *args, **kwargs):
    """ Assert correct gradiant compared to finite difference approximation
    """

    analytic_fprime = grad(x0, *args)
    approx_fprime = statsmodels.tools.numdiff.approx_fprime_cs(x0, func, args=args)

    np.testing.assert_almost_equal(analytic_fprime, approx_fprime, 5)


def learn_negbin_r_adjacent(negbin, x, l, min_length=10000., perc=90., bias=2.0):
    """ Learn the negative binomial dispersion parameter with the
    adjacent data points similarity assumption.

    Args:
        negbin (NegBinDistribution): negative binomial distribution
        x (numpy.array): observed read counts
        l (numpy.array): lengths of segments

    KwArgs:
        min_length: minimum length of segments for inclusion in optimization
        perc: filter outliers by specified percentile wrt likelihood
        bias: maximum likelihood estimation bias

    """

    # Remove segments with problematic lengths
    x = x[~np.isinf(l) & (l > min_length)]
    l = l[~np.isinf(l) & (l > min_length)]

    # Even number of data points required
    resize = int(len(x) / 2) * 2
    x = x[:resize]
    l = l[:resize]

    # Intial parameter estimate
    g0 = _sum_adjacent(x / l) / 2.
    r0 = 100.

    # Initial estimate of negative log likelihood per pair
    negbin.r = r0
    adj_nll = -negbin.log_likelihood(x, l * np.repeat(g0, 2))
    adj_nll = _sum_adjacent(adj_nll)

    # Outliers as percentile of log likelihood
    outliers = (adj_nll > np.percentile(adj_nll, perc))
    outliers = np.repeat(outliers, 2)

    # Remove outliers
    x = x[~outliers]
    l = l[~outliers]

    # Redo intial parameter estimate
    g0 = _sum_adjacent(x / l) / 2.
    r0 = 100.
    param0 = np.concatenate([g0, [r0]])

    result = scipy.optimize.minimize(nll_negbin, param0,
        jac=nll_negbin_partial_param,
        method='L-BFGS-B', 
        args=(negbin, x, l),
        bounds=((1e-6, 1e14),)*len(param0))

    if not result.success:
        raise Exception(result.message)

    r = result.x[-1] / bias

    negbin.r = r

    return r

