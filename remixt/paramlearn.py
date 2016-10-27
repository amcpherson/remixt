import numpy as np
import scipy.optimize


def _sum_adjacent(x):
    return x.reshape((len(x) / 2, 2)).sum(axis=1)


def nll_negbin_g(param, negbin, x, l):
    """ Negative log likelihood assuming adjacent pairs have equal
    expected read depth
    """
    g = np.repeat(param, 2)

    mu = l * g

    return -negbin.log_likelihood(x, mu).sum()


def nll_negbin_g_partial(param, negbin, x, l):
    """ Partial derivative of the negative log likelihood assuming
    adjacent pairs have equal expected read depth
    """
    g = np.repeat(param, 2)

    mu = l * g

    ll_partial_mu = negbin.log_likelihood_partial_mu(x, mu)
    ll_partial_g = _sum_adjacent(ll_partial_mu * l)

    return -ll_partial_g


def nll_negbin_r(param, negbin, x, l, g):
    """ Negative log likelihood optimized for g
    """
    r = param
    
    if r <= 0:
        return np.inf
    
    negbin.r = r
    
    result = scipy.optimize.minimize(
        nll_negbin_g, g,
        jac=nll_negbin_g_partial,
        method='L-BFGS-B', 
        args=(negbin, x, l),
        bounds=((1e-6, 1e14),)*len(g))
        
    g[:] = result.x

    assert not np.isnan(result.fun)
    
    return result.fun


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
    g = _sum_adjacent(x / l) / 2.
    
    result = scipy.optimize.minimize_scalar(
        nll_negbin_r,
        bounds=[1., 1e4],
        args=(negbin, x, l, g))

    if not result.success:
        raise Exception(result.message)

    r = result.x / bias

    negbin.r = r

    return r


def nll_betabin(param, betabin, k, n):
    """ Negative log likelihood assuming adjacent pairs have equal
    expected read depth
    """
    betabin.M = param[-1]
    p = np.repeat(param[:-1], 2)

    return -betabin.log_likelihood(k, n, p).sum()


def nll_betabin_partial_param(param, betabin, k, n):
    """ Partial derivative of the negative log likelihood assuming
    adjacent pairs have equal expected read depth
    """

    betabin.M = param[-1]
    p = np.repeat(param[:-1], 2)

    ll_partial_p = _sum_adjacent(betabin.log_likelihood_partial_p(k, n, p))
    ll_partial_M = betabin.log_likelihood_partial_M(k, n, p).sum()

    return -np.concatenate([ll_partial_p, [ll_partial_M]])


def learn_betabin_M_adjacent(betabin, k, n, min_readcount=1000., perc=90., bias=2.0):
    """ Learn the beta binomial dispersion parameter with the
    adjacent data points similarity assumption.

    Args:
        betabin (BetaBinDistribution): beta binomial distribution
        k (numpy.array): observed minor allelic read counts
        n (numpy.array): observed total allelic read counts

    KwArgs:
        min_readcount: minimum total reads for inclusion in optimization
        perc: filter outliers by specified percentile wrt likelihood
        bias: maximum likelihood estimation bias

    """

    # Remove segments with problematic readcounts
    k = k[(n >= min_readcount)]
    n = n[(n >= min_readcount)]

    # Even number of data points required
    resize = int(k.shape[0] / 2) * 2
    k = k[:resize]
    n = n[:resize]

    # Intial parameter estimate
    p0 = _sum_adjacent(k.astype(float) / n.astype(float)) / 2.
    M0 = 100.

    # Initial estimate of negative log likelihood per pair
    betabin.M = M0
    adj_nll = -betabin.log_likelihood(k, n, np.repeat(p0, 2))
    adj_nll = _sum_adjacent(adj_nll)

    # Outliers as percentile of log likelihood
    outliers = (adj_nll > np.percentile(adj_nll, perc))
    outliers = np.repeat(outliers, 2)

    # Remove outliers
    k = k[~outliers]
    n = n[~outliers]

    # Redo intial parameter estimate
    p0 = _sum_adjacent(k.astype(float) / n.astype(float)) / 2.
    M0 = 100.
    param0 = np.concatenate([p0, [M0]])

    bounds = ((1e-6, 1.-1e-6),)*len(p0) + ((1e-6, 1e14),)

    result = scipy.optimize.minimize(nll_betabin, param0,
        jac=nll_betabin_partial_param,
        method='L-BFGS-B', 
        args=(betabin, k, n),
        bounds=bounds)

    if not result.success:
        raise Exception(result.message)

    M = result.x[-1] / bias

    betabin.M = M

    return M
