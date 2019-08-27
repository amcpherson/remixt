import numpy as np
import scipy.misc


def geometric_deviation_pmf(base, geom_p, dev_max):
    """ PMF of geometrically distributed deviation from given base counts.
    
    Args:
        base (numpy.array): base counts to deviate from
        geom_p (float): probability for geometric distribution
        dev_max (int): maximum deviation
    
    Returns:
        tuple of numpy.array: counts, log_probs
    """

    base = base.flatten()
    
    num_dev = 2 * dev_max + 1

    dev = np.ones((num_dev,))
    dev = np.cumsum(dev) - dev_max - 1.

    log_probs = np.abs(dev) * np.log(1-geom_p) + np.log(geom_p)

    counts = np.tile(base, num_dev).reshape((num_dev, len(base))).T
    log_probs = np.tile(log_probs, len(base)).reshape((len(base), num_dev))

    counts = counts + dev

    # Exclude nonzero counts
    log_probs[counts < 0] = -np.inf

    # Normalize
    log_norm = scipy.misc.logsumexp(log_probs, axis=1).T
    log_probs = (log_probs.T - log_norm.T).T
    
    return counts, log_probs
    
    
def random_geometric_deviation(base, geom_p, dev_max):
    """ Generate a random geometrically distributed deviation from the given base counts.
    
    Args:
        base (numpy.array): base counts to deviate from
        geom_p (float): probability for geometric distribution
        dev_max (int): maximum deviation
    
    Returns:
        numpy.array: deviated counts
    """
    
    base = base.flatten()
    
    counts, log_probs = geometric_deviation_pmf(base, geom_p, dev_max)
    
    probs = np.exp(log_probs)
    
    # Cumulative distribution
    cumulative = np.cumsum(probs, axis=1)

    # Uniformly distributed samples
    samples = np.random.uniform(size=len(base))
    
    # Sampling from a discrete distribution trick
    selected = (cumulative.T > samples).T.cumsum(axis=1) == 1
    
    return counts[selected]
    

def generate_cn(N, M, ploidy, base_p, clone_p, dev_max):
    """ Generate a random copy number profile.
    
    Args:
        N (int): number of segments
        M (int): number of populations including normal
        ploidy (int): base haploid copy number of tumour populations
        base_p (float): probability for geometric random sampled base copy number
        clone_p (float): probability for geometric random sampled clone specific copy number deviation
        dev_max (int): maximum deviation
    
    Returns:
        numpy.array: Copy number matrix of tumour populations
    """
    
    cn = list()

    cn.append(np.ones((N, 2)))

    cn_base = np.ones((N, 2)) * ploidy
    cn_base = random_geometric_deviation(cn_base, base_p, dev_max).reshape((N, 2))

    cn.append(cn_base)

    for tidx in range(M-2):
        clone_cn = random_geometric_deviation(cn_base, clone_p, dev_max).reshape((N, 2))
        cn.append(clone_cn)

    cn = np.array([a.T for a in cn]).T

    cn = cn.swapaxes(-1, -2)

    return cn


