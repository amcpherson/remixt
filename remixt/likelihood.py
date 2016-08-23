import numpy as np
import pandas as pd
import scipy
import scipy.misc
from scipy.special import gammaln
from scipy.special import betaln
from scipy.special import digamma

import remixt.utils



class LikelihoodError(ValueError):
    def __init__(self, message, **variables):
        """ Error calculating a likelihood.

        Args:
            message (str): message detailing error

        KwArgs:
            **variables: variables to be printed

        """

        for name, value in variables.iteritems():
            message += '\n{0}={1}'.format(name, value)

        ValueError.__init__(self, message)


allele_measurement_matrix = np.array([[1, 0, 1], [0, 1, 1]])


def estimate_phi(x):
    """ Estimate proportion of genotypable reads.

    Args:
        x (numpy.array): major, minor, and total read counts

    Returns:
        numpy.array: estimate of proportion of genotypable reads.

    """

    phi = x[:,0:2].sum(axis=1).astype(float) / (x[:,2].astype(float) + 1.0)

    return phi


def proportion_measureable_matrix(phi):
    """ Proportion reads measureable matrix.

    Args:
        phi (numpy.array): estimate of proportion of genotypable reads.    

    Returns:
        numpy.array: N * K dim array, segment to measurement transform

    """

    return np.vstack([phi, phi, np.ones(phi.shape)]).T


def expected_read_count(l, cn, h, phi):
    """ Calculate expected major, minor and total read counts.

    Args:
        l (numpy.array): segment lengths
        cn (numpy.array): copy number state
        h (numpy.array): haploid read depths
        phi (numpy.array): estimate of proportion of genotypable reads

    Returns:
        numpy.array: expected read depths
    """

    p = proportion_measureable_matrix(phi)
    q = allele_measurement_matrix

    gamma = np.sum(cn * np.vstack([h, h]).T, axis=-2)

    x1 = np.dot(q.T, gamma.T).T

    x2 = x1 * p

    x3 = (x2.T * l.T).T

    x3 += 1e-16

    for n, ell in zip(*np.where(x3 <= 0)):
        raise ProbabilityError('mu <= 0', n=n, cn=cn[n], l=l[n], h=h, p=p[n], mu=x3[n])

    for n, ell in zip(*np.where(np.isnan(x3))):
        raise ProbabilityError('mu is nan', n=n, cn=cn[n], l=l[n], h=h, p=p[n], mu=x3[n])

    return x3

