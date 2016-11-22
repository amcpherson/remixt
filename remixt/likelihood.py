import numpy as np
import pandas as pd
import scipy
import scipy.misc
from scipy.special import gammaln
from scipy.special import betaln
from scipy.special import digamma

import remixt.utils


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

    return x3


def calculate_mean_cn(h, x, l):
    """ Calculate the mean raw copy number.

    Args:
        h (numpy.array): haploid read depths, h[0] for normal
        x (numpy.array): major, minor, and total read counts
        l (numpy.array): segment lengths

    Returns:
        numpy.array: N * L dim array, per segment per allele mean copy number

    """

    phi = remixt.likelihood.estimate_phi(x)

    depth = x[:,0:2] / (phi * l)[:, np.newaxis]

    mean_cn = (depth - h[0]) / h[1:].sum()

    return mean_cn



def calculate_mean_total_cn(h, x, l):
    """ Calculate the mean raw copy number.

    Args:
        h (numpy.array): haploid read depths, h[0] for normal
        x (numpy.array): major, minor, and total read counts
        l (numpy.array): segment lengths

    Returns:
        numpy.array: N * L dim array, per segment per allele mean copy number

    """

    depth = x[:, 2] / l

    mean_cn = (depth - h[0]) / h[1:].sum()

    return mean_cn
