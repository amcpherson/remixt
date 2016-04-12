import itertools
import numpy as np
import pandas as pd

import sklearn
import sklearn.cluster

import remixt.utils
import remixt.likelihood


def calculate_depth(experiment):
    """ Calculate the minor, major, total depth

    Args:
        experiment (remixt.Experiment): experiment object

    Returns:
        pandas.DataFrame: read depth table with columns, 'major', 'minor', 'total', 'length'

    """
    x = experiment.x.copy()
    l = experiment.l.copy()

    phi = remixt.likelihood.estimate_phi(x)
    p = remixt.likelihood.proportion_measureable_matrix(phi)

    is_filtered = (l > 0) & np.all(p > 0, axis=1)
    x = x[is_filtered,:]
    l = l[is_filtered]
    p = p[is_filtered,:]

    rd = ((x.T / p.T) / l.T).T
    rd.sort(axis=1)

    rd = pd.DataFrame(rd, columns=['minor', 'major', 'total'])
    rd['length'] = l

    return rd


def calculate_minor_modes(read_depth):
    """ Calculate modes in distribution of minor allele read depths

    Args:
        read_depth (pandas.DataFrame): read depth table

    Returns:
        numpy.array: read depth modes

    """

    # Remove extreme values from the upper end of the distribution of minor depths
    amp_rd = np.percentile(read_depth['minor'], 95)
    read_depth = read_depth[read_depth['minor'] < amp_rd]

    # Cluster read depths using kmeans
    rd_samples = remixt.utils.weighted_resample(read_depth['minor'].values, read_depth['length'].values)
    kmm = sklearn.cluster.KMeans(n_clusters=5)
    kmm.fit(rd_samples.reshape((rd_samples.size, 1)))
    means = kmm.cluster_centers_[:,0]

    return means


def calculate_candidate_h_monoclonal(minor_modes):
    """ Calculate possible haploid tumour depth from modes of minor allele depth.

    Args:
        minor_modes (list): minor read depth modes

    Returns:
        list: haploid depths
    """

    h_normal = minor_modes.min()
    
    h_candidates = list()

    for h_tumour in minor_modes:
        if h_tumour <= h_normal:
            continue

        h_tumour -= h_normal

        # Consider the possibility that the first minor mode
        # is composed of segments with 2 minor copies
        for scale in (1., 0.5):
            h_mono = np.array([h_normal, h_tumour * scale])
            h_candidates.append(h_mono)

    return h_candidates


def estimate_ploidy(h, experiment):
    """ Estimate ploidy for a candidate haploid depth.

    Args:
        h (numpy.array): haploid normal and tumour clones read depths
        experiment (remixt.Experiment): experiment object

    Returns:
        float: ploidy estimate

    """

    mean_cn = remixt.likelihood.calculate_mean_cn(h, experiment.x, experiment.l)
    return (mean_cn.T * experiment.l).sum() / experiment.l.sum()

