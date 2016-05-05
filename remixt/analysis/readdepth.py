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

    data = experiment.create_segment_table()

    phi = remixt.likelihood.estimate_phi(experiment.x)
    p = remixt.likelihood.proportion_measureable_matrix(phi)

    # Filter segments for which read depth calculation will be nan/inf
    data = data[(data['length'] > 0) & np.all(p > 0, axis=1)]

    data.rename(columns={
        'major_depth': 'major',
        'minor_depth': 'minor',
        'total_depth': 'total',
    }, inplace=True)

    data = data[[
        'chromosome',
        'start',
        'end',
        'length',
        'major',
        'minor',
        'total',
    ]]

    return data


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
    mean_cn[np.isnan(mean_cn) | np.isinf(mean_cn)] = 0.
    return (mean_cn * experiment.l[:, np.newaxis]).sum() / experiment.l.sum()

