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

    data = remixt.analysis.experiment.create_segment_table(experiment)
    
    data['segment_length'] = data['end'] - data['start'] + 1
    data['length_ratio'] = data['length'] / data['segment_length']
    data['allele_readcount'] = data['minor_readcount'] + data['major_readcount']

    data['high_quality'] = (
        (data['length'] > np.percentile(data['length'].values, 10)) &
        (data['allele_readcount'] > np.percentile(data['allele_readcount'].values, 10)) &
        (data['length_ratio'] > np.percentile(data['length_ratio'].values, 10)))

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
        'high_quality',
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

    # Filter insigificant clusters
    cluster_idx = kmm.predict(rd_samples.reshape((rd_samples.size, 1)))
    cluster_counts = np.bincount(cluster_idx)
    cluster_prop = cluster_counts.astype(float) / cluster_counts.sum()
    means = means[cluster_prop >= 0.01]

    return means


def calculate_candidate_h_monoclonal(minor_modes, h_normal=None, h_tumour=None):
    """ Calculate possible haploid tumour depth from modes of minor allele depth.

    Args:
        minor_modes (list): minor read depth modes

    KwArgs:
        h_normal (float): fix haploid normal at this value
        h_tumour (float): fix haploid tumour at this value

    Returns:
        list: haploid depths
    """

    if h_normal is None:
        h_normal = minor_modes.min()
        
    if h_tumour is not None:
        return np.array([[h_normal, h_tumour]])
    
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

    read_depth = calculate_depth(experiment)

    read_depth['major_raw'] = (read_depth['major'] - h[0]) / h[1:].sum()
    read_depth['minor_raw'] = (read_depth['minor'] - h[0]) / h[1:].sum()

    major, minor, length = read_depth.replace(np.inf, np.nan).dropna()[['major_raw', 'minor_raw', 'length']].values.T
    ploidy = ((major + minor) * length).sum() / length.sum()

    return ploidy
