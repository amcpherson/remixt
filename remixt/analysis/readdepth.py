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


def calculate_candidate_h(minor_modes, num_clones=None, num_mix_samples=20, dirichlet_alpha=1.0):
    """ Calculate modes in distribution of read depths for minor allele

    Args:
        minor_modes (list): minor read depth modes

    Kwargs:
        num_clones (int): number of clones, if None, sample from poisson
        num_mix_samples (int): number of mixture fraction samples
        dirichlet_alpha (float): alpha parameter for dirichlet sampling

    Returns:
        list of numpy.array: candidate haploid normal and tumour read depths

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
            for _ in xrange(num_mix_samples):
                if num_clones is None:
                    while True:
                        num_tumour_clones = np.random.geometric(p=0.5)
                        if num_tumour_clones <= 3:
                            break
                else:
                    num_tumour_clones = num_clones - 1

                mix = np.random.dirichlet([dirichlet_alpha] * num_tumour_clones)

                h = np.array([h_normal] + list(h_tumour * scale * mix))

                h_candidates.append(h)

    return h_candidates


def filter_high_ploidy(candidate_h, experiment, max_ploidy=5.0):
    """ Calculate modes in distribution of read depths for minor allele

    Args:
        candidate_h (list of numpy.array): candidate haploid normal and tumour read depths
        experiment (remixt.Experiment): experiment object

    Kwargs:
        max_ploidy (int): maximum ploidy of potential solutions

    Returns:
        candidate_h (list of numpy.array): candidate haploid normal and tumour read depths

    """

    def calculate_ploidy(h):
        mean_cn = remixt.likelihood.calculate_mean_cn(h, experiment.x, experiment.l)
        return (mean_cn.T * experiment.l).sum() / experiment.l.sum()

    return filter(lambda h: calculate_ploidy(h) < max_ploidy, candidate_h)


