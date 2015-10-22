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
    p = np.vstack([phi, phi, np.ones(phi.shape)]).T

    is_filtered = (l > 0) & np.all(p > 0, axis=1)
    x = x[is_filtered,:]
    l = l[is_filtered]
    p = p[is_filtered,:]

    rd = ((x.T / p.T) / l.T).T
    rd.sort(axis=1)

    rd = pd.DataFrame(rd, columns=['minor', 'major', 'total'])
    rd['length'] = l

    return rd


def calculate_modes(read_depth):
    """ Calculate modes in distribution of read depths

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


def calculate_candidate_h(minor_modes, mix_frac_resolution=20, num_clones=None):
    """ Calculate modes in distribution of read depths for minor allele

    Args:
        minor_modes (list): minor read depth modes

    Kwargs:
        mix_frac_resolution (int): number of mixture fraction candidates
        num_clones (int): number of clones

    Returns:
        list of tuple: candidate haploid normal and tumour read depths

    """

    h_normal = minor_modes.min()
    
    h_tumour_candidates = list()
    h_candidates = list()

    for h_tumour in minor_modes:
        if h_tumour <= h_normal:
            continue

        h_tumour -= h_normal

        # Consider the possibility that the first minor mode
        # is composed of segments with 2 minor copies
        for scale in (1., 0.5):

            h_tumour_scaled = h_tumour * scale

            h_tumour_candidates.append(h_tumour_scaled)

            if num_clones is not None and num_clones == 2:
                h_candidates.append(np.array([h_normal, h_tumour_scaled]))

    # Maximum of 3 clones
    mix_iter = itertools.product(xrange(1, mix_frac_resolution+1), repeat=2)
    for mix in mix_iter:
        if mix != tuple(reversed(sorted(mix))):
            continue

        if sum(mix) != mix_frac_resolution:
            continue
        
        mix = np.array(mix) / float(mix_frac_resolution)

        for h_tumour in h_tumour_candidates:
            h = np.array([h_normal] + list(h_tumour*mix))

            if num_clones is not None and num_clones == 3:
                h_candidates.append(h)

    return h_candidates


def plot_depth(ax, x, l, p, annotated=()):
    """ Plot read depth of major minor and total as a density

    Args:
        ax (matplotlib.axis): optional axis for plotting major/minor/total read depth
        x (numpy.array): observed major, minor, and total read counts
        l (numpy.array): observed lengths of segments
        p (numpy.array): proportion genotypable reads

    KwArgs:
        annotated (list): depths to annotate with verticle lines

    """

    rd = ((x.T / p.T) / l.T)
    rd.sort(axis=0)

    depth_max = np.percentile(rd[2], 95)
    cov = 0.0000001

    remixt.utils.filled_density_weighted(ax, rd[0], l, 'blue', 0.5, 0.0, depth_max, cov)
    remixt.utils.filled_density_weighted(ax, rd[1], l, 'red', 0.5, 0.0, depth_max, cov)
    remixt.utils.filled_density_weighted(ax, rd[2], l, 'grey', 0.5, 0.0, depth_max, cov)

    ylim = ax.get_ylim()
    for depth in annotated:
        ax.plot([depth, depth], [0, 1e16], 'g', lw=2)
    ax.set_ylim(ylim)


