import itertools
import numpy as np

import sklearn
import sklearn.cluster

import remixt.utils
import remixt.likelihood


def candidate_h(x, l, mix_frac_resolution=20, num_clones=None, ax=None):
    """ Use a k-means to identify candidate haploid read depths

    Args:
        x (numpy.array): observed major, minor, and total read counts
        l (numpy.array): observed lengths of segments

    Kwargs:
        mix_frac_resolution (int): number of mixture fraction candidates
        num_clones (int): number of clones
        ax (matplotlib.axis): optional axis for plotting major/minor/total read depth

    Returns:
        list of tuple: candidate haploid normal and tumour read depths

    """

    assert num_clones is None or num_clones in [2, 3]

    phi = remixt.likelihood.estimate_phi(x)
    p = np.vstack([phi, phi, np.ones(phi.shape)]).T

    is_filtered = (l > 0) & np.all(p > 0, axis=1)
    x = x[is_filtered,:]
    l = l[is_filtered]
    p = p[is_filtered,:]

    rd = ((x.T / p.T) / l.T)

    rd_min = np.minimum(rd[0], rd[1])
    rd_max = np.maximum(rd[0], rd[1])

    # Cluster minor read depths using kmeans
    rd_min_samples = remixt.utils.weighted_resample(rd_min, l)
    kmm = sklearn.cluster.KMeans(n_clusters=5)
    kmm.fit(rd_min_samples.reshape((rd_min_samples.size, 1)))
    means = kmm.cluster_centers_[:,0]

    h_normal = means.min()

    h_tumour_candidates = list()

    h_candidates = list()

    for h_tumour in means:
        
        if h_tumour <= h_normal:
            continue

        h_tumour -= h_normal

        h_tumour_candidates.append(h_tumour)

        if num_clones is not None and num_clones == 2:
            h_candidates.append(np.array([h_normal, h_tumour]))

    if ax is not None:
        plot_depth(ax, x, l, p, annotated=means)

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



