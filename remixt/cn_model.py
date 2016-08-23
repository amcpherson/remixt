import itertools
import collections
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import scipy.misc
import pickle

import remixt.model1
import remixt.model2
import remixt.model3


def _get_brkend_seg_orient(breakend):
    n, side = breakend
    if side == 1:
        n_left = n
        orient = +1
    elif side == 0:
        n_left = n - 1
        orient = -1
    return n_left, orient


class BreakpointModel(object):

    def __init__(self, x, l, adjacencies, breakpoints, max_copy_number=6, normal_contamination=True, prior_variance=1e6):
        """ Create a copy number model.

        Args:
            x (numpy.array): observed minor, major, total reads
            l (numpy.array): observed segment lengths
            adjacencies (list of tuple): pairs of adjacent segments
            breakpoints (list of frozenset of tuples): breakpoints as segment extremity pairs

        KwArgs:
            max_copy_number (int): maximum copy number of HMM state space
            normal_contamination (bool): whether the sample is contaminated by normal
            prior_variance (float): variance parameter of clone segment divergence prior

        """

        self.N = x.shape[0]

        self.breakpoints = list(breakpoints)

        self.normal_contamination = normal_contamination

        self.cn_max = max_copy_number

        self.transition_log_prob = 10.

        self.prior_variance = prior_variance

        # The factor graph model for breakpoint copy number allows only a single breakend
        # interposed between each pair of adjacent segments.  Where multiple breakends are
        # involved, additional zero lenght dummy segments must be added between those
        # breakends
        breakpoint_segment = collections.defaultdict(set)
        for bp_idx, breakpoint in enumerate(self.breakpoints):
            for be_idx, breakend in enumerate(breakpoint):
                n, orient = _get_brkend_seg_orient(breakend)
                breakpoint_segment[n].add((bp_idx, be_idx, orient))

        # Count the number of segments in the new segmentation
        self.N1 = 0
        for n in xrange(-1, self.N):
            if n in breakpoint_segment:
                self.N1 += len(breakpoint_segment[n])
                if (n, n + 1) not in adjacencies:
                    self.N1 += 1
            elif n >= 0:
                self.N1 += 1

        # Mapping from old segment index to new segment index
        self.seg_fwd_remap = np.zeros(self.N, dtype=int)

        # New segment refers to an original segment
        self.seg_is_original = np.zeros(self.N1, dtype=bool)

        self.is_telomere = np.ones(self.N1, dtype=int)
        self.breakpoint_idx = -np.ones(self.N1, dtype=int)
        self.breakpoint_orient = np.zeros(self.N1, dtype=int)

        # Index of new segmentation
        n_new = 0

        # There may be a breakend at the start of the first segment,
        # in which case there will be a breakend with segment n=-1
        for n in xrange(-1, self.N):
            if n >= 0:
                # Map old segment n to n_new
                self.seg_fwd_remap[n] = n_new
                self.seg_is_original[n_new] = True

            if n in breakpoint_segment:
                for bp_idx, be_idx, orient in breakpoint_segment[n]:
                    # Breakpoint index and orientation based on n_new
                    self.breakpoint_idx[n_new] = bp_idx
                    self.breakpoint_orient[n_new] = orient

                    # Breakpoint incident segments cannot be telomeres
                    self.is_telomere[n_new] = 0

                    # Next new segment, per introduced breakend
                    n_new += 1

                # If a breakend is at a telomere, create an additional new segment to be the telomere
                if (n, n + 1) not in adjacencies:
                    # Mark as a telomere
                    self.is_telomere[n_new] = 1

                    # Next new segment, after telomere
                    n_new += 1

            elif n >= 0:
                # If n is not a telomere, n_new is not a telomere
                if (n, n + 1) in adjacencies:
                    self.is_telomere[n_new] = 0

                # Next new segment
                n_new += 1

        assert not np.any((self.breakpoint_idx >= 0) & (self.is_telomere[n] == 1))
        assert np.all(np.bincount(self.breakpoint_idx[self.breakpoint_idx >= 0]) == 2)

        # These should be zero lengthed segments, and zero read counts
        # for dummy segments, but setting lengths and counts to 1 to prevent
        # NaNs should have little effect
        self.x1 = np.ones((self.N1, x.shape[1]), dtype=x.dtype)
        self.l1 = np.ones((self.N1,), dtype=l.dtype)

        self.x1[self.seg_fwd_remap, :] = x
        self.l1[self.seg_fwd_remap] = l

    def _write_model(self, model_filename):
        data = {}
        for a in dir(self.model):
            if a in ['__doc__', '__pyx_vtable__']:
                continue
            if callable(getattr(self.model, a)):
                continue
            data[a] = getattr(self.model, a)
        pickle.dump(data, open(model_filename, 'w'))


    def _read_model(self, model_filename):
        data = pickle.load(open(model_filename, 'r'))
        for a in dir(self.model):
            if a in data:
                setattr(self.model, a, data[a])


    def optimize(self, h_init, elbo_diff_threshold=1e-6, max_iter=5):
        
        M = h_init.shape[0]

        self.model = remixt.model1.RemixtModel(
            M, self.N1, len(self.breakpoints),
            self.cn_max,
            1,
            self.x1,
            self.is_telomere,
            self.breakpoint_idx,
            self.breakpoint_orient,
            self.l1, self.l1,
            self.transition_log_prob,
        )

        self.model.prior_variance = self.prior_variance

        self.model.h = h_init

        elbo = self.model.calculate_elbo()

        elbo_prev = None
        self.num_iter = 0
        self.converged = False
        self.prev_elbo_diff = None
        for self.num_iter in xrange(1, max_iter + 1):
            # import pstats, cProfile
            # cProfile.runctx("self.model.update()", globals(), locals(), "Profile.prof")
            # s = pstats.Stats("Profile.prof")
            # s.strip_dirs().sort_stats("cumtime").print_stats()
            # raise
            elbo = self.model.update()
            print 'elbo', elbo
            print 'h', np.asarray(self.model.h)
            print self.model.calculate_elbo()
            if elbo_prev is not None:
                self.prev_elbo_diff = elbo - elbo_prev
                print 'diff:', self.prev_elbo_diff
                if self.prev_elbo_diff < elbo_diff_threshold:
                    self.converged = True
                    break
            elbo_prev = elbo

        return elbo

    def optimal_cn(self):
        cn = np.zeros((self.model.num_segments, self.model.num_clones, self.model.num_alleles), dtype=int)
        self.model.infer_cn(cn)

        return cn[self.seg_fwd_remap]

    def optimal_brk_cn(self):
        brk_cn = np.argmax(self.model.p_breakpoint, axis=-1)
        brk_cn = dict(zip(self.breakpoints, brk_cn))

        return brk_cn

    def breakpoint_prob(self):
        p_breakpoint = np.asarray(self.model.p_breakpoint)
        brk_prob = dict(zip(self.breakpoints, p_breakpoint))

        return brk_prob

    @property
    def h(self):
        return np.asarray(self.model.h)
        
    @property
    def haplotype_length(self):
        return np.asarray(self.model.effective_lengths[:, 0])
    
    @property
    def phi(self):
        lengths = np.asarray(self.model.effective_lengths[:, 2])
        phi = self.haplotype_length / lengths
        phi[lengths <= 0] = 0
        return phi

    @property
    def a(self):
        return np.asarray(self.model.a)

    @property
    def p_garbage(self):
        return np.asarray(self.model.p_garbage)

    @property
    def p_breakpoint(self):
        return np.asarray(self.model.p_breakpoint)

    @property
    def p_allele(self):
        return np.asarray(self.model.p_allele)


def decode_breakpoints_naive(cn, adjacencies, breakpoints):
    """ Naive decoding of breakpoint copy number.  Finds most likely set of copy numbers given h.

    Args:
        cn (numpy.array): copy number matrix
        adjacencies (list of tuple): ordered pairs of segments representing wild type adjacencies
        breakpoints (list of frozenset of tuple): list of pairs of segment/side pairs representing detected breakpoints

    Returns:
        pandas.DataFrame: table of breakpoint copy number with columns:
            'n_1', 'ell_1', 'side_1', 'n_2', 'ell_2', 'side_2', 'cn_*'

    """

    breakend_adj = dict()
    for seg_1, seg_2 in adjacencies:
        breakend_1 = (seg_1, 1)
        breakend_2 = (seg_2, 0)
        breakend_adj[breakend_1] = breakend_2
        breakend_adj[breakend_2] = breakend_1

    M = cn.shape[1]

    brk_cn = list()

    for breakpoint in breakpoints:

        # Calculate the copy number 'flow' at each breakend for each allele
        breakend_cn = dict()

        for breakend in breakpoint:

            n, side = breakend

            for allele in (0, 1):
                cn_self = cn[n,:,allele]

                if breakend in breakend_adj:
                    n_adj, side_adj = breakend_adj[breakend]
                    cn_adj = cn[n_adj,:,allele]
                else:
                    cn_adj = 0

                cn_residual = np.maximum(cn_self - cn_adj, 0)

                breakend_cn[(n, allele, side)] = cn_residual

        ((n_1, side_1), (n_2, side_2)) = breakpoint

        # For each pair of alleles, starting with matching pairs
        # try to push flow through the breakpoint edge, essentially
        # take the minimum of residual at each breakend for each
        # edge, and update the breakend residual
        for allele_1, allele_2 in ((0, 0), (1, 1), (0, 1), (1, 0)):

            breakpoint_cn = np.minimum(
                breakend_cn[(n_1, allele_1, side_1)],
                breakend_cn[(n_2, allele_2, side_2)])

            breakend_cn[(n_1, allele_1, side_1)] -= breakpoint_cn
            breakend_cn[(n_2, allele_2, side_2)] -= breakpoint_cn

            brk_cn.append([n_1, allele_1, side_1, n_2, allele_2, side_2] + list(breakpoint_cn))

    edge_cols = ['n_1', 'ell_1', 'side_1', 'n_2', 'ell_2', 'side_2']
    cn_cols = ['cn_{0}'.format(m) for m in xrange(M)]

    brk_cn = pd.DataFrame(brk_cn, columns=edge_cols+cn_cols)

    return brk_cn

