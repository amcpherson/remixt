import itertools
import collections
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import scipy.misc
import pickle

import remixt.model1a
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


class CopyNumberPrior(object):

    def __init__(self, l, max_divergence=1, divergence_weight=1e-7):
        """ Create a copy number prior.

        Args:
            l (numpy.array): observed lengths of segments

        KwArgs:
            max_divergence (int): maxmimum allowed divergence between segments
            divergence_weight (float): prior length scaled weight on divergent segments

        """
        
        self.l = l

        self.max_divergence = max_divergence
        self.divergence_weight = divergence_weight

        if divergence_weight < 0.:
            raise ValueError('divergence weight should be positive, was {}'.format(divergence_weight))


    def log_prior(self, cn):
        """ Evaluate log prior probability of segment copy number.
        
        Args:
            cn (numpy.array): copy number state of segments

        Returns:
            numpy.array: log prior per segment

        """

        subclonal = (cn[:,1:,:].max(axis=1) != cn[:,1:,:].min(axis=1)) * 1
        lp = -1.0 * np.sum(subclonal, axis=1) * self.l * self.divergence_weight

        subclonal_divergence = (cn[:,1:,:].max(axis=1) - cn[:,1:,:].min(axis=1)) * 1
        invalid_divergence = (subclonal_divergence > self.max_divergence).any(axis=1)
        lp -= np.where(invalid_divergence, 1000., 0.)

        return lp


class BreakpointModel(object):

    def __init__(self, h_init, x, l, adjacencies, breakpoints, **kwargs):
        """ Create a copy number model.

        Args:
            h_init (numpy.array): per clone haploid read depth
            x (numpy.array): observed minor, major, total reads
            l (numpy.array): observed segment lengths
            adjacencies (list of tuple): pairs of adjacent segments
            breakpoints (list of frozenset of tuples): breakpoints as segment extremity pairs

        KwArgs:
            max_copy_number (int): maximum copy number of HMM state space
            normal_contamination (bool): whether the sample is contaminated by normal
            divergence_weight (float): clone segment divergence prior parameter
            min_segment_length (float): minimum size of segments for segment likelihood mask
            min_proportion_genotyped (float): minimum proportion genotyped reads for segment likelihood mask
            transition_log_prob (float): penalty on transitions, per copy number change

        """

        self.M = h_init.shape[0]
        self.N = x.shape[0]

        self.breakpoints = list(breakpoints)
        
        self.max_copy_number = kwargs.get('max_copy_number', 6)
        self.max_copy_number_diff = kwargs.get('max_copy_number_diff', 1)
        self.normal_contamination = kwargs.get('normal_contamination', True)
        self.divergence_weight = kwargs.get('divergence_weight', 1e6)
        self.min_segment_length = kwargs.get('min_segment_length', 10000)
        self.min_proportion_genotyped = kwargs.get('min_proportion_genotyped', 0.01)
        self.transition_log_prob = kwargs.get('transition_log_prob', 10.)

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

        assert not np.any((self.breakpoint_idx >= 0) & (self.is_telomere == 1))
        assert np.all(np.bincount(self.breakpoint_idx[self.breakpoint_idx >= 0]) == 2)

        # These should be zero lengthed segments, and zero read counts
        # for dummy segments, but setting lengths and counts to 1 to prevent
        # NaNs should have little effect
        self.x1 = np.ones((self.N1, x.shape[1]), dtype=x.dtype)
        self.l1 = np.ones((self.N1,), dtype=l.dtype)

        self.x1[self.seg_fwd_remap, :] = x
        self.l1[self.seg_fwd_remap] = l

        # Create emission / prior / copy number models
        self.emission = remixt.likelihood.NegBinBetaBinLikelihood(self.x1, self.l1)
        self.emission.h = h_init

        # Create prior probability model
        self.prior = remixt.cn_model.CopyNumberPrior(self.l1, divergence_weight=self.divergence_weight)

        # Mask amplifications and poorly modelled segments from likelihood
        self.emission.add_amplification_mask(self.max_copy_number)
        self.emission.add_segment_length_mask(self.min_segment_length)
        self.emission.add_proportion_genotyped_mask(self.min_proportion_genotyped)

        self.model = remixt.model1a.RemixtModel(
            self.M, self.N1, len(self.breakpoints),
            self.max_copy_number,
            self.max_copy_number_diff,
            self.normal_contamination,
            self.is_telomere,
            self.breakpoint_idx,
            self.breakpoint_orient,
            self.transition_log_prob,
        )

        self.prev_elbo = None
        self.prev_elbo_diff = None
        self.num_update_iter = 1
        
    
    @property
    def likelihood_params(self):
        return [
            self.emission.h_param,
            self.emission.r_param,
            self.emission.M_param,
            # self.emission.z_param,
            self.emission.hdel_mu_param,
            self.emission.loh_p_param,
        ]


    def get_model_data(self):
        data = {}
        for a in dir(self.model):
            if a in ['__doc__', '__pyx_vtable__']:
                continue
            if callable(getattr(self.model, a)):
                continue
            if type(getattr(self.model, a)).__name__ == '_memoryviewslice':
                data[a] = np.asarray(getattr(self.model, a))
            else:
                data[a] = getattr(self.model, a)
        return data


    def _write_model(self, model_filename):
        data = self.get_model_data()
        pickle.dump(data, open(model_filename, 'w'))


    def _read_model(self, model_filename):
        data = pickle.load(open(model_filename, 'r'))
        for a in dir(self.model):
            if a in data:
                setattr(self.model, a, data[a])
                
                
    def log_likelihood(self, cn):
        return self.emission.log_likelihood(cn) + self.prior.log_prior(cn)
        
        
    def posterior_marginals(self):
        framelogprob = np.zeros((self.model.num_segments, self.model.num_cn_states))
        
        cn_states = []
        for s in xrange(self.model.num_cn_states):
            cn = np.array([np.asarray(self.model.cn_states[s])] * self.model.num_segments)
            framelogprob[:, s] = self.log_likelihood(cn)
            cn_states.append(cn)
        cn_states = np.array(cn_states)
            
        self.model.framelogprob = framelogprob

        for num_iter in xrange(self.num_update_iter):
            self.update()
        
        posterior_marginals = np.asarray(self.model.posterior_marginals).T
        
        return self.prev_elbo, cn_states, posterior_marginals


    def _check_elbo(self, prev_elbo, name):
        threshold = -1e-6
        elbo = self.model.calculate_elbo()
        print 'elbo: {:.10f}'.format(elbo)
        print 'elbo diff: {:.10f}'.format(elbo - prev_elbo)
        if elbo - prev_elbo < threshold:
            raise Exception('elbo error for step {}!'.format(name))
        prev_elbo = elbo
        return elbo
        
        
    def update(self, check_elbo=True):
        """ Single update of all variational parameters.
        """

        if self.prev_elbo is None:
            self.prev_elbo = self.model.calculate_elbo()
            print 'elbo: {:.10f}'.format(self.prev_elbo)

        elbo = self.prev_elbo

        print 'update_p_cn'
        self.model.update_p_cn()

        if check_elbo:
            elbo = self._check_elbo(elbo, 'p_cn')

        print 'update_p_breakpoint'
        self.model.update_p_breakpoint()

        if check_elbo:
            elbo = self._check_elbo(elbo, 'p_breakpoint')

        print 'done'
        
        if not check_elbo:
            elbo = self.model.calculate_elbo()
            print 'elbo: {:.10f}'.format(elbo)
            
        self.prev_elbo_diff = self.prev_elbo - elbo
        self.prev_elbo = elbo


    def optimize(self, h_init, elbo_diff_threshold=1e-6, max_update_iter=5, max_update_var_iter=5):
        
        M = h_init.shape[0]

        elbo = self.model.calculate_elbo()

        elbo_prev = None
        self.num_iter = 0
        self.converged = False
        self.prev_elbo_diff = None
        
        for self.num_iter in xrange(1, max_update_iter + 1):
            # import pstats, cProfile
            # cProfile.runctx("self.model.update()", globals(), locals(), "Profile.prof")
            # s = pstats.Stats("Profile.prof")
            # s.strip_dirs().sort_stats("cumtime").print_stats()
            # raise
            elbo = self.update(update_variance=False)
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

        for self.num_iter in xrange(1, max_update_var_iter + 1):
            elbo = self.update(update_variance=True)
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
        brk_cn = []
        
        for brk_idx, p_breakpoint in enumerate(np.asarray(self.model.p_breakpoint)):
            prob = p_breakpoint.max()
            s_b, v_1, v_2 = np.where(p_breakpoint == prob)
            s_b = s_b[0]
            v_1 = v_1[0]
            v_2 = v_2[0]
            
            brk_cn.append(np.asarray(self.model.brk_states)[s_b])
            
        brk_cn = dict(zip(self.breakpoints, brk_cn))

        return brk_cn

    def breakpoint_prob(self):
        p_breakpoint = np.asarray(self.model.p_breakpoint)
        brk_prob = dict(zip(self.breakpoints, p_breakpoint))

        return brk_prob

    @property
    def h(self):
        return np.asarray(self.emission.h)

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

