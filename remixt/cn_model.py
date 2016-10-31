import itertools
import collections
import numpy as np
import pandas as pd
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
            disable_breakpoints (bool): disable integrated breakpoint copy number inference 

        """
        
        # Observed data should be ordered as major, minor, total
        assert np.all(x[:, 1] <= x[:, 0])

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
        self.disable_breakpoints = kwargs.get('disable_breakpoints', False)

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
        
        # Create variables required for lower level model object
        num_breakpoints = len(self.breakpoints)
        is_telomere = np.ones(self.N1, dtype=int)
        breakpoint_idx = -np.ones(self.N1, dtype=int)
        breakpoint_orient = np.zeros(self.N1, dtype=int)

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
                    breakpoint_idx[n_new] = bp_idx
                    breakpoint_orient[n_new] = orient

                    # Breakpoint incident segments cannot be telomeres
                    is_telomere[n_new] = 0

                    # Next new segment, per introduced breakend
                    n_new += 1

                # If a breakend is at a telomere, create an additional new segment to be the telomere
                if (n, n + 1) not in adjacencies:
                    # Mark as a telomere
                    is_telomere[n_new] = 1

                    # Next new segment, after telomere
                    n_new += 1

            elif n >= 0:
                # If n is not a telomere, n_new is not a telomere
                if (n, n + 1) in adjacencies:
                    is_telomere[n_new] = 0

                # Next new segment
                n_new += 1

        assert not np.any((breakpoint_idx >= 0) & (is_telomere == 1))
        assert np.all(np.bincount(breakpoint_idx[breakpoint_idx >= 0]) == 2)

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
        
        # Optionally disable integrated breakpoint copy number inference
        if self.disable_breakpoints:
            num_breakpoints = 0
            breakpoint_idx = -np.ones(breakpoint_idx.shape, dtype=int)
            breakpoint_orient = np.zeros(breakpoint_orient.shape, dtype=int)

        self.model = remixt.model1a.RemixtModel(
            self.M, self.N1, num_breakpoints,
            self.max_copy_number,
            self.max_copy_number_diff,
            self.normal_contamination,
            is_telomere,
            breakpoint_idx,
            breakpoint_orient,
            self.transition_log_prob,
        )

        self.prev_elbo = None
        self.prev_elbo_diff = None
        self.num_update_iter = 1
        
        # Indicator variables for allele swaps
        # Initalize to uniform over 2 states
        self.p_allele_swap = np.ones((self.N1, 2)) * 0.5
        
        self.cn_states = np.concatenate([
            np.asarray(self.model.cn_states),
            np.asarray(self.model.cn_states)[:, :, ::-1]])

    @property
    def likelihood_params(self):
        params = [
            self.emission.h_param(self.cn_states),
            self.emission.r_param(self.cn_states),
            self.emission.M_param(self.cn_states),
            self.emission.betabin_mix_param(self.cn_states),
            self.emission.negbin_mix_param(self.cn_states),
        ]

        if not self.normal_contamination:
            params.append([
                self.emission.r_hdel_param(self.cn_states),
                self.emission.M_loh_param(self.cn_states),
                self.emission.betabin_loh_mix(self.cn_states),
                self.emission.negbin_hdel_mix(self.cn_states),
                self.emission.hdel_mu(self.cn_states),
                self.emission.loh_p(self.cn_states),
            ])

        return params

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

    def log_likelihood(self, s):
        cn = self.cn_states[s, :, :][np.newaxis, :, :]
        return self.emission.log_likelihood(cn) + self.prior.log_prior(cn)

    def calculate_framelogprob(self, allele_swap):
        framelogprob = np.zeros((self.model.num_segments, self.model.num_cn_states))
        
        for s in xrange(self.model.num_cn_states):
            cn = np.array([np.asarray(self.model.cn_states[s])] * self.model.num_segments)
            if allele_swap == 1:
                cn = cn[:, :, ::-1]
            framelogprob[:, s] = self.emission.log_likelihood(cn) + self.prior.log_prior(cn)
        
        return framelogprob

    def posterior_marginals(self):
        framelogprob = (
            self.p_allele_swap[:, 0][:, np.newaxis] * self.calculate_framelogprob(0) + 
            self.p_allele_swap[:, 1][:, np.newaxis] * self.calculate_framelogprob(1))
        
        self.model.framelogprob = framelogprob

        for num_iter in xrange(self.num_update_iter):
            self.update()
            
        posterior_marginals = np.asarray(self.model.posterior_marginals)
        
        posterior_marginals = np.concatenate([
            (posterior_marginals * self.p_allele_swap[:, 0][:, np.newaxis]).T,
            (posterior_marginals * self.p_allele_swap[:, 1][:, np.newaxis]).T])
        
        return self.prev_elbo, posterior_marginals

    def _check_elbo(self, prev_elbo, name):
        threshold = -1e-6
        elbo = self.model.calculate_elbo()
        print 'elbo: {:.10f}'.format(elbo)
        print 'elbo diff: {:.10f}'.format(elbo - prev_elbo)
        if elbo - prev_elbo < threshold:
            raise Exception('elbo error for step {}!'.format(name))
        prev_elbo = elbo
        return elbo

    def update(self, check_elbo=False):
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
            
        print 'update_p_allele_swap'
        posterior_marginals = np.asarray(self.model.posterior_marginals)
        log_p_allele_swap = np.zeros(self.p_allele_swap.shape)
        log_p_allele_swap[:, 0] = (posterior_marginals * self.calculate_framelogprob(0)).sum(axis=1)
        log_p_allele_swap[:, 1] = (posterior_marginals * self.calculate_framelogprob(1)).sum(axis=1)
        self.p_allele_swap = np.exp(log_p_allele_swap - scipy.misc.logsumexp(log_p_allele_swap, axis=1)[:, np.newaxis])

        if check_elbo:
            elbo = self._check_elbo(elbo, 'update_p_allele_swap')
            
        print 'done'
        
        if not check_elbo:
            elbo = self.model.calculate_elbo()
            print 'elbo: {:.10f}'.format(elbo)
            
        self.prev_elbo_diff = self.prev_elbo - elbo
        self.prev_elbo = elbo

    def optimal_cn(self):
        cn = np.zeros((self.model.num_segments, self.model.num_clones, self.model.num_alleles), dtype=int)
        
        self.model.infer_cn(cn)
        
        # Swap alleles as required
        swap_alleles = self.p_allele_swap[:, 1] > self.p_allele_swap[:, 0]
        cn[swap_alleles, :, :] = cn[swap_alleles, :, ::-1]
        
        # Remap to original segmentation
        cn = cn[self.seg_fwd_remap]

        return cn
        
    def optimal_brk_cn(self):
        brk_cn = []
        
        for brk_idx, p_breakpoint in enumerate(np.asarray(self.model.p_breakpoint)):
            s_b = p_breakpoint.argmax()
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
    
    # Calculate breakpoint copy number based on total copy number transitions
    cn = cn.sum(axis=-1)

    breakend_adj = dict()
    for seg_1, seg_2 in adjacencies:
        breakend_1 = (seg_1, 1)
        breakend_2 = (seg_2, 0)
        breakend_adj[breakend_1] = breakend_2
        breakend_adj[breakend_2] = breakend_1

    M = cn.shape[1]

    brk_cn = dict()

    for breakpoint in breakpoints:

        # Calculate the copy number 'flow' at each breakend
        breakend_cn = dict()

        for breakend in breakpoint:
            n, side = breakend

            cn_self = cn[n,:]

            if breakend in breakend_adj:
                n_adj, side_adj = breakend_adj[breakend]
                cn_adj = cn[n_adj, :]
            else:
                cn_adj = 0

            cn_residual = np.maximum(cn_self - cn_adj, 0)

            breakend_cn[(n, side)] = cn_residual

        ((n_1, side_1), (n_2, side_2)) = breakpoint

        breakpoint_cn = np.minimum(
            breakend_cn[(n_1, side_1)],
            breakend_cn[(n_2, side_2)])

        brk_cn[breakpoint] = breakpoint_cn

    return brk_cn
