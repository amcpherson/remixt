import itertools
import collections
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import scipy.misc
import pickle

from remixt.hmm import _viterbi as hmm_viterbi
from remixt.hmm import _forward as hmm_forward
from remixt.hmm import _backward as hmm_backward
import remixt.vhmm
import remixt.model3
import remixt.model1


class ProbabilityError(ValueError):
    def __init__(self, message, **variables):
        """ Error calculating a probability.

        Args:
            message (str): message detailing error

        KwArgs:
            **variables: variables to be printed

        """

        for name, value in variables.iteritems():
            message += '\n{0}={1}'.format(name, value)

        ValueError.__init__(self, message)


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


class HiddenMarkovModel(object):

    def __init__(self, N, M, emission, prior, chains, max_copy_number=6, normal_contamination=True):
        """ Create a copy number model.

        Args:
            N (int): number of segments
            M (int): number of clones including normal
            emission (ReadCountLikelihood): read count likelihood
            prior (CopyNumberPrior): copy number prior
            chains (list of tuple): start end indices of chromosome chains

        KwArgs:
            max_copy_number (int): maximum copy number of HMM state space
            normal_contamination (bool): whether the sample is contaminated by normal

        """

        self.N = N
        self.M = M
        
        self.emission = emission
        self.prior = prior

        self.chains = chains

        self.normal_contamination = normal_contamination

        self.cn_max = max_copy_number

        self.transition_log_prob = -10.
        self.transition_model = 'geometric_total'

        self.cn_dev_max = 1

        self.log_trans_mat = None


    def build_cns(self, cn_max, cn_dev_max, bounded=True):
        """ Generate a list of copy number states.

        Args:
            cn_max (int): max copy number
            cn_dev_max (int): max clonal deviation of copy number

        KwArgs:
            bounded (bool): filter negative and greater than max copy numbers

        Yields:
            numpy.array: array of copy number states

        """

        base_cn_iter = itertools.product(np.arange(0.0, cn_max + 1.0, 1.0), repeat=2)
        dev_cn_iter = itertools.product(np.arange(-cn_dev_max, cn_dev_max + 1.0, 1.0), repeat=2*(self.M-2))

        for base_cn, dev_cn in itertools.product(base_cn_iter, dev_cn_iter):

            base_cn = np.array(base_cn).reshape(2)
            dev_cn = np.array(dev_cn).reshape((self.M-2,2))
            
            subclone_cn = dev_cn + base_cn
            
            if bounded and (np.any(subclone_cn < 0) or np.any(subclone_cn > cn_max)):
                continue

            if self.normal_contamination:
                normal_cn = np.ones(2)
            else:
                normal_cn = np.zeros(2)

            cn = np.array([normal_cn] + [base_cn] + list(subclone_cn))

            # Check for the situation in which not all alleles are the same
            # across all clones, but all of the major alleles have less or
            # equal copy number than the minor alleles
            if np.any(cn[1:,0] != cn[1:,1]) and np.all(cn[1:,0] <= cn[1:,1]):
                continue

            cn = np.array([cn] * self.N)

            yield cn


    def build_cn_states(self):
        """ Build a list of hmm copy number states.

        Returns:
            numpy.array: array of hmm copy number states

        """

        cn_states = np.array(list(self.build_cns(self.cn_max, self.cn_dev_max)))

        return cn_states


    def build_log_trans_mat(self):
        """ Build the log transition matrix.

        Returns:
            numpy.array: transition matrix

        """

        if self.log_trans_mat is not None:
            return self.log_trans_mat

        cn_states = self.build_cn_states()
        num_states = cn_states.shape[0]

        self.log_trans_mat = np.zeros((num_states, num_states))

        def _transition_cost_step(cn_1, cn_2):
            return np.any(cn_1 != cn_2) * 1.

        def _transition_cost_step_total(cn_1, cn_2):
            return np.sum(cn_1.sum(axis=-1) != cn_2.sum(axis=-1), axis=-1)

        def _transition_cost_step_allele(cn_1, cn_2):
            return np.any(cn_1 != cn_2).sum(axis=(-1, -2))

        def _transition_cost_geometric_total(cn_1, cn_2):
            return np.sum(np.absolute(cn_1.sum(axis=-1) - cn_2.sum(axis=-1)), axis=-1)

        def _transition_cost_geometric_allele(cn_1, cn_2):
            return np.absolute(cn_1 - cn_2).sum(axis=(-1, -2))

        f_cost = None
        if self.transition_model == 'step':
            f_cost = _transition_cost_step
        elif self.transition_model == 'step_total':
            f_cost = _transition_cost_step_total
        elif self.transition_model == 'step_allele':
            f_cost = _transition_cost_step_allele
        elif self.transition_model == 'geometric_total':
            f_cost = _transition_cost_geometric_total
        elif self.transition_model == 'geometric_allele':
            f_cost = _transition_cost_geometric_allele
        else:
            raise ValueError('Unknown transition model {0}'.format(self.transition_model))

        for idx_1 in xrange(num_states):
            cn_1 = cn_states[idx_1, 0, :, :]
            for idx_2 in xrange(num_states):
                cn_2 = cn_states[idx_2, 0, :, :]
                self.log_trans_mat[idx_1, idx_2] = self.transition_log_prob * f_cost(cn_1, cn_2)

        self.log_trans_mat -= scipy.misc.logsumexp(self.log_trans_mat, axis=0)

        return self.log_trans_mat


    def emission_probabilities(self):
        """ Calculate the log posterior over copy number states.
        
        Returns:
            numpy.array: copy number states
            numpy.array: emission probabilities

        """

        cn_states = self.build_cn_states()

        probs = list()

        for cn in cn_states:

            log_prob = self.emission.log_likelihood(cn) + self.prior.log_prior(cn)

            probs.append(log_prob)

        probs = np.array(probs)

        return cn_states, probs


    def _forward_backward(self, log_start_prob, log_trans_mat, frame_log_prob):

        n_observations, n_components = frame_log_prob.shape
        fwd_lattice = np.zeros((n_observations, n_components))
        hmm_forward(n_observations, n_components, log_start_prob, log_trans_mat, frame_log_prob, fwd_lattice)
        fwd_lattice[fwd_lattice <= -1e200] = -np.inf

        log_prob = scipy.misc.logsumexp(fwd_lattice[-1])

        bwd_lattice = np.zeros((n_observations, n_components))
        hmm_backward(n_observations, n_components, log_start_prob, log_trans_mat, frame_log_prob, bwd_lattice)
        bwd_lattice[bwd_lattice <= -1e200] = -np.inf

        gamma = fwd_lattice + bwd_lattice

        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively. (copied from hmmlearn)

        posteriors = np.exp(gamma.T - scipy.misc.logsumexp(gamma, axis=1)).T
        posteriors /= np.sum(posteriors, axis=1).reshape((-1, 1))
        assert not np.any(np.isnan(posteriors))
        posteriors = posteriors.T

        return log_prob, posteriors


    def posterior_marginals(self):
        """ Calculate the forward backward posterior marginals.
        
        Returns:
            float: log posterior
            numpy.array: posterior marginals

        Posterior marginals matrix has shape (S,N) for N variables with S states

        """

        cn_states, probs = self.emission_probabilities()

        log_start_prob = np.zeros((cn_states.shape[0],))
        log_start_prob -= scipy.misc.logsumexp(log_start_prob)

        log_trans_mat = self.build_log_trans_mat()

        frame_log_prob = probs.T

        posteriors = np.zeros(probs.shape)

        log_prob = 0
        for n_1, n_2 in self.chains:
            chain_log_prob, posteriors[:, n_1:n_2] = self._forward_backward(log_start_prob, log_trans_mat, frame_log_prob[n_1:n_2, :])
            log_prob += chain_log_prob

        return log_prob, cn_states, posteriors


    def _viterbi(self, log_start_prob, log_trans_mat, frame_log_prob):

        n_observations, n_components = frame_log_prob.shape
        state_sequence, log_prob = hmm_viterbi(n_observations, n_components, log_start_prob, log_trans_mat, frame_log_prob)

        return log_prob, state_sequence


    def optimal_state(self):
        """ Calculate the viterbi path
        
        Returns:
            numpy.array: log posterior
            numpy.array: variable state

        State array has shape (N,...) for N variables

        """

        cn_states, probs = self.emission_probabilities()

        log_start_prob = np.zeros((cn_states.shape[0],))
        log_start_prob -= scipy.misc.logsumexp(log_start_prob)

        log_trans_mat = self.build_log_trans_mat()

        frame_log_prob = probs.T

        state_sequence = np.zeros((frame_log_prob.shape[0],), dtype=int)

        log_prob = 0
        for n_1, n_2 in self.chains:
            chain_log_prob, state_sequence[n_1:n_2] = self._viterbi(log_start_prob, log_trans_mat, frame_log_prob[n_1:n_2, :])
            log_prob += chain_log_prob

        cn_state = cn_states[state_sequence, xrange(len(state_sequence))]

        return log_prob, cn_state


    def log_likelihood(self, state):
        return self.emission.log_likelihood(state) + self.prior.log_prior(state)


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


    def initialize_1(self, h_normal, h_tumour, split_sequence=None):

        self.model = remixt.model1.RemixtModel(
            2, self.N1, len(self.breakpoints),
            self.cn_max,
            self.x1,
            self.is_telomere,
            self.breakpoint_idx,
            self.breakpoint_orient,
            self.l1, self.l1,
            self.transition_log_prob,
        )

        self.model.prior_variance = self.prior_variance

        self.model.h[0] = h_normal
        self.model.h[1] = h_tumour

        for _ in range(3):
            for m in range(1, self.model.num_clones):
                print _, m, '-----'
                elbo_prev = self.model.calculate_elbo()
                self.model.update_p_cn(m)
                elbo = self.model.calculate_elbo()
                print 'elbo diff:', elbo - elbo_prev

        posterior_marginals = np.asarray(self.model.posterior_marginals)**0.1
        posterior_marginals /= np.sum(posterior_marginals, axis=-1)[:, :, np.newaxis]
        self.model.posterior_marginals = posterior_marginals

        joint_posterior_marginals = np.asarray(self.model.joint_posterior_marginals)**0.1
        joint_posterior_marginals /= np.sum(joint_posterior_marginals, axis=(-1, -2))[:, :, np.newaxis, np.newaxis]
        self.model.joint_posterior_marginals = joint_posterior_marginals

        for n in xrange(self.model.num_segments):
            for m in xrange(1, self.model.num_clones):
                if self.breakpoint_idx[n] == 10:
                    pst = np.asarray(self.model.posterior_marginals[m, n, :])
                    pr = pst.max()
                    print m, pr, np.where(pst==pr), np.asarray(self.model.posterior_marginals[m, n, :])[[10, 18]]
                    pst = np.asarray(self.model.posterior_marginals[m, n+1, :])
                    pr = pst.max()
                    print m, pr, np.where(pst==pr), np.asarray(self.model.posterior_marginals[m, n, :])[[10, 18]]

        m, f = split_sequence[0]
        self.model.split_clone(m, f)

        print np.asarray(self.model.h)

        for _ in range(2):
            for m in range(1, self.model.num_clones):
                print _, m, '-----'
                elbo_prev = self.model.calculate_elbo()
                self.model.update_p_cn(m)
                elbo = self.model.calculate_elbo()
                print 'elbo diff:', elbo - elbo_prev

        # for m, f in itertools.chain([(None, None)], split_sequence):
        #     print '-' * 100

        #     if m is not None:
        #         self.model.split_clone(m, f)
        #         print np.asarray(self.model.h)

        #     for _ in range(3):
        #         for m in range(1, self.model.num_clones):
        #             print _, m, '-----'
        #             elbo_prev = self.model.calculate_elbo()
        #             self.model.update_p_cn(m)
        #             elbo = self.model.calculate_elbo()
        #             print 'elbo diff:', elbo - elbo_prev

        #         print _, 'breakpoint', '-----'
        #         elbo_prev = self.model.calculate_elbo()
        #         self.model.update_p_breakpoint()
        #         elbo = self.model.calculate_elbo()
        #         print 'elbo diff:', elbo - elbo_prev

        #         print self.breakpoint_prob()[frozenset([(7324, 0), (7520, 1)])]
        #         print np.asarray(self.model.p_allele)[7324]
        #         print np.asarray(self.model.p_allele)[7521]

        #         print _, 'allele', '-----'
        #         elbo_prev = self.model.calculate_elbo()
        #         self.model.update_p_allele()
        #         elbo = self.model.calculate_elbo()
        #         print 'elbo diff:', elbo - elbo_prev



        # for m, f in itertools.chain([(None, None)], split_sequence):
        #     print '-' * 100

        #     if m is not None:
        #         self.model.split_clone(m, f)

        #         p_breakpoint = np.zeros((self.model.num_breakpoints, self.model.num_clones, self.model.cn_max + 1))
        #         p_breakpoint[:, :, 0] = 0.25
        #         p_breakpoint[:, :, 1] = 0.75
        #         self.model.p_breakpoint = p_breakpoint

        #         p_allele = np.ones((self.model.num_segments, 2, 2))
        #         p_allele[:, 0, 0] = 0.25
        #         p_allele[:, 0, 1] = 0.25
        #         p_allele[:, 1, 0] = 0.25
        #         p_allele[:, 1, 1] = 0.25
        #         self.model.p_allele = p_allele

        #         break

        #     for _ in range(3):
        #         for m in reversed(range(1, self.model.num_clones)):
        #             print _, m, '-----'
        #             elbo_prev = self.model.calculate_elbo()
        #             self.model.update_p_cn(m)
        #             elbo = self.model.calculate_elbo()
        #             print 'elbo diff:', elbo - elbo_prev

        #         print _, 'breakpoint', '-----'
        #         elbo_prev = self.model.calculate_elbo()
        #         self.model.update_p_breakpoint()
        #         elbo = self.model.calculate_elbo()
        #         print 'elbo diff:', elbo - elbo_prev

        #         print self.breakpoint_prob()[frozenset([(7324, 0), (7520, 1)])]
        #         print np.asarray(self.model.p_allele)[7324]
        #         print np.asarray(self.model.p_allele)[7521]

        #         print _, 'allele', '-----'
        #         elbo_prev = self.model.calculate_elbo()
        #         self.model.update_p_allele()
        #         elbo = self.model.calculate_elbo()
        #         print 'elbo diff:', elbo - elbo_prev

    def initialize_2(self, ):

        self.model = remixt.model1.RemixtModel(
            3, self.N1, len(self.breakpoints),
            self.cn_max,
            self.x1,
            self.is_telomere,
            self.breakpoint_idx,
            self.breakpoint_orient,
            self.l1, self.l1,
            self.transition_log_prob,
        )

        self.model.prior_variance = self.prior_variance

        self.model.h[0] = 0.04
        self.model.h[1] = 0.06
        self.model.h[2] = 0.04

        jointlogprob = np.ones((self.model.num_segments, self.model.num_cn_states, self.model.num_cn_states)) * -np.inf

        s_0 = self.model.normal_state
        for s_1, s_2 in itertools.product(range(len(self.model.cn_states)), repeat=2):
            cn_1 = np.asarray(self.model.cn_states[s_1])
            cn_2 = np.asarray(self.model.cn_states[s_2])
            if np.any(np.absolute(cn_1 - cn_2) > 1):
                continue
            jointlogprob[:, s_1, s_2] = self.model.calculate_ll(np.array([s_0, s_1, s_2]))

        posterior_marginals = np.zeros((self.model.num_clones, self.model.num_segments, self.model.num_cn_states))

        print 'postmarg'
        posterior_marginals[0, :, s_0] = 1.
        for n in xrange(self.model.num_segments):
            joint_posterior = np.exp(jointlogprob[n, :, :])
            joint_posterior /= joint_posterior.sum()
            posterior_marginals[1, n, :] = joint_posterior.sum(axis=1)
            posterior_marginals[2, n, :] = joint_posterior.sum(axis=0)
        print 'postmarg done'

        posterior_marginals = np.asarray(posterior_marginals)**0.25
        posterior_marginals /= np.sum(posterior_marginals, axis=-1)[:, :, np.newaxis]
        self.model.posterior_marginals = posterior_marginals

        for n in xrange(self.model.num_segments):
            if self.breakpoint_idx[n] == self.print_breakpoint_idx:
                pst = np.asarray(jointlogprob[n, :, :])
                pr = pst.max()
                print n, pr, np.where(pst==pr), np.asarray(self.model.effective_lengths)[n]
                pst = np.asarray(jointlogprob[n+1, :, :])
                pr = pst.max()
                print n+1, pr, np.where(pst==pr), np.asarray(self.model.effective_lengths)[n+1]

        for n in xrange(self.model.num_segments):
            for m in xrange(1, self.model.num_clones):
                if self.breakpoint_idx[n] == self.print_breakpoint_idx:
                    pst = np.asarray(posterior_marginals[m, n, :])
                    pr = pst.max()
                    print n, m, pr, np.where(pst==pr)
                    pst = np.asarray(posterior_marginals[m, n+1, :])
                    pr = pst.max()
                    print n+1, m, pr, np.where(pst==pr)

        self.model.posterior_marginals = posterior_marginals

        for _ in range(5):
            for m in range(1, self.model.num_clones):
                print _, m, '-----'
                elbo_prev = self.model.calculate_elbo()
                self.model.update_p_cn(m)
                elbo = self.model.calculate_elbo()
                print 'elbo diff:', elbo - elbo_prev


    def initialize_3(self, ):

        self.model = remixt.model1.RemixtModel(
            3, self.N1, len(self.breakpoints),
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

        self.model.h[0] = 0.04
        self.model.h[1] = 0.06
        self.model.h[2] = 0.04

    def optimize(self, h_init, elbo_diff_threshold=1e-6, max_iter=5):

        self.model = remixt.model1.RemixtModel(
            3, self.N1, len(self.breakpoints),
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
                print 'diff:', elbo - elbo_prev
                if elbo - elbo_prev < elbo_diff_threshold:
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

