import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import scipy.misc
from scipy.special import gammaln

import sklearn
import sklearn.cluster
import sklearn.mixture

import hmmlearn
from hmmlearn._hmmc import _viterbi as hmm_viterbi
from hmmlearn._hmmc import _forward as hmm_forward
from hmmlearn._hmmc import _backward as hmm_backward

import remixt.genome_graph
import remixt.nb_overdispersion
import remixt.utils
import remixt.em


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

    def __init__(self, N, M, cn_prob, allele_specific=True):
        """Create a copy number model.

        Args:
            N (int): number of segments
            M (int): number of clones including normal
            cn_prob (numpy.array): copy number prior probability matrix

        KwArgs:
            allele_specific (boolean): calculate allele specific prior

        Copy number probability matrix is a symmetric matrix of prior probabilities of
        each copy number state.  The last row and column are the prior of seeing a 
        state not representable by the matrix.

        """

        self.N = N
        self.M = M
        self.allele_specific = allele_specific

        self.cn_max = cn_prob.shape[0] - 2
        self.cn_prob = cn_prob
        self.cn_prob_allele = cn_prob.sum(axis=1)

        self.transition_log_prob = -10.
        self.transition_model = 'step'

        self.min_length_likelihood = 10000
        
        self.divergence_probs = np.array([0.8, 0.2])

        self.prior_cn_scale = 5e-8


    def log_prior(self, cn):
        """ Evaluate log prior probability of segment copy number.
        
        Args:
            cn (numpy.array): copy number state of segments

        Returns:
            numpy.array: log prior per segment

        """

        cn = cn.copy().astype(int)

        if self.allele_specific:
            cn[np.any(cn > self.cn_max + 1, axis=(1, 2)),:,:] = self.cn_max + 1

            cn_minor, cn_major = cn.swapaxes(0, 2).swapaxes(1, 2)

            cn_prop = self.cn_prob[cn_minor, cn_major]

            lp = np.sum(np.log(cn_prop), axis=1) * self.l * self.prior_cn_scale

        else:
            cn[cn > self.cn_max + 1] = self.cn_max + 1

            cn_prop = self.cn_prob_allele[cn]

            lp = np.sum(np.log(cn_prop), axis=(1, 2)) * self.l * self.prior_cn_scale

        subclonal = (cn[:,1:,:].max(axis=1) != cn[:,1:,:].min(axis=1)) * 1
        subclonal_prob = self.divergence_probs[subclonal]

        lp += (np.sum(np.log(subclonal_prob), axis=1)) * self.l * self.prior_cn_scale

        lp[self.l < self.min_length_likelihood] = 0.0

        return lp


    def set_lengths(self, l):
        """ Set the observed lengths

        Args:
            l (numpy.array): observed lengths of segments

        """

        self.l = l



class HiddenMarkovModel(object):

    def __init__(self, N, M, emission, prior):
        """Create a copy number model.

        Args:
            N (int): number of segments
            M (int): number of clones including normal
            emission (ReadCountLikelihood): read count likelihood
            prior (CopyNumberPrior): copy number prior 

        """

        self.N = N
        self.M = M
        
        self.emission = emission
        self.prior = prior

        self.cn_max = prior.cn_max

        self.transition_log_prob = -10.
        self.transition_model = 'step'

        self.cn_dev_max = 1

        self.hmm_cns = None

        self.wildcard_cn_max = 2

        self.log_trans_mat = None


    def set_observed_data(self, x, l):
        """ Set observed data

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments

        """

        self.x = x
        self.l = l


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
                
            cn = np.array([np.ones(2)] + [base_cn] + list(subclone_cn))

            cn = np.array([cn] * self.N)

            yield cn


    def build_hmm_cns(self):
        """ Build a list of hmm copy number states.

        Returns:
            numpy.array: array of hmm copy number states

        """

        if self.hmm_cns is None:
            self.hmm_cns = np.array(list(self.build_cns(self.cn_max, self.cn_dev_max)))

        return self.hmm_cns


    def build_wildcard_cns(self):
        """ Build a list of wild card copy number states centered around posterior mode

        Returns:
            numpy.array: copy number states

        """

        h = self.emission.h
        p = self.emission.proportion_genotypeable_matrix()

        # Calculate the total haploid depth of the tumour clones
        h_t = (((self.x[:,0:2] + 1e-16) / (p[:,0:2] + 1e-16)).T / self.l).T

        for n, ell in zip(*np.where(np.isnan(h_t))):
            raise ProbabilityError('h_t is nan', n=n, x=x[n], l=l[n], h=h, p=p[n])

        # Calculate the dominant cn assuming no divergence
        dom_cn = (h_t - h[0]) / h[1:].sum()
        dom_cn = np.clip(dom_cn.round().astype(int), 0, int(1e6))

        # Do not allow competition with HMM states
        dom_cn[np.all(dom_cn < self.cn_max, axis=1),:] += 100

        # Reshape from N,L to N,M,L
        dom_cn = np.array([dom_cn] * (self.M-1))
        dom_cn = dom_cn.swapaxes(0, 1)

        wildcard_cns = list()

        for cn in self.build_cns(self.wildcard_cn_max, self.cn_dev_max, bounded=False):

            # Center around dominant cn prediction
            cn[:,1:,:] += dom_cn - self.wildcard_cn_max/2

            # Some copy number matrices may be negative, and should
            # be removed from consideration.  Adding a large number to
            # the negative copy number entries will do this.
            cn[cn < 0] += 100

            wildcard_cns.append(cn)

        return wildcard_cns


    def count_wildcard_cns(self):
        """ Count the number of wild card copy number states that will be used

        Returns:
            int: number of wild card copy number states

        """
        
        return len(list(self.build_cns(self.wildcard_cn_max, self.cn_dev_max, bounded=False)))


    def build_cn_states(self):
        """ Calculate the log posterior over copy number states.
        
        Returns:
            numpy.array: copy number states, shape (S,N,M,L)

        """

        cn_states = np.concatenate([self.build_hmm_cns(), self.build_wildcard_cns()])

        return cn_states


    def build_log_trans_mat(self):
        """ Build the log transition matrix.

        Returns:
            numpy.array: transition matrix

        """

        if self.log_trans_mat is not None:
            return self.log_trans_mat

        hmm_cns = self.build_hmm_cns()

        num_standard_cns = hmm_cns.shape[0]
        num_wildcard_cns = self.count_wildcard_cns()

        num_states = num_standard_cns + num_wildcard_cns

        self.log_trans_mat = np.zeros((num_states, num_states))

        if self.transition_model == 'step':

            self.log_trans_mat[0:num_standard_cns, 0:num_states] = self.transition_log_prob
            self.log_trans_mat[0:num_states, 0:num_standard_cns] = self.transition_log_prob

            self.log_trans_mat[xrange(num_states), xrange(num_states)] = 0.

        elif self.transition_model == 'geometric':

            def get_cn(idx):
                if idx < num_standard_cns:
                    return hmm_cns[idx]
                else:
                    return np.zeros(hmm_cns[0].shape) + self.cn_max + 1

            for idx_1 in xrange(num_states):
                cn_1 = get_cn(idx_1)

                for idx_2 in xrange(num_states):
                    cn_2 = get_cn(idx_2)

                    for ell in xrange(2):

                        cn_diff = np.absolute(cn_1[0,1:,ell] - cn_2[0,1:,ell]).sum()
                        self.log_trans_mat[idx_1, idx_2] = self.transition_log_prob * cn_diff

        else:
            raise ValueError('Unknown transition model {0}'.format(self.transition_model))

        self.log_trans_mat -= scipy.misc.logsumexp(self.log_trans_mat, axis=0)

        return self.log_trans_mat


    def emission_probabilities(self):
        """ Calculate the log posterior over copy number states.
        
        Returns:
            numpy.array: copy number states
            numpy.array: emission probabilities

        """

        N = self.x.shape[0]

        cn_states = self.build_cn_states()

        probs = list()

        for cn in cn_states:

            log_prob = self.emission.log_likelihood(self.x, self.l, cn) + self.prior.log_prior(cn)

            probs.append(log_prob)

        probs = np.array(probs)

        return cn_states, probs


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
        posteriors += np.finfo(np.float32).eps
        posteriors /= np.sum(posteriors, axis=1).reshape((-1, 1))
        posteriors = posteriors.T

        return log_prob, cn_states, posteriors


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

        state_sequence, log_prob = hmm_viterbi(frame_log_prob.shape[0], frame_log_prob.shape[1],
            log_start_prob, log_trans_mat, frame_log_prob)

        cn_state = cn_states[state_sequence,xrange(len(state_sequence))]

        return log_prob, cn_state


    def set_parameter(self, param, value):
        setattr(self.emission, param, value)


    def get_parameter_bounds(self, param):
        return self.emission.param_bounds[param]


    def get_parameter_is_global(self, param):
        return not self.emission.param_per_segment[param]


    def log_likelihood(self, state):
        return self.emission.log_likelihood(self.x, self.l, state)


    def log_likelihood_partial(self, param, state):
        return self.emission.log_likelihood_partial_param(self.x, self.l, state, param)



def decode_breakpoints_naive(self, cn, adjacencies, breakpoints):
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

            if breakend not in breakend_adj:
                continue

            n_adj, side_adj = breakend_adj[breakend]

            for allele in (0, 1):

                cn_self = cn[n,:,allele]
                cn_adj = cn[n_adj,:,allele]
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

