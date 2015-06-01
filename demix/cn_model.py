import itertools
import numpy as np
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

import demix.genome_graph
import demix.nb_overdispersion
import demix.utils


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


class CopyNumberModel(object):

    def __init__(self, adjacencies, breakpoints):
        """Create a copy number model.

        Args:
            adjacencies (list of tuple): ordered pairs of segments representing wild type adjacencies
            breakpoints (list of frozenset of tuple): list of pairs of segment/side pairs representing detected breakpoints

        Attributes:
            wt_adj (list of 'breakpoints'): list of 'breakpoint' representing wild type adjacencies
            tmr_adj (list of 'breakpoints'): list of 'breakpoint' representing detected tumour specific breakpoints

        A 'breakend' is represented as the tuple (('segment', 'allele'), 'side').

        A 'breakpoint' is represented as the frozenset (['breakend_1', 'breakend_2']).

        """

        self.wt_adj = set()
        self.wt_neighbour = dict()

        for seg_1, seg_2 in adjacencies:
            for allele in (0, 1):
                breakend_1 = ((seg_1, allele), 1)
                breakend_2 = ((seg_2, allele), 0)
                self.wt_adj.add(frozenset([breakend_1, breakend_2]))
                self.wt_neighbour[breakend_1] = breakend_2
                self.wt_neighbour[breakend_2] = breakend_1

        self.tmr_adj = set()
        for brkend_1, brkend_2 in breakpoints:
            for allele_1, allele_2 in itertools.product((0, 1), repeat=2):
                brkend_al_1 = ((brkend_1[0], allele_1), brkend_1[1])
                brkend_al_2 = ((brkend_2[0], allele_2), brkend_2[1])
                self.tmr_adj.add(frozenset([brkend_al_1, brkend_al_2]))

        self.transition_log_prob = -10.

        self.emission_model = 'negbin'
        self.total_cn = True

        self.e_step_method = 'independent'

        self.cn_max = 6
        self.cn_dev_max = 1

        self.hmm_cns = None

        self.wildcard_cn_max = 2

        self.log_trans_mat = None
        
        self.graph = None

        self.num_em_iter = 100
        self.mix_frac_resolution = 20

        self.p = None
        self.r = None

        self.major_cn_proportions = np.array([
            0.0011,
            0.3934,
            0.4239,
            0.1231,
            0.0262,
            0.0126,
            0.0066,
        ])

        self.minor_cn_proportions = np.array([
            0.2666,
            0.5613,
            0.1561,
            0.0052,
            0.0032,
            0.0015,
            0.0007,
        ])

        self.prior_cn_scale = 5e-8


    @staticmethod
    def expected_read_count_unopt(l, cn, h, p):
        """Calculate expected major, minor and total read counts.
        
        Unoptimized version.
        
        Args:
            l (numpy.array): length of segments
            cn (numpy.array): copy number matrices of normal and tumour populations
            h (numpy.array): haploid read depths, h[0] for normal
            p (numpy.array): proportion genotypable reads
        
        Returns:
            numpy.array: expected read depths
        """
        
        N = cn.shape[0]
        M = h.shape[0]
        K = 3

        x = np.zeros((N, K))
        
        for n in xrange(N):
            
            gamma = np.zeros((2,))
            
            for m in xrange(M):
                
                for ell in xrange(2):
                    
                    gamma[ell] += h[m] * cn[n,m,ell]
            
            x[n,0] = l[n] * p[n,0] * gamma[0]
            x[n,1] = l[n] * p[n,1] * gamma[1]
            x[n,2] = l[n] * p[n,2] * (gamma[0] + gamma[1])
            
        x += 1e-16

        for n, ell in zip(*np.where(x <= 0)):
            raise ProbabilityError('mu <= 0', n=n, cn=cn[n], l=l[n], h=h, p=p[n], mu=x[n])

        for n, ell in zip(*np.where(np.isnan(x))):
            raise ProbabilityError('mu is nan', n=n, cn=cn[n], l=l[n], h=h, p=p[n], mu=x[n])

        return x


    @staticmethod
    def expected_read_count(l, cn, h, p):
        """Calculate expected major, minor and total read counts.
        
        Args:
            l (numpy.array): length of segments
            cn (numpy.array): copy number matrices of normal and tumour populations
            h (numpy.array): haploid read depths, h[0] for normal
            p (numpy.array): proportion genotypable reads
        
        Returns:
            numpy.array: expected read depths
        """
        
        q = np.array([[1, 0], [0, 1], [1, 1]])
        
        gamma = np.sum(cn * np.vstack([h, h]).T, axis=-2)

        x1 = np.dot(q, gamma.T).T

        x2 = x1 * p
        
        x3 = (x2.T * l.T).T

        x3 += 1e-16

        for n, ell in zip(*np.where(x3 <= 0)):
            raise ProbabilityError('mu <= 0', n=n, cn=cn[n], l=l[n], h=h, p=p[n], mu=x3[n])

        for n, ell in zip(*np.where(np.isnan(x3))):
            raise ProbabilityError('mu is nan', n=n, cn=cn[n], l=l[n], h=h, p=p[n], mu=x3[n])

        return x3


    def log_likelihood_cn_negbin_unopt(self, x, mu, r):
        """ Calculate log likelihood of the copy number given read data, haploid
        depths and overdispersion.
        
        Unoptimized version.
        
        Args:
            x (numpy.array): measured major, minor, and total read counts
            mu (numpy.array): expected major, minor, and total read counts
            r (numpy.array): read depth overdispersion
        
        Returns:
            float: log likelihood per segment
            
        The pmf of the negative binomial is:
        
            C(x + r - 1, x) * p^x * (1-p)^r
            
        where p = mu / (r + mu), with mu the mean of the distribution.  The log
        likelihood with respect to mu is thus:
        
            log(G(x+r)) - log(x!) - log(G(r)) + x * log(p) + r * log(1 - p)
        """
        
        N = x.shape[0]
        K = (2, 3)[self.total_cn]

        ll = np.zeros((N,))

        for n in xrange(N):
            
            for k in xrange(K):
            
                nb_p = mu[n,k] / (r[k] + mu[n,k])
                
                ll[n] += gammaln(x[n,k] + r[k]) - gammaln(x[n,k] + 1) - gammaln(r[k])
                ll[n] += x[n,k] * np.log(nb_p) + r[k] * np.log(1 - nb_p)
                
        return ll


    def log_likelihood_cn_negbin(self, x, mu, r):
        """ Calculate log likelihood of the copy number given read data, haploid
        depths and overdispersion.
        
        Args:
            x (numpy.array): measured major, minor, and total read counts
            mu (numpy.array): expected major, minor, and total read counts
            r (numpy.array): read depth overdispersion
        
        Returns:
            float: log likelihood per segment
            
        The pmf of the negative binomial is:
        
            C(x + r - 1, x) * p^x * (1-p)^r
            
        where p = mu / (r + mu), with mu the mean of the distribution.  The log
        likelihood with respect to mu is thus:
        
            log(G(x+r)) - log(x!) - log(G(r)) + x * log(p) + r * log(1 - p)
        """
        
        if not self.total_cn:
            mu = mu[:,0:2]
            x = x[:,0:2]

        nb_p = mu / (r + mu)

        nb_p[nb_p < 0.] = 0.5
        nb_p[nb_p > 1.] = 0.5

        ll = gammaln(x + r) - gammaln(x + 1) - gammaln(r)
        ll += x * np.log(nb_p) + r * np.log(1 - nb_p)
        
        ll = np.sum(ll, axis=1)
        
        return ll


    def log_likelihood_cn_poisson_unopt(self, x, mu):
        """ Calculate poisson log likelihood of the copy number given read data, haploid
        depths and overdispersion.
        
        Unoptimized version.
        
        Args:
            x (numpy.array): measured major, minor, and total read counts
            mu (numpy.array): expected major, minor, and total read counts
        
        Returns:
            float: log likelihood per segment
            
        The pmf of the poisson is:
        
            mu^x * e^-mu / x!
            
        The log likelihood with respect to mu is thus:
        
            x * log(mu) - mu - log(x!)
        """
        
        N = x.shape[0]
        K = (2, 3)[self.total_cn]
        
        ll = np.zeros((N,))

        for n in xrange(N):
            
            for k in xrange(K):
                
                ll[n] += x[n,k] * np.log(mu[n,k]) - mu[n,k] - gammaln(x[n,k] + 1)
                
        return ll


    def log_likelihood_cn_poisson(self, x, mu):
        """ Calculate poisson log likelihood of the copy number given read data, haploid
        depths and overdispersion.
        
        Args:
            x (numpy.array): measured major, minor, and total read counts
            mu (numpy.array): expected major, minor, and total read counts
        
        Returns:
            float: log likelihood per segment
            
        The pmf of the negative binomial is:
        
            mu^x * e^-mu / x!
            
        The log likelihood with respect to mu is thus:
        
            x * log(mu) - mu - log(x!)
        """

        if not self.total_cn:
            mu = mu[:,0:2]
            x = x[:,0:2]

        mu[mu <= 0] = 1

        ll = x * np.log(mu) - mu - gammaln(x + 1)

        ll = np.sum(ll, axis=1)

        return ll


    def log_likelihood_cn_negbin_partial_h_unopt(self, x, l, cn, h, p, r):
        """ Calculate the partial derivative of the log likelihood of
        the copy number with respect to h, given read data, haploid
        depths and overdispersion.
        
        Unoptimized version.
        
        Args:
            x (numpy.array): measured major, minor, and total read counts
            l (numpy.array): length of segments
            cn (numpy.array): copy number matrices of normal and tumour populations
            h (numpy.array): haploid read depths
            p (numpy.array): proportion genotypable reads
            r (numpy.array): read depth overdispersion
        
        Returns:
            numpy.array: log likelihood derivative per segment per clone
            
        The partial derivative of the log pmf of the negative binomial with 
        respect to h_m is:
        
            sum_k (x[n,k] / mu[n,k] - (r[k] + x[n,k]) / (r[k] + mu[n,k])) * cn[n,m,ell] * q[ell,k] * p[n,k] * l[n]
            
        """

        mu = self.expected_read_count_unopt(l, cn, h, p)
        
        q = np.array([[1, 0, 1], [0, 1, 1]])
        
        N = x.shape[0]
        M = h.shape[0]
        K = (2, 3)[self.total_cn]

        partial_h = np.zeros((N, M))

        for n in xrange(N):

            for k in range(K):

                a = x[n,k] / mu[n,k] - (r[k] + x[n,k]) / (r[k] + mu[n,k])

                for m in xrange(M):
            
                    for ell in xrange(2):
                        
                        partial_h[n,m] += a * cn[n,m,ell] * q[ell,k] * p[n,k] * l[n]
                    
        return partial_h


    def log_likelihood_cn_negbin_partial_h(self, x, l, cn, h, p, r):
        """ Calculate the partial derivative of the log likelihood of
        the copy number with respect to h, given read data, haploid
        depths and overdispersion.
        
        Args:
            x (numpy.array): measured major, minor, and total read counts
            l (numpy.array): length of segments
            cn (numpy.array): copy number matrices of normal and tumour populations
            h (numpy.array): haploid read depths
            p (numpy.array): proportion genotypable reads
            r (numpy.array): read depth overdispersion
        
        Returns:
            numpy.array: log likelihood derivative per segment per clone
            
        The partial derivative of the log pmf of the negative binomial with 
        respect to h_m is:
        
            sum_k (x[n,k] / mu[n,k] - (r[k] + x[n,k]) / (r[k] + mu[n,k])) * cn[n,m,ell] * q[ell,k] * p[n,k] * l[n]

        """

        mu = self.expected_read_count(l, cn, h, p)

        q = np.array([[1, 0, 1], [0, 1, 1]])
        
        if not self.total_cn:
            mu = mu[:,0:2]
            x = x[:,0:2]
            p = p[:,0:2]
            q = q[:,0:2]
        
        a = x / mu - (r + x) / (r + mu)

        partial_h = np.einsum('...l,...jk,...kl,...l,...->...j', a, cn, q, p, l)
        
        return partial_h


    def log_likelihood_cn_poisson_partial_h_unopt(self, x, l, cn, h, p):
        """ Calculate the partial derivative of the poisson log likelihood of
        the copy number with respect to h, given read data, haploid
        depths and overdispersion.
        
        Unoptimized version.
        
        Args:
            x (numpy.array): measured major, minor, and total read counts
            l (numpy.array): length of segments
            cn (numpy.array): copy number matrices of normal and tumour populations
            h (numpy.array): haploid read depths
            p (numpy.array): proportion genotypable reads
        
        Returns:
            numpy.array: log likelihood derivative per segment per clone
            
        The partial derivative of the log pmf of the poisson with 
        respect to h_m is:
        
            sum_k (x[n,k] / mu[n,k] - 1) * cn[n,m,ell] * q[ell,k] * p[n,k] * l[n]
            
        """

        mu = self.expected_read_count_unopt(l, cn, h, p)
        
        q = np.array([[1, 0, 1], [0, 1, 1]])
        
        N = x.shape[0]
        M = h.shape[0]
        K = (2, 3)[self.total_cn]

        partial_h = np.zeros((N, M))
        
        for n in xrange(N):

            for k in range(K):

                a = x[n,k] / mu[n,k] - 1.

                for m in xrange(M):
            
                    for ell in xrange(2):
                        
                        partial_h[n,m] += a * cn[n,m,ell] * q[ell,k] * p[n,k] * l[n]
                    
        return partial_h


    def log_likelihood_cn_poisson_partial_h(self, x, l, cn, h, p):
        """ Calculate the partial derivative of the poisson log likelihood of
        the copy number with respect to h, given read data, haploid
        depths and overdispersion.
        
        Args:
            x (numpy.array): measured major, minor, and total read counts
            l (numpy.array): length of segments
            cn (numpy.array): copy number matrices of normal and tumour populations
            h (numpy.array): haploid read depths
            p (numpy.array): proportion genotypable reads
        
        Returns:
            numpy.array: log likelihood derivative per segment per clone
            
        The partial derivative of the log pmf of the poisson with 
        respect to h_m is:
        
            sum_k (x[n,k] / mu[n,k] - 1) * cn[n,m,ell] * q[ell,k] * p[n,k] * l[n]

        """

        mu = self.expected_read_count(l, cn, h, p)
        
        q = np.array([[1, 0, 1], [0, 1, 1]])
        
        if not self.total_cn:
            mu = mu[:,0:2]
            x = x[:,0:2]
            p = p[:,0:2]
            q = q[:,0:2]
        
        a = x / mu - 1.

        partial_h = np.einsum('...l,...jk,...kl,...l,...->...j', a, cn, q, p, l)
        
        return partial_h


    def log_likelihood_cn(self, x, l, cn, h):
        """ Evaluate log likelihood
        
        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            cn (numpy.array): copy number state of segments
            h (numpy.array): new haploid read depths to evaluate

        Returns:
            numpy.array: log likelihood per segment

        """

        if self.p is None:
            raise ValueError('must infer p')
        p = self.p

        if self.emission_model == 'negbin':
            if self.r is None:
                raise ValueError('must infer r')
            r = self.r

        mu = self.expected_read_count(l, cn, h, p)

        if self.emission_model == 'poisson':
            ll = self.log_likelihood_cn_poisson(x, mu)
        elif self.emission_model == 'negbin':
            ll = self.log_likelihood_cn_negbin(x, mu, r)

        for n in zip(*np.where(np.isnan(ll))):
            raise ProbabilityError('ll is nan', n=n, x=x[n], cn=cn[n], l=l[n], h=h, p=p[n], mu=mu[n])

        ll[np.where(np.any(cn < 0, axis=(-1, -2)))] = -np.inf
        
        return ll


    def log_prior_cn(self, l, cn):
        """ Evaluate log prior probability of segment copy number.
        
        Args:
            l (numpy.array): observed lengths of segments
            cn (numpy.array): copy number state of segments

        Returns:
            numpy.array: log prior per segment

        """

        cn = np.sort(cn, axis=2).astype(int)

        cn[cn > self.cn_max] = self.cn_max

        cn_minor, cn_major = cn.swapaxes(0, 2).swapaxes(1, 2)

        cn_minor_prop = self.minor_cn_proportions[cn_minor]
        cn_major_prop = self.major_cn_proportions[cn_major]

        lp = (np.sum(np.log(cn_minor_prop), axis=1) + np.sum(np.log(cn_major_prop), axis=1)) * l * self.prior_cn_scale
        
        self.divergence_probs = np.array([0.8, 0.2])

        subclonal = (cn[:,1:,:].max(axis=1) != cn[:,1:,:].min(axis=1)) * 1
        subclonal_prob = self.divergence_probs[subclonal]

        lp += (np.sum(np.log(subclonal_prob), axis=1)) * l * self.prior_cn_scale

        return lp


    def log_likelihood_cn_partial_h(self, x, l, cn, h):
        """ Evaluate partial derivative of log likelihood with respect to h
        
        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            cn (numpy.array): copy number state of segments
            h (numpy.array): new haploid read depths to evaluate

        Returns:
            numpy.array: partial derivative of log likelihood per segment per clone

        """

        if self.p is None:
            raise ValueError('must infer p')
        p = self.p

        if self.emission_model == 'negbin':
            if self.r is None:
                raise ValueError('must infer r')
            r = self.r

        mu = self.expected_read_count(l, cn, h, p)

        if self.emission_model == 'poisson':
            ll_partial_h = self.log_likelihood_cn_poisson_partial_h(x, l, cn, h, p)
        elif self.emission_model == 'negbin':
            ll_partial_h = self.log_likelihood_cn_negbin_partial_h(x, l, cn, h, p, r)

        for n in zip(*np.where(np.isnan(ll_partial_h))):
            raise ProbabilityError('ll derivative is nan', n=n, x=x[n], cn=cn[n], l=l[n], h=h, p=p[n], mu=mu[n])

        return ll_partial_h


    def evaluate_q(self, h, x, l, cns, resps):
        """ Evaluate q function, expected value of complete data log likelihood
        with respect to conditional given previous h
        
        Args:
            h (numpy.array): new haploid read depths to evaluate
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            cns (list): list of copy number states for which conditional was calculated
            resps (list): list of conditionals (responsibilities) for given copy number states

        Returns:
            numpy.array: expected value of complete data log likelihood
        """

        h[h < 0.] = 0.
        
        q_value = 0.0
        
        for cn, resp in zip(cns, resps):
            
            log_likelihood = resp * self.log_likelihood_cn(x, l, cn, h)

            log_likelihood[resp == 0] = 0

            for n in zip(*np.where(np.isnan(log_likelihood))):
                raise ProbabilityError('ll is nan', n=n, x=x[n], cn=cn[n], l=l[n], h=h, resp=resp[n])

            q_value += np.sum(log_likelihood)

        return -q_value


    def evaluate_q_derivative(self, h, x, l, cns, resps):
        """ Evaluate derivative of q function, expected complete data
        with respect to conditional given previous h
        
        Args:
            h (numpy.array): new haploid read depths to evaluate
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            cns (list): list of copy number states for which conditional was calculated
            resps (list): list of conditionals (responsibilities) for given copy number states
        
        Returns:
            numpy.array: partial derivative of expected value of complete data log likelihood

        """

        h[h < 0.] = 0.
        
        q_derivative = np.zeros(h.shape)

        for cn, resp in zip(cns, resps):

            log_likelihood_partial_h = (resp.T * self.log_likelihood_cn_partial_h(x, l, cn, h).T).T
            
            for n in zip(*np.where(np.isnan(log_likelihood_partial_h))):
                raise ProbabilityError('ll derivative is nan', n=n, x=x[n], cn=cn[n], l=l[n], h=h, resp=resp[n])

            q_derivative += np.sum(log_likelihood_partial_h.T, axis=-1)

        return -q_derivative


    def build_cns(self, N, M, cn_max, cn_dev_max, bounded=True):
        """ Generate a list of copy number states.

        Args:
            N (int): number of segments
            M (int): number of clones including normal
            cn_max (int): max copy number
            cn_dev_max (int): max clonal deviation of copy number

        KwArgs:
            bounded (bool): filter negative and greater than max copy numbers

        Yields:
            numpy.array: array of copy number states

        """

        base_cn_iter = itertools.product(np.arange(0.0, cn_max + 1.0, 1.0), repeat=2)
        dev_cn_iter = itertools.product(np.arange(-cn_dev_max, cn_dev_max + 1.0, 1.0), repeat=2*(M-2))

        for base_cn, dev_cn in itertools.product(base_cn_iter, dev_cn_iter):

            base_cn = np.array(base_cn).reshape(2)
            dev_cn = np.array(dev_cn).reshape((M-2,2))
            
            subclone_cn = dev_cn + base_cn
            
            if bounded and (np.any(subclone_cn < 0) or np.any(subclone_cn > cn_max)):
                continue
                
            cn = np.array([np.ones(2)] + [base_cn] + list(subclone_cn))

            cn = np.array([cn] * N)

            yield cn


    def build_hmm_cns(self, N, M):
        """ Build a list of hmm copy number states.

        Args:
            N (int): number of segments
            M (int): number of clones including normal

        Returns:
            numpy.array: array of hmm copy number states

        """

        if self.hmm_cns is None:
            self.hmm_cns = np.array(list(self.build_cns(N, M, self.cn_max, self.cn_dev_max)))

        return self.hmm_cns


    def build_wildcard_cns(self, x, l, h):
        """ Build a list of wild card copy number states centered around posterior mode

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            h (numpy.array): haploid read depth
        
        Returns:
            numpy.array: copy number states

        """

        N = x.shape[0]
        M = h.shape[0]

        # Calculate the total haploid depth of the tumour clones
        h_t = (((x[:,0:2] + 1e-16) / (self.p[:,0:2] + 1e-16)).T / l).T

        for n, ell in zip(*np.where(np.isnan(h_t))):
            raise ProbabilityError('h_t is nan', n=n, x=x[n], l=l[n], h=h, p=self.p[n])

        # Calculate the dominant cn assuming no divergence
        dom_cn = (h_t - h[0]) / h[1:].sum()
        dom_cn = np.clip(dom_cn.round().astype(int), 0, int(1e6))

        # Do not allow competition with HMM states
        dom_cn[np.all(dom_cn < self.cn_max, axis=1),:] += 100

        # Reshape from N,L to N,M,L
        dom_cn = np.array([dom_cn] * (M-1))
        dom_cn = dom_cn.swapaxes(0, 1)

        wildcard_cns = list()

        for cn in self.build_cns(N, M, self.wildcard_cn_max, self.cn_dev_max, bounded=False):

            # Center around dominant cn prediction
            cn[:,1:,:] += dom_cn - self.wildcard_cn_max/2

            # Some copy number matrices may be negative, and should
            # be removed from consideration.  Adding a large number to
            # the negative copy number entries will do this.
            cn[cn < 0] += 100

            wildcard_cns.append(cn)

        return wildcard_cns


    def count_wildcard_cns(self, M):
        """ Count the number of wild card copy number states that will be used

        Args:
            M (int): number of clones including normal

        Returns:
            int: number of wild card copy number states

        """
        
        return len(list(self.build_cns(0, M, self.wildcard_cn_max, self.cn_dev_max, bounded=False)))


    def build_log_trans_mat(self, M):
        """ Build the log transition matrix.

        Args:
            M (int): number of clones including normal

        Returns:
            numpy.array: transition matrix

        """

        if self.log_trans_mat is not None and M == self.log_trans_mat_M
            return self.log_trans_mat

        hmm_cns = self.build_hmm_cns(1, M)
        num_wildcard_cns = self.count_wildcard_cns(M)

        num_states = hmm_cns.shape[0] + num_wildcard_cns

        self.log_trans_mat = np.zeros((num_states, num_states))

        for idx_1, cn_1 in enumerate(hmm_cns):

            for idx_2, cn_2 in enumerate(hmm_cns):

                for ell in xrange(2):

                    cn_diff = cn_1[0,1:,ell] - cn_2[0,1:,ell]

                    if not np.all(cn_diff == 0):
                        self.log_trans_mat[idx_1, idx_2] += self.transition_log_prob

        self.log_trans_mat -= scipy.misc.logsumexp(self.log_trans_mat, axis=0)

        self.log_trans_mat_M = M

        return self.log_trans_mat


    def emission_probabilities(self, x, l, h):
        """ Calculate the log posterior over copy number states.

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            h (numpy.array): new haploid read depths to evaluate
        
        Returns:
            numpy.array: copy number states
            numpy.array: emission probabilities

        """

        N = x.shape[0]
        M = h.shape[0]

        cns = list()
        probs = list()

        for cn in self.build_hmm_cns(N, M):

            log_prob = self.log_likelihood_cn(x, l, cn, h) + self.log_prior_cn(l, cn)

            cns.append(cn)
            probs.append(log_prob)

        for cn in self.build_wildcard_cns(x, l, h):

            log_prob = self.log_likelihood_cn(x, l, cn, h) + self.log_prior_cn(l, cn)

            cns.append(cn)
            probs.append(log_prob)

        cns = np.array(cns)
        probs = np.array(probs)

        return cns, probs


    def e_step_independent(self, x, l, h):
        """ E Step: Calculate responsibilities for copy number states independently
        
        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            h (numpy.array): new haploid read depths to evaluate
        
        Returns:
            numpy.array: copy number states
            numpy.array: unconditional posteriors
            numpy.array: log posterior

        """

        cns, probs = self.emission_probabilities(x, l, h)

        norm = scipy.misc.logsumexp(probs, axis=0)
        log_posterior = np.sum(norm)

        probs -= norm
        probs = np.exp(probs)

        return cns, probs, log_posterior


    def e_step_viterbi(self, x, l, h):
        """ Calculate the viterbi path.

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            h (numpy.array): new haploid read depths to evaluate
        
        Returns:
            numpy.array: copy number states
            numpy.array: posterior marginals
            numpy.array: log posterior

        """

        M = h.shape[0]

        cns, probs = self.emission_probabilities(x, l, h)

        log_start_prob = np.zeros((cns.shape[0],))
        log_start_prob -= scipy.misc.logsumexp(log_start_prob)
        log_trans_mat = self.build_log_trans_mat(M)
        frame_log_prob = probs.T

        state_sequence, log_prob = hmm_viterbi(frame_log_prob.shape[0], frame_log_prob.shape[1],
                                               log_start_prob, log_trans_mat, frame_log_prob)

        cns = np.array([cns[state_sequence,xrange(len(state_sequence)),:,:]])
        posteriors = np.zeros(cns.shape[0:2])

        return cns, posteriors, log_prob
 

    def e_step_forwardbackward(self, x, l, h):
        """ Calculate the forward backward posterior marginals.

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            h (numpy.array): new haploid read depths to evaluate
        
        Returns:
            numpy.array: copy number states
            numpy.array: posterior marginals
            numpy.array: log posterior

        """

        M = h.shape[0]

        cns, probs = self.emission_probabilities(x, l, h)

        log_start_prob = np.zeros((cns.shape[0],))
        log_trans_mat = self.build_log_trans_mat(M)
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

        return cns, posteriors, log_prob


    def e_step_genomegraph(self, x, l, h):
        """ Calculate the genome graph based optimal copy number.

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            h (numpy.array): new haploid read depths to evaluate
        
        Returns:
            numpy.array: copy number states
            numpy.array: posterior marginals
            numpy.array: log posterior

        """

        if self.graph is None:

            cns, posteriors, log_prob = self.e_step_viterbi(x, l, h)

            opt = posteriors.argmax(axis=0)
            cn = cns[opt,xrange(len(opt)),:,:]

            self.graph = demix.genome_graph.GenomeGraph(self, x, l, cn, self.wt_adj, self.tmr_adj)

        cn, log_prob = self.graph.optimize(h)

        cns = np.array([cn])
        posteriors = np.zeros(cns.shape[0:2])

        return cns, posteriors, log_prob


    def e_step(self, x, l, h):
        """ E Step: Calculate responsibilities for copy number states
        
        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            h (numpy.array): new haploid read depths to evaluate
        
        Returns:
            numpy.array: copy number states
            numpy.array: responsibilities
            numpy.array: log posterior

        """

        if self.e_step_method == 'independent':
            return self.e_step_independent(x, l, h)
        elif self.e_step_method == 'viterbi':
            return self.e_step_viterbi(x, l, h)
        elif self.e_step_method == 'forwardbackward':
            return self.e_step_forwardbackward(x, l, h)
        elif self.e_step_method == 'genomegraph':
            return self.e_step_genomegraph(x, l, h)


    class OptimizeException(Exception):
        pass

    def m_step(self, x, l, h_init, cns, resps):
        """ M Step.  Maximize expected complete data log likelihood given copy number state responsibilities.

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            h_init (numpy.array): haploid read depth initial guess
            cns (list): list of copy number states for which conditional was calculated
            resps (list): list of conditionals (responsibilities) for given copy number states
        
        Returns:
            numpy.array: optimal haploid read depth, maximized value of complete data log likelihood

        """

        M = h_init.shape[0]

        result = scipy.optimize.minimize(self.evaluate_q, h_init,
                                         method='L-BFGS-B',
                                         jac=self.evaluate_q_derivative,
                                         args=(x, l, cns, resps),
                                         bounds=((0., 1.),)*M,
                                         options={'ftol':1e-3})

        if not result.success:
            raise CopyNumberModel.OptimizeException(result.message)

        h = result.x

        return h


    def decode_breakpoints_naive(self, cn):
        """ Naive decoding of breakpoint copy number.  Finds most likely set of copy numbers given h.

        Args:
            cn (numpy.array): copy number matrix

        Returns:
            dict: dictionary of breakpoint copy number
                keys: frozenset of breakends
                values: copy number matrix

        """

        brk_cn = dict()

        for breakpoint in self.tmr_adj:

            cn_diffs = list()

            for breakend in breakpoint:

                cn_self = cn[breakend[0][0],:,breakend[0][1]]

                breakend_adj = self.wt_neighbour.get(breakend, np.zeros(cn_self.shape))

                cn_adj = cn[breakend_adj[0][0],:,breakend_adj[0][1]]

                cn_diff = cn_self - cn_adj

                cn_diffs.append(cn_diff)

            brk_cn[breakpoint] = np.maximum(cn_diffs[0], cn_diffs[1])

        return brk_cn


    def decode(self, x, l, h):
        """ Decode Step.  Find most likely set of copy numbers given h.

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            h (numpy.array): new haploid read depths to evaluate

        Returns:
            numpy.array: copy number matrix
            dict: dictionary of breakpoint copy number
                keys: frozenset of breakends
                values: copy number matrix

        """

        if self.e_step_method == 'genomegraph':

            cns, resps, log_posterior = self.e_step_genomegraph(x, l, h)

            opt = resps.argmax(axis=0)
            cn = cns[opt,xrange(len(opt)),:,:]

            brk_cn = self.graph.breakpoint_copy_number

            return cn, brk_cn

        if self.e_step_method == 'independent':
            cns, resps, log_posterior = self.e_step_independent(x, l, h)
        elif self.e_step_method == 'viterbi' or self.e_step_method == 'forwardbackward':
            cns, resps, log_posterior = self.e_step_viterbi(x, l, h)

        opt = resps.argmax(axis=0)
        cn = cns[opt,xrange(len(opt)),:,:]

        brk_cn = self.decode_breakpoints_naive(cn)

        return cn, brk_cn


    def optimize_h(self, x, l, h_init):
        """ Optimize h given an initial estimate.

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments
            h_init (numpy.array): initial haploid read depths

        Returns:
            numpy.array: haploid read depths
            float: log posterior
            bool: converged

        """

        h = h_init

        print repr(list(h))

        log_posterior_prev = None

        converged = False

        for _ in xrange(self.num_em_iter):

            # Maximize Log likelihood with respect to copy number
            cns, resps, log_posterior = self.e_step(x, l, h)

            # Maximize Log likelihood with respect to haploid read depth
            h = self.m_step(x, l, h, cns, resps)

            print repr(list(h)), log_posterior

            if log_posterior_prev is not None and abs(log_posterior_prev - log_posterior) < 1e-3:
                converged = True
                break

            log_posterior_prev = log_posterior

        return h, log_posterior, converged


    def infer_p(self, x):
        """ Infer proportion of genotypable reads.

        Args:
            x (numpy.array): observed major, minor, and total read counts

        """

        self.p = x[:,0:2].sum(axis=1).astype(float) / (x[:,2].astype(float) + 1.0)
        self.p = np.vstack([self.p, self.p, np.ones(self.p.shape)]).T


    def infer_r(self, x, l):
        """ Use max likelihood to infer negative binomial overdispersion.

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments

        """

        K = (2, 3)[self.total_cn]
        self.r = np.array([demix.nb_overdispersion.infer_disperion(x[:,k], l*self.p[:,k]) for k in xrange(K)])


    def infer_offline_parameters(self, x, l):
        """ Offline parameter inference.

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments

        """
        
        self.infer_p(x)

        if self.emission_model == 'negbin':
            self.infer_r(x, l)


    def candidate_h(self, x, l, ax=None):
        """ Use a GMM to identify candidate haploid read depths

        Args:
            x (numpy.array): observed major, minor, and total read counts
            l (numpy.array): observed lengths of segments

        Kwargs:
            ax (matplotlib.axis): optional axis for plotting major/minor/total read depth

        Returns:
            list of tuple: candidate haploid normal and tumour read depths

        """

        p = self.p

        is_filtered = (l > 0) & np.all(p > 0, axis=1)
        x = x[is_filtered,:]
        l = l[is_filtered]
        p = p[is_filtered,:]

        rd = ((x.T / p.T) / l.T)

        rd_min = np.minimum(rd[0], rd[1])
        rd_max = np.maximum(rd[0], rd[1])

        # Cluster minor read depths using kmeans
        rd_min_samples = demix.utils.weighted_resample(rd_min, l)
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

            h_candidates.append(np.array([h_normal, h_tumour]))

        if ax is not None:
            self.plot_depth(ax, x, l, p, annotated=means)

        # Maximum of 3 clones

        mix_iter = itertools.product(xrange(1, self.mix_frac_resolution+1), repeat=2)

        for mix in mix_iter:
            
            if mix != tuple(reversed(sorted(mix))):
                continue
            if sum(mix) != self.mix_frac_resolution:
                continue
            
            mix = np.array(mix) / float(self.mix_frac_resolution)

            for h_tumour in h_tumour_candidates:

                h = np.array([h_normal] + list(h_tumour*mix))

                h_candidates.append(h)

        return h_candidates


    def plot_depth(self, ax, x, l, p, annotated=()):
        """ 
        """

        rd = ((x.T / p.T) / l.T)
        rd.sort(axis=0)

        depth_max = np.percentile(rd[2], 95)
        cov = 0.0000001

        demix.utils.filled_density_weighted(ax, rd[0], l, 'blue', 0.5, 0.0, depth_max, cov)
        demix.utils.filled_density_weighted(ax, rd[1], l, 'red', 0.5, 0.0, depth_max, cov)
        demix.utils.filled_density_weighted(ax, rd[2], l, 'grey', 0.5, 0.0, depth_max, cov)

        ylim = ax.get_ylim()
        for depth in annotated:
            ax.plot([depth, depth], [0, 1e16], 'g', lw=2)
        ax.set_ylim(ylim)



