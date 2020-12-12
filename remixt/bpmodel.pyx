# cython: profile=False
# cython: initializedcheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.math cimport exp, log, fabs, lgamma, isnan
import numpy as np
import scipy
import itertools
cimport numpy as np
cimport cython

np.import_array()

cdef np.float64_t _NINF = -np.inf
cdef np.float64_t _PI = np.pi


cdef np.float64_t _max(np.float64_t[:] values):
    cdef np.float64_t vmax = _NINF
    for i in range(values.shape[0]):
        if values[i] > vmax:
            vmax = values[i]
    return vmax


cdef np.float64_t _max2(np.float64_t[:, :] values):
    cdef np.float64_t vmax = _NINF
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if values[i, j] > vmax:
                vmax = values[i, j]
    return vmax


cdef np.float64_t _max3(np.float64_t[:, :, :] values):
    cdef np.float64_t vmax = _NINF
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            for k in range(values.shape[2]):
                if values[i, j, k] > vmax:
                    vmax = values[i, j, k]
    return vmax


cdef int _argmax(np.float64_t[:] values):
    cdef np.float64_t vmax = _NINF
    cdef int imax = 0
    for i in range(values.shape[0]):
        if values[i] > vmax:
            vmax = values[i]
            imax = i
    return imax


cdef void _argmax2(np.float64_t[:, :] values, np.int64_t[:] indices):
    cdef np.float64_t vmax = _NINF
    indices[0] = 0
    indices[1] = 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if values[i, j] > vmax:
                vmax = values[i, j]
                indices[0] = i
                indices[1] = j


cdef np.float64_t _logsumpair(np.float64_t x, np.float64_t y):
    cdef np.float64_t vmax = max(x, y)
    cdef np.float64_t power_sum = exp(x-vmax) + exp(y-vmax)

    return log(power_sum) + vmax


cdef np.float64_t _logsum(np.float64_t[:] X):
    cdef np.float64_t vmax = _max(X)
    cdef np.float64_t power_sum = 0

    for i in range(X.shape[0]):
        power_sum += exp(X[i]-vmax)

    return log(power_sum) + vmax


cdef np.float64_t _logsum2(np.float64_t[:, :] X):
    cdef np.float64_t vmax = _max2(X)
    cdef np.float64_t power_sum = 0

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            power_sum += exp(X[i, j]-vmax)

    return log(power_sum) + vmax


cdef np.float64_t _logsum3(np.float64_t[:, :, :] X):
    cdef np.float64_t vmax = _max3(X)
    cdef np.float64_t power_sum = 0

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                power_sum += exp(X[i, j, k]-vmax)

    return log(power_sum) + vmax


cdef np.float64_t _entropy(np.float64_t[:] X):
    cdef np.float64_t entropy = 0

    for i in range(X.shape[0]):
        if X[i] > 0.:
            entropy += X[i] * log(X[i])

    return entropy


cdef np.float64_t _exp_normalize(np.float64_t[:] Y, np.float64_t[:] X):
    cdef np.float64_t normalize = _logsum(X)
    for i in range(X.shape[0]):
        Y[i] = exp(X[i] - normalize)
    normalize = 0.
    for i in range(X.shape[0]):
        normalize += Y[i]
    for i in range(X.shape[0]):
        Y[i] /= normalize


cdef np.float64_t _exp_normalize2(np.float64_t[:, :] Y, np.float64_t[:, :] X):
    cdef np.float64_t normalize = _logsum2(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i, j] = exp(X[i, j] - normalize)
    normalize = 0.
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            normalize += Y[i, j]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i, j] /= normalize


cdef np.float64_t _exp_normalize3(np.float64_t[:, :, :] Y, np.float64_t[:, :, :] X):
    cdef np.float64_t normalize = _logsum3(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                Y[i, j, k] = exp(X[i, j, k] - normalize)
    normalize = 0.
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                normalize += Y[i, j, k]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                Y[i, j, k] /= normalize


cdef np.float64_t digamma(np.float64_t x) except *:
    """
      Purpose:

        DIGAMMA calculates DIGAMMA ( X ) = d ( LOG ( GAMMA ( X ) ) ) / dX

      Licensing:

        This code is distributed under the GNU LGPL license. 

      Modified:

        20 March 2016

      Author:

        Original FORTRAN77 version by Jose Bernardo.
        Adapted from C version by John Burkardt.

      Reference:

        Jose Bernardo,
        Algorithm AS 103:
        Psi ( Digamma ) Function,
        Applied Statistics,
        Volume 25, Number 3, 1976, pages 315-317.

      Parameters:

        Input, double X, the argument of the digamma function.
        0 < X.

        Output, int *IFAULT, error flag.
        0, no error.
        1, X <= 0.

        Output, double DIGAMMA, the value of the digamma function at X.
    """
    cdef np.float64_t c = 8.5
    cdef np.float64_t euler_mascheroni = 0.57721566490153286060
    cdef np.float64_t r
    cdef np.float64_t value
    cdef np.float64_t x2

    # Check the input.
    if ( x <= 0.0 ):
        raise ValueError('x <= 0.0 for x: {}'.format(x))

    # Use approximation for small argument.
    if ( x <= 0.000001 ):
        value = - euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x
        return value

    # Reduce to DIGAMA(X + N).
    value = 0.0
    x2 = x
    while ( x2 < c ):
        value = value - 1.0 / x2
        x2 = x2 + 1.0

    # Use Stirling's (actually de Moivre's) expansion.
    r = 1.0 / x2
    value = value + log ( x2 ) - 0.5 * r

    r = r * r

    value = ( value 
        - r * ( 1.0 / 12.0 
        - r * ( 1.0 / 120.0 
        - r * ( 1.0 / 252.0 
        - r * ( 1.0 / 240.0
        - r * ( 1.0 / 132.0 ) ) ) ) ) )

    return value;


cdef np.float64_t negbin_log_likelihood(np.float64_t x, np.float64_t mu, np.float64_t r) except *:
    """ Calculate negative binomial read count log likelihood.
    
    Args:
        x (float): observed read counts
        mu (float): expected read counts
        r (float): over-dispersion
    
    Returns:
        float: log likelihood per segment
        
    The pmf of the negative binomial is:
    
        C(x + r - 1, x) * p^x * (1-p)^r
        
    where p = mu / (r + mu), with mu the mean of the distribution.  The log
    likelihood is thus:
    
        log(G(x+r)) - log(G(x+1)) - log(G(r)) + x * log(p) + r * log(1 - p)
    """

    cdef np.float64_t nb_p, ll

    nb_p = mu / (r + mu)

    if nb_p < 0. or nb_p > 1.:
        nb_p = 0.5

    ll = (lgamma(x + r) - lgamma(x + 1) - lgamma(r)
        + x * log(nb_p) + r * log(1 - nb_p))

    if isnan(ll):
        raise ValueError('ll is nan for x: {}, mu: {}, r: {}'.format(x, mu, r))

    return ll


cdef np.float64_t negbin_log_likelihood_partial_mu(np.float64_t x, np.float64_t mu, np.float64_t r) except *:
    """ Calculate the partial derivative of the negative binomial read count
    log likelihood with respect to mu

    Args:
        x (numpy.array): observed read counts
        mu (numpy.array): expected read counts
        r (float): over-dispersion
    
    Returns:
        numpy.array: log likelihood derivative per segment
        
    The partial derivative of the log pmf of the negative binomial with 
    respect to mu is:
    
        x / mu - (r + x) / (r + mu)

    """

    cdef np.float64_t partial_mu

    partial_mu = x / mu - (r + x) / (r + mu)

    if isnan(partial_mu):
        raise ValueError('partial_mu is nan for x: {}, mu: {}, r: {}'.format(x, mu, r))

    return partial_mu


cdef np.float64_t betabin_log_likelihood(np.float64_t k, np.float64_t n, np.float64_t p, np.float64_t M) except *:
    """ Calculate beta binomial allele count log likelihood.
    
    Args:
        k (float): observed minor allelic read counts
        n (float): observed total allelic read counts
        p (float): expected minor allele fraction
        M (float): over-dispersion
    
    Returns:
        float: log likelihood per segment
        
    The pmf of the beta binomial is:
    
        C(n, k) * B(k + M * p, n - k + M * (1 - p)) / B(M * p, M * (1 - p))

    Where p=mu[1]/(mu[0]+mu[1]), k=x[1], n=x[0]+x[1], and M is the over-dispersion
    parameter.

    The log likelihood is thus:
    
        log(G(n+1)) - log(G(k+1)) - log(G(n-k+1))
            + log(G(k + M * p)) + log(G(n - k + M * (1 - p)))
            - log(G(n + M))
            - log(G(M * p)) - log(G(M * (1 - p)))
            + log(G(M))

    """

    cdef np.float64_t ll

    if p <= 0. or (1 - p) <= 0.:
        raise ValueError('p <= 0 or (1 - p) <= 0. for p: {}'.format(p))

    ll = (lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        + lgamma(k + M * p) + lgamma(n - k + M * (1 - p))
        - lgamma(n + M)
        - lgamma(M * p) - lgamma(M * (1 - p))
        + lgamma(M))

    if isnan(ll):
        raise ValueError('ll is nan for k: {}, n: {}, p: {}, M: {}'.format(k, n, p, M))

    return ll


cdef np.float64_t betabin_log_likelihood_partial_p(np.float64_t k, np.float64_t n, np.float64_t p, np.float64_t M) except *:
    """ Calculate the partial derivative of the beta binomial allele count
    log likelihood with respect to p

    Args:
        k (float): observed minor allelic read counts
        n (float): observed total allelic read counts
        p (float): expected minor allele fraction
        M (float): over-dispersion
    
    Returns:
        numpy.array: log likelihood derivative per segment per clone

    The log likelihood of the beta binomial in terms of p, and with beta
    functions expanded is:

        log(G(k + M * p))
            + log(G(n - k + M * (1 - p)))
            - log(G(M * p))
            - log(G(M * (1 - p)))

    The partial derivative of the log pmf of the beta binomial with 
    respect to p is:

        M * digamma(k + M * p)
            + (-M) * digamma(n - k + M * (1 - p))
            - M * digamma(M * p)
            - (-M) * digamma(M * (1 - p))

    """

    cdef np.float64_t partial_p

    if p <= 0. or (1 - p) <= 0.:
        raise ValueError('p <= 0 or (1 - p) <= 0. for p: {}'.format(p))

    partial_p = (M * digamma(k + M * p)
        + (-M) * digamma(n - k + M * (1 - p))
        - M * digamma(M * p)
        - (-M) * digamma(M * (1 - p)))

    if isnan(partial_p):
        raise ValueError('partial_p is nan for k: {}, n: {}, p: {}, M: {}'.format(k, n, p, M))

    return partial_p


cdef class RemixtModel:
    cdef public int num_clones
    cdef public int num_segments
    cdef public int num_breakpoints
    cdef public int num_alleles
    cdef public int cn_max
    cdef public bint normal_contamination
    cdef public int num_cn_states
    cdef public np.int64_t[:, :, :, :] cn_states
    cdef public np.int64_t[:, :, :] cn_states_total
    cdef public int num_brk_states
    cdef public np.int64_t[:, :] brk_states
    cdef public np.int64_t[:, :] num_alleles_subclonal
    cdef public np.int64_t[:, :] is_hdel
    cdef public np.int64_t[:, :] is_loh

    cdef public np.int64_t[:] is_telomere
    cdef public np.int64_t[:] breakpoint_idx
    cdef public np.int64_t[:] breakpoint_orient
    cdef public np.int64_t[:] breakpoint_side
    cdef public np.float64_t transition_penalty
    cdef public np.float64_t divergence_weight

    cdef public np.float64_t[:, :] p_breakpoint

    cdef public np.float64_t hmm_log_norm_const
    cdef public np.float64_t[:, :] framelogprob
    cdef public np.float64_t[:, :, :] log_transmat
    cdef public np.float64_t[:, :, :] cached_log_transmat
    cdef public np.float64_t[:, :] posterior_marginals
    cdef public np.float64_t[:, :, :] joint_posterior_marginals

    cdef public np.float64_t[:, :] p_allele_swap
    cdef public np.float64_t[:, :] p_outlier_total
    cdef public np.float64_t[:, :] p_outlier_allele

    cdef public np.float64_t prior_outlier_total
    cdef public np.float64_t prior_outlier_allele

    cdef public np.float64_t[:] l
    cdef public np.float64_t[:] x
    cdef public np.float64_t[:, :] y
    cdef public np.int64_t[:] total_likelihood_mask
    cdef public np.int64_t[:] allele_likelihood_mask

    cdef public np.float64_t[:] h

    cdef public np.float64_t negbin_r_0
    cdef public np.float64_t negbin_r_1
    cdef public np.float64_t negbin_hdel_mu
    cdef public np.float64_t negbin_hdel_r_0
    cdef public np.float64_t negbin_hdel_r_1

    cdef public np.float64_t betabin_M_0
    cdef public np.float64_t betabin_M_1
    cdef public np.float64_t betabin_loh_p
    cdef public np.float64_t betabin_loh_M_0
    cdef public np.float64_t betabin_loh_M_1

    cdef public int transition_model

    cdef np.float64_t[:] _p_d
    cdef np.float64_t[:] _allele_cn_change

    def __cinit__(self,
        int num_clones,
        int num_segments,
        int num_breakpoints,
        bint normal_contamination,
        np.ndarray[np.int64_t, ndim=4] cn_states,
        np.ndarray[np.int64_t, ndim=2] brk_states,
        np.ndarray[np.float64_t, ndim=1] h_init,
        np.ndarray[np.float64_t, ndim=1] l,
        np.ndarray[np.float64_t, ndim=1] x,
        np.ndarray[np.float64_t, ndim=2] y,
        np.ndarray[np.int64_t, ndim=1] is_telomere,
        np.ndarray[np.int64_t, ndim=1] breakpoint_idx,
        np.ndarray[np.int64_t, ndim=1] breakpoint_orient,
        np.float64_t transition_penalty,
        np.float64_t divergence_weight):

        self.num_clones = num_clones
        self.num_segments = num_segments
        self.num_breakpoints = num_breakpoints
        self.normal_contamination = normal_contamination
        self.cn_states = cn_states
        self.brk_states = brk_states
        self.h = h_init
        self.l = l
        self.x = x
        self.y = y
        self.num_alleles = 2
        self.cn_max = max(cn_states.max(), brk_states.max())
        self.num_cn_states = self.cn_states.shape[1]
        self.num_brk_states = self.brk_states.shape[0]

        self.total_likelihood_mask = np.ones((self.num_segments,), dtype=np.int64)
        self.allele_likelihood_mask = np.ones((self.num_segments,), dtype=np.int64)

        # Create total states for convenience
        self.cn_states_total = np.zeros((self.num_segments, self.num_cn_states, self.num_clones), dtype=np.int64)
        for n in range(self.num_segments):
            for s in range(self.num_cn_states):
                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        self.cn_states_total[n, s, m] += self.cn_states[n, s, m, ell]

        # Create is subclonal and cn state indicators for convenience
        self.num_alleles_subclonal = np.sum((np.asarray(self.cn_states)[:, :, 1:, :].max(axis=-2) != np.asarray(self.cn_states)[:, :, 1:, :].min(axis=-2)), axis=-1)
        self.is_hdel = np.all(np.asarray(self.cn_states) == 0, axis=(-2, -1)) * 1
        self.is_loh = np.any(np.asarray(self.cn_states).sum(axis=-2) == 0, axis=-1) * 1

        if ((cn_states.shape[0] != num_segments) or (cn_states.shape[1] != self.num_cn_states) or
            (cn_states.shape[2] != num_clones) or (cn_states.shape[3] != self.num_alleles)):
            raise ValueError('cn_states must have shape (num_segments, num_cn_states, num_clones, num_alleles)')

        if ((brk_states.shape[0] != self.num_brk_states) or (brk_states.shape[1] != num_clones)):
            raise ValueError('cn_states must have shape (num_brk_states, num_clones)')

        if h_init.shape[0] != num_clones:
            raise ValueError('h must have length equal to num_clones')

        if is_telomere.shape[0] != num_segments:
            raise ValueError('is_telomere must have length equal to num_segments')

        if breakpoint_idx.shape[0] != num_segments:
            raise ValueError('breakpoint_idx must have length equal to num_segments')

        if breakpoint_orient.shape[0] != num_segments:
            raise ValueError('breakpoint_orient must have length equal to num_segments')

        if breakpoint_idx.max() + 1 != num_breakpoints:
            raise ValueError('breakpoint_idx must have maximum of num_breakpoints positive indices')

        self.is_telomere = is_telomere
        self.breakpoint_idx = breakpoint_idx
        self.breakpoint_orient = breakpoint_orient
        self.transition_penalty = fabs(transition_penalty)
        self.divergence_weight = fabs(divergence_weight)
        
        self.breakpoint_side = np.zeros((self.num_segments,), dtype=np.int64)
        sides = np.zeros((self.num_breakpoints,), dtype=np.int64)
        for n in range(self.num_segments):
            if self.breakpoint_idx[n] < 0:
                continue
                
            self.breakpoint_side[n] = sides[self.breakpoint_idx[n]]
            sides[self.breakpoint_idx[n]] += 1

        # Initialize to favour single copy change
        self.p_breakpoint = np.zeros((self.num_breakpoints, self.num_brk_states))
        for k in range(self.num_breakpoints):
            for s_b in range(self.num_brk_states):
                brk = np.array(self.brk_states[s_b])
                if brk.max() > 1:
                    continue
                self.p_breakpoint[k, s_b] = 1.
        self.p_breakpoint /= np.sum(self.p_breakpoint, axis=-1)[:, np.newaxis]

        self.hmm_log_norm_const = 0.
        self.framelogprob = np.ones((self.num_segments, self.num_cn_states))
        self.log_transmat = np.zeros((self.num_segments - 1, self.num_cn_states, self.num_cn_states))
        self.cached_log_transmat = np.zeros((self.num_segments - 1, self.num_cn_states, self.num_cn_states))
        self.posterior_marginals = np.zeros((self.num_segments, self.num_cn_states))
        self.joint_posterior_marginals = np.zeros((self.num_segments - 1, self.num_cn_states, self.num_cn_states))

        # Initialize to something valid
        self.posterior_marginals[:] = 1.
        self.posterior_marginals /= np.sum(self.posterior_marginals, axis=-1)[:, np.newaxis]
        self.joint_posterior_marginals[:] = 1.
        self.joint_posterior_marginals /= np.sum(self.joint_posterior_marginals, axis=(-2, -1))[:, np.newaxis, np.newaxis]

        # Indicator for allele swapping
        self.p_allele_swap = np.ones((self.num_segments, 2)) * 0.5

        # Indicator for outlier states, and prior
        self.prior_outlier_total = 0.01
        self.prior_outlier_allele = 0.01

        self.p_outlier_total = np.ones((self.num_segments, 2))
        self.p_outlier_allele = np.ones((self.num_segments, 2))

        # Initialize to prior
        self.p_outlier_total[:, 0] = 1. - self.prior_outlier_total
        self.p_outlier_total[:, 1] = self.prior_outlier_total

        self.p_outlier_allele[:, 0] = 1. - self.prior_outlier_allele
        self.p_outlier_allele[:, 1] = self.prior_outlier_allele

        # Initialize likelihood parameters
        self.negbin_r_0 = 500.
        self.negbin_r_1 = 10.
        self.negbin_hdel_mu = 1e-5
        self.negbin_hdel_r_0 = 10.
        self.negbin_hdel_r_1 = 1.

        self.betabin_M_0 = 500.
        self.betabin_M_1 = 10.
        self.betabin_loh_p = 1e-3
        self.betabin_loh_M_0 = 10.
        self.betabin_loh_M_1 = 1.

        # Temporary buffers
        self._p_d = np.zeros(((self.cn_max + 1) * 2,))
        self._allele_cn_change = np.zeros((2,))

        # Cached transmat expecation for elbo calc, updated with p_breakpoint
        self.calculate_log_transmat(self.cached_log_transmat)

    @cython.profile(False)
    cdef inline np.float64_t calc_transition(self, np.float64_t cn_diff):
        """ Calculate transition function for a copy number difference.
        """
        if self.transition_model == 0:
            return fabs(cn_diff)
        elif self.transition_model == 1:
            if cn_diff == 0:
                return 0.
            else:
                return 1.

    @cython.wraparound(True)
    cdef void add_log_breakpoint_p_expectation_cn(self, np.float64_t[:] log_breakpoint_p, np.float64_t[:, :] p_cn,
                                             int n, int m, int breakpoint_orient, np.float64_t mult_const) except *:
        """ Calculate the expected log transition matrix wrt pairwise
        copy number probability.
        """

        cdef int d, s_1, s_2, s_b
        cdef np.ndarray[np.float64_t, ndim=1] p_d

        self._p_d[:] = 0.

        for s_1 in range(self.num_cn_states):
            for s_2 in range(self.num_cn_states):
                d = self.cn_states_total[n, s_1, m] - self.cn_states_total[n + 1, s_2, m]
                self._p_d[d] += p_cn[s_1, s_2]

        for s_b in range(self.num_brk_states):
            for d in range(-self.cn_max - 1, self.cn_max + 2):
                log_breakpoint_p[s_b] += mult_const * self._p_d[d] * self.calc_transition(d - breakpoint_orient * self.brk_states[s_b, m])

    @cython.wraparound(True)
    cpdef void calculate_log_transmat(self, np.float64_t[:, :, :] log_transmat) except *:
        """ Calculate the log transition matrix given current breakpoint and
        allele probabilities.
        """
        cdef int n, m, d, s_b, s_1, s_2, flip, allele

        log_transmat[:] = 0.

        for n in range(0, self.num_segments - 1):
            if self.is_telomere[n] > 0:
                continue

            elif self.breakpoint_idx[n] < 0:
                for m in range(self.num_clones):
                    for s_1 in range(self.num_cn_states):
                        for s_2 in range(self.num_cn_states):
                            log_transmat[n, s_1, s_2] += -self.transition_penalty * self.calc_transition(self.cn_states_total[n, s_1, m] - self.cn_states_total[n + 1, s_2, m])

            else:
                for m in range(self.num_clones):
                    self._p_d[:] = 0.

                    for d in range(-self.cn_max - 1, self.cn_max + 2):
                        for s_b in range(self.num_brk_states):
                            self._p_d[d] += self.p_breakpoint[self.breakpoint_idx[n], s_b] * self.calc_transition(d - self.breakpoint_orient[n] * self.brk_states[s_b, m])

                    for s_1 in range(self.num_cn_states):
                        for s_2 in range(self.num_cn_states):
                            log_transmat[n, s_1, s_2] += -self.transition_penalty * self._p_d[self.cn_states_total[n, s_1, m] - self.cn_states_total[n + 1, s_2, m]]

            self._allele_cn_change[:] = 0.

            for s_1 in range(self.num_cn_states):
                for s_2 in range(self.num_cn_states):
                    for flip in range(2):
                        self._allele_cn_change[flip] = 0.
                        for m in range(self.num_clones):
                            for allele in range(self.num_alleles):
                                if flip == 1:
                                    other_allele = 1 - allele
                                else:
                                    other_allele = allele
                                self._allele_cn_change[flip] += self.calc_transition(self.cn_states[n, s_1, m, allele] - self.cn_states[n + 1, s_2, m, other_allele])
                            self._allele_cn_change[flip] -= self.calc_transition(self.cn_states_total[n, s_1, m] - self.cn_states_total[n + 1, s_2, m])
                    log_transmat[n, s_1, s_2] += -self.transition_penalty * min(self._allele_cn_change[0], self._allele_cn_change[1])

    cpdef np.float64_t calculate_expected_total_reads(self, int n, int s) except *:
        """ Calculate expected total read count for a segment.
        """

        cdef np.float64_t mu = 0.
        cdef int m

        for m in range(self.num_clones):
            mu += self.h[m] * self.cn_states_total[n, s, m]

        mu *= self.l[n]

        return mu

    cpdef np.float64_t calculate_expected_total_reads_partial_h(self, int n, int s, np.float64_t[:] partial_h) except *:
        """ Calculate expected total read count for a segment.
        """

        cdef int m

        for m in range(self.num_clones):
            partial_h[m] = self.l[n] * self.cn_states_total[n, s, m]

    cpdef np.float64_t calculate_expected_allele_ratio(self, int n, int s) except *:
        """ Calculate expected allele ratio for a segment.
        """

        cdef np.float64_t minor_depth = 0.
        cdef np.float64_t total_depth = 0.
        cdef int m

        for m in range(self.num_clones):
            minor_depth += self.h[m] * self.cn_states[n, s, m, 0]
            total_depth += self.h[m] * self.cn_states_total[n, s, m]

        if total_depth <= 0:
            raise ValueError('total_depth <= 0 for s: {}'.format(s))

        return minor_depth / total_depth

    cpdef np.float64_t calculate_expected_allele_ratio_partial_h(self, int n, int s, np.float64_t[:] partial_h) except *:
        """ Calculate expected allele ratio for a segment.
        """

        cdef np.float64_t minor_depth = 0.
        cdef np.float64_t total_depth = 0.
        cdef int m

        for m in range(self.num_clones):
            minor_depth += self.h[m] * self.cn_states[n, s, m, 0]
            total_depth += self.h[m] * self.cn_states_total[n, s, m]

        if total_depth <= 0:
            raise ValueError('total_depth <= 0 for s: {}'.format(s))

        for m in range(self.num_clones):
            partial_h[m] = (
                (self.cn_states[n, s, m, 0] * total_depth - minor_depth * self.cn_states_total[n, s, m]) /
                (total_depth * total_depth))

    cpdef np.float64_t calculate_log_prior_cn(self, int n, int s) except *:
        """ Calculate the log prior of the copy number state for a segment.
        """
        return -1.0 * self.num_alleles_subclonal[n, s] * self.l[n] * self.divergence_weight

    cpdef np.float64_t calculate_log_likelihood_total(self, int n, int s, int u) except *:
        """ Calculate the log likelihood of total read counts for a segment.
        """

        cdef np.float64_t mu, r

        if self.total_likelihood_mask[n] == 0:
            return 0.

        if not self.normal_contamination and self.is_hdel[n, s] == 1:
            mu = self.negbin_hdel_mu

            if u == 0:
                r = self.negbin_hdel_r_0
            else:
                r = self.negbin_hdel_r_1

        else:
            mu = self.calculate_expected_total_reads(n, s)

            if u == 0:
                r = self.negbin_r_0
            else:
                r = self.negbin_r_1

        return negbin_log_likelihood(self.x[n], mu, r)

    cpdef void calculate_log_likelihood_total_partial_h(self, int n, int s, int u, np.float64_t[:] partial_h) except *:
        """ Calculate the partial derivative of the log likelihood
        of total read counts for a segment with respect to haploid
        read depths.
        """

        cdef np.float64_t mu, r, log_likelihood_partial_mu

        if self.total_likelihood_mask[n] == 0:
            partial_h[:] = 0.
            return

        if not self.normal_contamination and self.is_hdel[n, s] == 1:
            partial_h[:] = 0.
            return
            
        else:
            mu = self.calculate_expected_total_reads(n, s)

            if u == 0:
                r = self.negbin_r_0
            else:
                r = self.negbin_r_1

        log_likelihood_partial_mu = negbin_log_likelihood_partial_mu(self.x[n], mu, r)

        self.calculate_expected_total_reads_partial_h(n, s, partial_h)

        for m in range(self.num_clones):
            partial_h[m] *= log_likelihood_partial_mu

    cpdef np.float64_t calculate_log_likelihood_allele(self, int n, int s, int v, int w) except *:
        """ Calculate the log likelihood of allele read counts for a segment.
        """

        cdef np.float64_t p, M, allelic_readcount, minor_readcount

        if self.allele_likelihood_mask[n] == 0:
            return 0.

        if self.is_hdel[n, s] == 1:
            p = 0.
        else:
            p = self.calculate_expected_allele_ratio(n, s)

        if not self.normal_contamination and self.is_loh[n, s] == 1:
            if p == 0.:
                p = self.betabin_loh_p
            elif p == 1.:
                p = 1. - self.betabin_loh_p
            else:
                raise ValueError('expected p {} for loh state {}'.format(p, s))

            if v == 0:
                M = self.betabin_loh_M_0
            else:
                M = self.betabin_loh_M_1

        else:
            if v == 0:
                M = self.betabin_M_0
            else:
                M = self.betabin_M_1

        allelic_readcount = self.y[n, 0] + self.y[n, 1]

        if allelic_readcount == 0:
            return 0.

        if w == 0:
            minor_readcount = self.y[n, 0]

        else:
            minor_readcount = self.y[n, 1]

        return betabin_log_likelihood(minor_readcount, allelic_readcount, p, M)

    cpdef void calculate_log_likelihood_allele_partial_h(self, int n, int s, int v, int w, np.float64_t[:] partial_h) except *:
        """ Calculate the partial derivative of the log likelihood
        of allele read counts for a segment with respect to haploid
        read depths.
        """

        cdef np.float64_t p, M, allelic_readcount, minor_readcount, log_likelihood_partial_p

        if self.allele_likelihood_mask[n] == 0:
            partial_h[:] = 0.
            return

        if not self.normal_contamination and self.is_loh[n, s] == 1:
            partial_h[:] = 0.
            return

        else:
            p = self.calculate_expected_allele_ratio(n, s)

            if v == 0:
                M = self.betabin_M_0
            else:
                M = self.betabin_M_1

        allelic_readcount = self.y[n, 0] + self.y[n, 1]

        if allelic_readcount == 0:
            partial_h[:] = 0.
            return

        if w == 0:
            minor_readcount = self.y[n, 0]

        else:
            minor_readcount = self.y[n, 1]

        log_likelihood_partial_p = betabin_log_likelihood_partial_p(minor_readcount, allelic_readcount, p, M)

        self.calculate_expected_allele_ratio_partial_h(n, s, partial_h)

        for m in range(self.num_clones):
            partial_h[m] *= log_likelihood_partial_p

    cpdef void update_framelogprob(self) except *:
        """ Update the log probability of each segment from the log likelihood and
        likelihood states
        """
        cdef int n, s, u, v, w

        self.framelogprob[:, :] = 0.
        
        for n in range(self.num_segments):
            for s in range(self.num_cn_states):
                for u in range(2):
                    self.framelogprob[n, s] += (
                        self.p_outlier_total[n, u] * self.calculate_log_likelihood_total(n, s, u))

                for v in range(2):
                    for w in range(2):
                        self.framelogprob[n, s] += (
                            self.p_outlier_allele[n, v] * 
                            self.p_allele_swap[n, w] * 
                            self.calculate_log_likelihood_allele(n, s, v, w))

                self.framelogprob[n, s] += self.calculate_log_prior_cn(n, s)

    cpdef void update_p_cn(self) except *:
        """ Update the parameters of the approximating HMM.
        """

        cdef np.ndarray[np.float64_t, ndim=2] alphas = np.empty((self.num_segments, self.num_cn_states))
        cdef np.ndarray[np.float64_t, ndim=2] betas = np.empty((self.num_segments, self.num_cn_states))

        cdef int n, s, s_
        cdef np.ndarray[np.float64_t, ndim=1] log_posterior_marginals = np.zeros((self.num_cn_states,))
        cdef np.ndarray[np.float64_t, ndim=2] log_joint_posterior_marginals = np.zeros((self.num_cn_states, self.num_cn_states))

        # Update frame log probabilities
        self.update_framelogprob()
        assert self.framelogprob.shape[0] == self.num_segments
        assert self.framelogprob.shape[1] == self.num_cn_states
        assert not np.any(np.isnan(self.framelogprob))

        # Build the log transition probabilities of this chain
        self.calculate_log_transmat(self.log_transmat)

        sum_product(self.framelogprob[:, :], self.log_transmat[:, :, :], alphas, betas)

        assert not np.any(np.isnan(alphas))
        assert not np.any(np.isnan(betas))

        self.hmm_log_norm_const = _logsum(alphas[-1, :])

        for n in range(self.num_segments):
            log_posterior_marginals[:] = alphas[n, :] + betas[n, :]
            _exp_normalize(self.posterior_marginals[n, :], log_posterior_marginals)

        assert not np.any(np.isnan(self.posterior_marginals))

        for n in range(self.num_segments - 1):
            for s in range(self.num_cn_states):
                for s_ in range(self.num_cn_states):
                    log_joint_posterior_marginals[s, s_] = (alphas[n, s] + self.log_transmat[n, s, s_]
                        + self.framelogprob[n + 1, s_] + betas[n + 1, s_])

            _exp_normalize2(self.joint_posterior_marginals[n, :, :], log_joint_posterior_marginals)

        assert not np.any(np.isnan(self.joint_posterior_marginals))

    cpdef void update_p_breakpoint(self) except *:
        """ Update the breakpoint approximating distributions.
        """

        cdef int n, m, k
        cdef np.ndarray[np.float64_t, ndim=2] log_p_breakpoint = np.zeros((self.num_breakpoints, self.num_brk_states))

        for n in range(0, self.num_segments - 1):
            if self.breakpoint_idx[n] < 0:
                continue

            for m in range(self.num_clones):
                self.add_log_breakpoint_p_expectation_cn(
                    log_p_breakpoint[self.breakpoint_idx[n], :],
                    self.joint_posterior_marginals[n, :, :],
                    n, m, self.breakpoint_orient[n],
                    -self.transition_penalty)

        for k in range(self.num_breakpoints):
            _exp_normalize(self.p_breakpoint[k, :], log_p_breakpoint[k, :])

        self.calculate_log_transmat(self.cached_log_transmat)

    cpdef void update_p_outlier_total(self) except *:
        """ Update the total read count outlier indicator approximating distributions.
        """
        cdef int n, s, u
        cdef np.ndarray[np.float64_t, ndim=1] log_p_outlier_total = np.zeros((2,))

        for n in range(self.num_segments):
            log_p_outlier_total[0] = log(1. - self.prior_outlier_total)
            log_p_outlier_total[1] = log(self.prior_outlier_total)

            for s in range(self.num_cn_states):
                for u in range(2):
                    log_p_outlier_total[u] += (
                        self.posterior_marginals[n, s] *
                        self.calculate_log_likelihood_total(n, s, u))

            _exp_normalize(self.p_outlier_total[n, :], log_p_outlier_total)

    cpdef void update_p_outlier_allele(self) except *:
        """ Update the allele read count outlier indicator approximating distributions.
        """
        cdef int n, s, v, w
        cdef np.ndarray[np.float64_t, ndim=1] log_p_outlier_allele = np.zeros((2,))

        for n in range(self.num_segments):
            log_p_outlier_allele[0] = log(1. - self.prior_outlier_allele)
            log_p_outlier_allele[1] = log(self.prior_outlier_allele)

            for s in range(self.num_cn_states):
                for v in range(2):
                    for w in range(2):
                        log_p_outlier_allele[v] += (
                            self.p_allele_swap[n, w] *
                            self.posterior_marginals[n, s] *
                            self.calculate_log_likelihood_allele(n, s, v, w))

            _exp_normalize(self.p_outlier_allele[n, :], log_p_outlier_allele)

    cpdef void update_p_allele_swap(self) except *:
        """ Update the allele swap indicator approximating distributions.
        """
        cdef int n, s, v, w
        cdef np.ndarray[np.float64_t, ndim=1] log_p_allele_swap = np.zeros((2,))

        for n in range(self.num_segments):
            log_p_allele_swap[:] = 0.

            for s in range(self.num_cn_states):
                for v in range(2):
                    for w in range(2):
                        log_p_allele_swap[w] += (
                            self.p_outlier_allele[n, v] *
                            self.posterior_marginals[n, s] *
                            self.calculate_log_likelihood_allele(n, s, v, w))

            _exp_normalize(self.p_allele_swap[n, :], log_p_allele_swap)

    cpdef np.float64_t calculate_variational_entropy(self) except *:
        """ Calculate the entropy of the approximating distribution.
        """

        cdef np.float64_t entropy = 0.

        entropy += -np.sum(self.hmm_log_norm_const)
        entropy += np.sum(np.asarray(self.posterior_marginals) * np.asarray(self.framelogprob))
        entropy += np.sum(np.asarray(self.joint_posterior_marginals) * np.asarray(self.log_transmat))
        entropy += _entropy(np.asarray(self.p_breakpoint).flatten())
        entropy += _entropy(np.asarray(self.p_outlier_total).flatten())
        entropy += _entropy(np.asarray(self.p_outlier_allele).flatten())
        entropy += _entropy(np.asarray(self.p_allele_swap).flatten())

        return entropy

    cpdef np.float64_t calculate_variational_energy(self) except *:
        """ Calculate the expectation of the true distribution wrt the
        approximating distribution.
        """

        cdef int n, s, s_, u, v, w
        cdef np.float64_t energy = 0.
        
        # Prior factor
        for n in range(self.num_segments):
            for s in range(self.num_cn_states):
                energy += (
                    self.posterior_marginals[n, s] *
                    self.calculate_log_prior_cn(n, s))

        # Total likelihood factors
        for n in range(self.num_segments):
            for s in range(self.num_cn_states):
                for u in range(2):
                    energy += (
                        self.posterior_marginals[n, s] *
                        self.p_outlier_total[n, u] *
                        self.calculate_log_likelihood_total(n, s, u))

            energy += (
                self.p_outlier_total[n, 0] *
                log(1. - self.prior_outlier_total))

            energy += (
                self.p_outlier_total[n, 1] *
                log(self.prior_outlier_total))

        # Allele likelihood factors
        for n in range(self.num_segments):
            for s in range(self.num_cn_states):
                for v in range(2):
                    for w in range(2):
                        energy += (
                            self.posterior_marginals[n, s] *
                            self.p_outlier_allele[n, v] *
                            self.p_allele_swap[n, w] *
                            self.calculate_log_likelihood_allele(n, s, v, w))

            energy += (
                self.p_outlier_allele[n, 0] *
                log(1. - self.prior_outlier_allele))

            energy += (
                self.p_outlier_allele[n, 1] *
                log(self.prior_outlier_allele))

        # Transitions factor
        for n in range(0, self.num_segments - 1):
            for s in range(self.num_cn_states):
                for s_ in range(self.num_cn_states):
                    energy += self.joint_posterior_marginals[n, s, s_] * self.cached_log_transmat[n, s, s_]

        return energy

    cpdef np.float64_t calculate_elbo(self) except *:
        """ Calculate the evidence lower bound.
        """

        return self.calculate_variational_energy() - self.calculate_variational_entropy()

    cpdef np.float64_t calculate_expected_log_likelihood(self, np.int64_t[:] sample) except *:
        """ Calculate the expectation of the log likelihood wrt the
        approximating distribution.
        """

        cdef int n, s, s_, u, v, w
        cdef np.float64_t energy = 0.
        
        # Total likelihood factors
        for n in range(self.num_segments):
            if sample[n] == 0:
                continue
            for s in range(self.num_cn_states):
                for u in range(2):
                    energy += (
                        self.posterior_marginals[n, s] *
                        self.p_outlier_total[n, u] *
                        self.calculate_log_likelihood_total(n, s, u))

        # Allele likelihood factors
        for n in range(self.num_segments):
            if sample[n] == 0:
                continue
            for s in range(self.num_cn_states):
                for v in range(2):
                    for w in range(2):
                        energy += (
                            self.posterior_marginals[n, s] *
                            self.p_outlier_allele[n, v] *
                            self.p_allele_swap[n, w] *
                            self.calculate_log_likelihood_allele(n, s, v, w))

        return energy

    cpdef void calculate_expected_log_likelihood_partial_h(self, np.int64_t[:] sample, np.float64_t[:] partial_h) except *:
        """ Calculate the expectation of the log likelihood wrt the
        approximating distribution.
        """

        cdef int n, m, s, s_, u, v
        cdef np.ndarray[np.float64_t, ndim=1] segment_ll_partial_h = np.zeros((self.num_clones,))
        
        partial_h[:] = 0.

        # Total likelihood factors
        for n in range(self.num_segments):
            if sample[n] == 0:
                continue
            for s in range(self.num_cn_states):
                for u in range(2):
                    self.calculate_log_likelihood_total_partial_h(n, s, u, segment_ll_partial_h)
                    for m in range(self.num_clones):
                        partial_h[m] += (
                            self.posterior_marginals[n, s] *
                            self.p_outlier_total[n, u] *
                            segment_ll_partial_h[m])

        # Allele likelihood factors
        for n in range(self.num_segments):
            if sample[n] == 0:
                continue
            for s in range(self.num_cn_states):
                for v in range(2):
                    for w in range(2):
                        self.calculate_log_likelihood_allele_partial_h(n, s, v, w, segment_ll_partial_h)
                        for m in range(self.num_clones):
                            partial_h[m] += (
                                self.posterior_marginals[n, s] *
                                self.p_outlier_allele[n, v] *
                                self.p_allele_swap[n, w] *
                                segment_ll_partial_h[m])

    cpdef void infer_cn(self, np.ndarray[np.int64_t, ndim=3] cn) except *:
        """ Infer optimal copy number state sequence.
        """
        cdef int n, m, ell
        cdef np.ndarray[np.int64_t, ndim=1] state_sequence = np.zeros((self.num_segments,), dtype=np.int64)

        max_product(self.framelogprob[:, :], self.log_transmat[:, :, :], state_sequence)

        for n in range(self.num_segments):
            for m in range(self.num_clones):
                for ell in range(self.num_alleles):
                    if self.p_allele_swap[n, 1] > self.p_allele_swap[n, 0]:
                        ell = 1 - ell
                    cn[n, m, ell] = self.cn_states[n, state_sequence[n], m, ell]


cpdef void sum_product(
        np.float64_t[:, :] framelogprob,
        np.float64_t[:, :, :] log_transmat,
        np.float64_t[:, :] alphas,
        np.float64_t[:, :] betas) except *:
    """ Sum product algorithm for chain topology distributions.
    """

    cdef int n, i, j
    cdef np.ndarray[np.float64_t, ndim = 1] work_buffer

    cdef int n_observations = framelogprob.shape[0]
    cdef int n_components = framelogprob.shape[1]

    work_buffer = np.zeros((n_components,))

    for i in range(n_components):
        alphas[0, i] = framelogprob[0, i]

    for n in range(1, n_observations):
        for j in range(n_components):
            for i in range(n_components):
                work_buffer[i] = alphas[n - 1, i] + log_transmat[n - 1, i, j]
            alphas[n, j] = _logsum(work_buffer) + framelogprob[n, j]

    for i in range(n_components):
        betas[n_observations - 1, i] = 0.0

    for n in range(n_observations - 2, -1, -1):
        for i in range(n_components):
            for j in range(n_components):
                work_buffer[j] = (log_transmat[n, i, j] + framelogprob[n + 1, j]
                    + betas[n + 1, j])
            betas[n, i] = _logsum(work_buffer)


cpdef void sum_product_2paramtrans(
        np.float64_t[:, :] framelogprob,
        np.float64_t[:, :] alphas,
        np.float64_t[:, :] betas,
        np.float64_t e0,
        np.float64_t e1,
    ) except *:
    """ Sum product algorithm for chain topology distributions.
    """

    cdef int n, i
    cdef np.ndarray[np.float64_t, ndim = 1] work_buffer

    cdef int n_observations = framelogprob.shape[0]
    cdef int n_components = framelogprob.shape[1]

    work_buffer = np.zeros((n_components,))

    cdef np.float64_t log_e0 = log(e0)
    cdef np.float64_t log_e1 = log(e1)

    cdef np.float64_t notrans

    for i in range(n_components):
        alphas[0, i] = framelogprob[0, i]

    for n in range(1, n_observations):
        notrans = log_e0 + _logsum(alphas[n - 1, :])
        for i in range(n_components):
            alphas[n, i] = _logsumpair(notrans, log_e1 + alphas[n - 1, i]) + framelogprob[n, i]
        for i in range(n_components):
            alphas[n, i] = alphas[n, i]

    for i in range(n_components):
        betas[n_observations - 1, i] = 0.0

    for n in range(n_observations - 2, -1, -1):
        for i in range(n_components):
            work_buffer[i] = framelogprob[n + 1, i] + betas[n + 1, i]
        notrans = log_e0 + _logsum(work_buffer)
        for i in range(n_components):
            betas[n, i] = _logsumpair(notrans, log_e1 + framelogprob[n + 1, i] + betas[n + 1, i])
        for i in range(n_components):
            betas[n, i] = betas[n, i]


@cython.wraparound(True)
cpdef np.float64_t max_product(
        np.float64_t[:, :] framelogprob,
        np.float64_t[:, :, :] log_transmat,
        np.int64_t[:] state_sequence) except *:

    cdef int n, i, j
    cdef np.float64_t logprob
    cdef np.ndarray[np.float64_t, ndim = 2] viterbi_lattice
    cdef np.ndarray[np.float64_t, ndim = 1] work_buffer

    cdef int n_observations = framelogprob.shape[0]
    cdef int n_components = framelogprob.shape[1]

    work_buffer = np.zeros((n_components,))
    viterbi_lattice = np.zeros((n_observations, n_components))

    # Initialization
    viterbi_lattice[0] = framelogprob[0]

    # Induction
    for n in range(1, n_observations):
        for j in range(n_components):
            for i in range(n_components):
                work_buffer[i] = viterbi_lattice[n - 1, i] + log_transmat[n - 1, i, j]
            viterbi_lattice[n, j] = _max(work_buffer) + framelogprob[n, j]

    # Observation traceback
    max_pos = np.argmax(viterbi_lattice[-1, :])
    state_sequence[-1] = max_pos
    logprob = viterbi_lattice[-1, max_pos]

    for n in range(n_observations - 2, -1, -1):
        for i in range(n_components):
            work_buffer[i] = viterbi_lattice[n, i] + log_transmat[n, i, state_sequence[n + 1]]
        max_pos = _argmax(work_buffer)
        state_sequence[n] = max_pos

    return logprob
