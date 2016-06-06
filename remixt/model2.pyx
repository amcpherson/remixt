# cython: profile=True
from libc.math cimport exp, log, fabs
import numpy as np
import scipy
cimport numpy as np
cimport cython

np.import_array()

cdef np.float64_t _NINF = -np.inf
cdef np.float64_t _PI = np.pi


@cython.boundscheck(False)
cdef np.float64_t _max(np.float64_t[:] values):
    cdef np.float64_t vmax = _NINF
    for i in range(values.shape[0]):
        if values[i] > vmax:
            vmax = values[i]
    return vmax


@cython.boundscheck(False)
cdef np.float64_t _max2(np.float64_t[:, :] values):
    cdef np.float64_t vmax = _NINF
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if values[i, j] > vmax:
                vmax = values[i, j]
    return vmax


@cython.boundscheck(False)
cdef int _argmax(np.float64_t[:] values):
    cdef np.float64_t vmax = _NINF
    cdef int imax = 0
    for i in range(values.shape[0]):
        if values[i] > vmax:
            vmax = values[i]
            imax = i
    return imax


@cython.boundscheck(False)
cdef np.float64_t _logsum(np.float64_t[:] X):
    cdef np.float64_t vmax = _max(X)
    cdef np.float64_t power_sum = 0

    for i in range(X.shape[0]):
        power_sum += exp(X[i]-vmax)

    return log(power_sum) + vmax


@cython.boundscheck(False)
cdef np.float64_t _logsum2(np.float64_t[:, :] X):
    cdef np.float64_t vmax = _max2(X)
    cdef np.float64_t power_sum = 0

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            power_sum += exp(X[i, j]-vmax)

    return log(power_sum) + vmax


@cython.boundscheck(False)
cdef np.float64_t _entropy(np.float64_t[:] X):
    cdef np.float64_t entropy = 0

    for i in range(X.shape[0]):
        if X[i] > 0.:
            entropy += X[i] * log(X[i])

    return entropy


@cython.boundscheck(False)
cdef np.float64_t _exp_normalize(np.float64_t[:] Y, np.float64_t[:] X):
    cdef np.float64_t normalize = _logsum(X)
    for i in range(X.shape[0]):
        Y[i] = exp(X[i] - normalize)
    normalize = 0.
    for i in range(X.shape[0]):
        normalize += Y[i]
    for i in range(X.shape[0]):
        Y[i] /= normalize


@cython.boundscheck(False)
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


@cython.boundscheck(False)
@cython.cdivision(True)
cdef np.float64_t _lognormpdf(np.float64_t x, np.float64_t mu, np.float64_t v):
    return -0.5 * log(2 * _PI * v) - (x - mu)**2 / (2. * v)


@cython.boundscheck(False)
def calculate_log_transmat(
        np.float64_t[:, :] log_transmat,
        int cn_max,
        np.float64_t mult_const):
    """ Calculate un-normalized log transition matrix for no breakpoint.
    """

    cdef int cn_1, cn_2

    for cn_1 in range(cn_max + 1):
        for cn_2 in range(cn_max + 1):
            log_transmat[cn_1, cn_2] += mult_const * fabs(cn_1 - cn_2)


@cython.boundscheck(False)
def calculate_log_transmat_expectation_brk(
        np.float64_t[:, :] log_transmat,
        np.float64_t[:] p_breakpoint,
        int breakpoint_orient,
        int cn_max,
        np.float64_t mult_const):
    """ Calculate expected log transition matrix wrt breakpoint
    probability for the total copy number case.
    """

    cdef int d, b, cn_1, cn_2
    cdef np.ndarray[np.float64_t, ndim=1] p_b_geq, p_b_b_geq, p_b_lt, p_b_b_lt

    p_b_geq = np.zeros(p_breakpoint.shape[0] * 2)
    p_b_b_geq = np.zeros(p_breakpoint.shape[0] * 2)

    p_b_lt = np.zeros(p_breakpoint.shape[0] * 2)
    p_b_b_lt = np.zeros(p_breakpoint.shape[0] * 2)

    for d in range(-cn_max - 1, cn_max + 2):
        for b in range(cn_max + 1):
            if d - breakpoint_orient * b >= 0:
                p_b_geq[d] += p_breakpoint[b]
                p_b_b_geq[d] += p_breakpoint[b] * b

            if d - breakpoint_orient * b < 0:
                p_b_lt[d] += p_breakpoint[b]
                p_b_b_lt[d] += p_breakpoint[b] * b

    for cn_1 in range(cn_max + 1):
        for cn_2 in range(cn_max + 1):
            log_transmat[cn_1, cn_2] += mult_const * (
                (cn_1 - cn_2) * p_b_geq[cn_1 - cn_2] - breakpoint_orient * p_b_b_geq[cn_1 - cn_2] +
                (-cn_1 + cn_2) * p_b_lt[cn_1 - cn_2] + breakpoint_orient * p_b_b_lt[cn_1 - cn_2]
            )


@cython.boundscheck(False)
def calculate_log_breakpoint_p_expectation_cn(
        np.float64_t[:] log_breakpoint_p,
        np.float64_t[:, :] p_cn,
        int breakpoint_orient,
        int cn_max,
        np.float64_t mult_const):
    """ Calculate the expected log transition matrix wrt pairwise
    copy number probability.
    """

    cdef int d, cn_1, cn_2, cn_b
    cdef np.ndarray[np.float64_t, ndim=1] p_d

    p_d = np.zeros(((cn_max + 1) * 2,))

    for cn_1 in range(cn_max + 1):
        for cn_2 in range(cn_max + 1):
            d = cn_1 - cn_2
            p_d[d] += p_cn[cn_1, cn_2]

    for cn_b in range(cn_max + 1):
        for d in range(-cn_max - 1, cn_max + 2):
            log_breakpoint_p[cn_b] += mult_const * p_d[d] * fabs(d - breakpoint_orient * cn_b)


cdef class RemixtModel:
    cdef public int num_clones
    cdef public int num_segments
    cdef public int num_breakpoints
    cdef public int num_alleles
    cdef public int cn_max
    cdef public int num_measurements
    cdef public np.int64_t[:, :] x
    cdef public np.int64_t[:] is_telomere
    cdef public np.int64_t[:] breakpoint_idx
    cdef public np.int64_t[:] breakpoint_orient
    cdef public np.float64_t[:, :] effective_lengths
    cdef public np.float64_t[:] true_lengths
    cdef public np.float64_t transition_penalty

    cdef public np.float64_t[:] h
    cdef public np.float64_t a
    cdef public np.float64_t[:, :] unscaled_variance
    cdef public np.float64_t[:, :] likelihood_variance
    cdef public np.float64_t prior_variance
    cdef public np.float64_t prior_total_garbage
    cdef public np.float64_t prior_allele_garbage

    cdef public np.float64_t[:, :, :] p_breakpoint
    cdef public np.float64_t[:, :] p_allele
    cdef public np.float64_t[:, :] p_obs_allele
    cdef public np.float64_t[:, :, :] p_garbage

    cdef public np.float64_t[:, :] default_transition
    cdef public np.float64_t[:, :] hmm_log_norm_const
    cdef public np.float64_t[:, :, :, :] framelogprob
    cdef public np.float64_t[:, :, :, :, :] log_transmat
    cdef public np.float64_t[:, :, :, :] posterior_marginals
    cdef public np.float64_t[:, :, :, :, :] joint_posterior_marginals

    cdef public np.int64_t[:, :, :] w_mat

    def __cinit__(self,
        int num_clones, int num_segments,
        int num_breakpoints, int cn_max,
        np.ndarray[np.int64_t, ndim=2] x,
        np.ndarray[np.int64_t, ndim=1] is_telomere,
        np.ndarray[np.int64_t, ndim=1] breakpoint_idx,
        np.ndarray[np.int64_t, ndim=1] breakpoint_orient,
        np.ndarray[np.float64_t, ndim=1] effective_lengths,
        np.ndarray[np.float64_t, ndim=1] true_lengths,
        np.float64_t transition_penalty):

        self.num_clones = num_clones
        self.num_segments = num_segments
        self.num_breakpoints = num_breakpoints
        self.cn_max = cn_max
        self.num_alleles = 2

        self.x = x

        if is_telomere.shape[0] != num_segments:
            raise ValueError('is_telomere must have length equal to num_segments')

        if breakpoint_idx.shape[0] != num_segments:
            raise ValueError('breakpoint_idx must have length equal to num_segments')

        if breakpoint_orient.shape[0] != num_segments:
            raise ValueError('breakpoint_orient must have length equal to num_segments')

        if breakpoint_idx.max() + 1 != num_breakpoints:
            raise ValueError('breakpoint_idx must have maximum of num_breakpoints positive indices')

        if effective_lengths.shape[0] != num_segments:
            raise ValueError('effective_lengths must have length equal to num_segments')

        if true_lengths.shape[0] != num_segments:
            raise ValueError('true_lengths must have length equal to num_segments')

        self.is_telomere = is_telomere
        self.breakpoint_idx = breakpoint_idx
        self.breakpoint_orient = breakpoint_orient
        self.true_lengths = true_lengths
        self.transition_penalty = fabs(transition_penalty)

        # Effective lengths set directly for total, learned for major minor
        phi = effective_lengths * (x[:, :2].sum(axis=1).astype(float) + 1.) / (x[:, 2].astype(float) + 1.)
        measurement_effective_lengths = np.zeros((self.num_segments, 3))
        measurement_effective_lengths[:, 0] = phi
        measurement_effective_lengths[:, 1] = phi
        measurement_effective_lengths[:, 2] = effective_lengths
        self.effective_lengths = measurement_effective_lengths

        # Must be initialized by user
        self.h = np.zeros((self.num_clones,))
        self.unscaled_variance = (x + 1)**1.75
        self.a = 0.062
        self.likelihood_variance = self.a * np.asarray(self.unscaled_variance)
        self.prior_variance = 1e7
        self.prior_total_garbage = 1e-10
        self.prior_allele_garbage = 1e-10

        # Uniform initialization
        self.p_breakpoint = np.ones((self.num_breakpoints, self.num_clones, self.cn_max + 1))
        self.p_breakpoint /= np.sum(self.p_breakpoint, axis=2)[:, :, np.newaxis]

        self.p_allele = np.ones((self.num_segments, self.num_alleles))
        self.p_allele /= np.sum(self.p_allele, axis=1)[:, np.newaxis]

        self.p_obs_allele = np.array([[0.9, 0.1]] * self.num_segments)

        # Initialize to prior
        self.p_garbage = np.ones((self.num_segments, 3, 2))
        self.p_garbage[:, 0, 0] = (1. - self.prior_total_garbage)
        self.p_garbage[:, 0, 1] = self.prior_total_garbage
        self.p_garbage[:, 1, 0] = (1. - self.prior_allele_garbage)
        self.p_garbage[:, 1, 1] = self.prior_allele_garbage
        self.p_garbage[:, 2, 0] = (1. - self.prior_allele_garbage)
        self.p_garbage[:, 2, 1] = self.prior_allele_garbage

        self.default_transition = np.zeros((self.cn_max + 1, self.cn_max + 1))
        calculate_log_transmat(self.default_transition, self.cn_max, -transition_penalty)

        self.hmm_log_norm_const = np.zeros((self.num_clones, self.num_alleles))
        self.framelogprob = np.ones((self.num_clones, self.num_alleles, self.num_segments, self.cn_max + 1))
        self.log_transmat = np.zeros((self.num_clones, self.num_alleles, self.num_segments - 1, self.cn_max + 1, self.cn_max + 1))
        self.posterior_marginals = np.zeros((self.num_clones, self.num_alleles, self.num_segments, self.cn_max + 1))
        self.joint_posterior_marginals = np.zeros((self.num_clones, self.num_alleles, self.num_segments - 1, self.cn_max + 1, self.cn_max + 1))

        self.w_mat = np.array(
            [[[0, 1],
              [1, 0],
              [1, 1]],
             [[1, 0],
              [0, 1],
              [1, 1]]])

        self.num_measurements = 3

        # Initialize to something valid
        self.posterior_marginals[:] = 1.
        self.posterior_marginals /= np.sum(self.posterior_marginals, axis=-1)[:, :, :, np.newaxis]
        self.joint_posterior_marginals[:] = 1.
        self.joint_posterior_marginals /= np.sum(self.joint_posterior_marginals, axis=(-1, -2))[:, :, :, np.newaxis, np.newaxis]

        # Initialize normal clone
        self.posterior_marginals[0, :, :, :] = 0.
        self.posterior_marginals[0, :, :, 1] = 1.
        self.joint_posterior_marginals[0, :, :, :, :] = 0
        self.joint_posterior_marginals[0, :, :, 1, 1] = 1.

    cpdef void split_clone(self, int m, np.float64_t f) except*:
        """
        """
        self.num_clones += 1

        # Parition clonal fraction for clone m
        self.h = np.append(self.h, [self.h[m] * f])
        self.h[m] = self.h[m] * (1. - f)

        # Duplicate clone specific values
        self.p_breakpoint = np.append(self.p_breakpoint, np.asarray(self.p_breakpoint[:, m, :])[:, np.newaxis, :], axis=1)
        self.hmm_log_norm_const = np.append(self.hmm_log_norm_const, np.asarray(self.hmm_log_norm_const[m, :])[np.newaxis, :], axis=0)
        self.framelogprob = np.append(self.framelogprob, np.asarray(self.framelogprob[m, :, :, :])[np.newaxis, :, :, :], axis=0)
        self.log_transmat = np.append(self.log_transmat, np.asarray(self.log_transmat[m, :, :, :, :])[np.newaxis, :, :, :, :], axis=0)
        self.posterior_marginals = np.append(self.posterior_marginals, np.asarray(self.posterior_marginals[m, :, :, :])[np.newaxis, :, :, :], axis=0)
        self.joint_posterior_marginals = np.append(self.joint_posterior_marginals, np.asarray(self.joint_posterior_marginals[m, :, :, :, :])[np.newaxis, :, :, :, :], axis=0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void calculate_log_transmat(self,
        int n, int m, int ell,
        np.float64_t[:, :] log_transmat) except*:
        """ Calculate the log transition matrix given current breakpoint and
        allele probabilities.
        """

        cdef int v

        if self.is_telomere[n] > 0:
            log_transmat[:] = 0.

        elif self.breakpoint_idx[n] < 0:
            log_transmat[:] = self.default_transition[:]

        else:
            log_transmat[:] = 0.

            for v in range(2):
                if ell == v:
                    calculate_log_transmat_expectation_brk(
                        log_transmat,
                        self.p_breakpoint[self.breakpoint_idx[n], m, :],
                        self.breakpoint_orient[n],
                        self.cn_max,
                        -self.transition_penalty * self.p_allele[n, v])

                else:
                    calculate_log_transmat(
                        log_transmat,
                        self.cn_max,
                        -self.transition_penalty * self.p_allele[n, v])

    cpdef void init_p_cn(self) except*:
        """ Initialize chains to something valid.
        """

        cdef int m, ell

        for _ in range(100):
            for m in range(1, self.num_clones):
                for ell in (1, 0):
                    print m, ell
                    # print self.x
                    # print self.calculate_expected_x2()
                    print np.argmax(self.posterior_marginals[m, ell, :, :], axis=-1)
                    print self.effective_lengths[-3:, 0]
                    self.update_p_cn(m, ell)
                    # print self.x
                    # print self.calculate_expected_x2()
                    print np.argmax(self.posterior_marginals[m, ell, :, :], axis=-1)

            # self.update_phi()

    cpdef calculate_expected_x(self):
        cdef int n, i, m, ell, c, idx_w
        cdef np.ndarray[np.float64_t, ndim=2] x = np.ones((self.num_segments, self.num_measurements))
        for n in range(self.num_segments):
            for i in range(self.num_measurements):
                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        for c in range(self.cn_max + 1):
                            for idx_w in range(2):
                                x[n, i] += self.posterior_marginals[m, ell, n, c] * self.p_obs_allele[n, idx_w] * self.h[m] * c * self.w_mat[idx_w, i, ell] * self.effective_lengths[n, i]
        return x

    cpdef calculate_expected_x2(self):
        cdef int n, i, m, ell
        cdef np.ndarray[np.float64_t, ndim=2] x = np.ones((self.num_segments, self.num_measurements))
        cs = np.argmax(self.posterior_marginals[:, :, :, :], axis=-1)
        for n in range(self.num_segments):
            for i in range(self.num_measurements):
                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        idx_w = 0
                        x[n, i] += self.h[m] * cs[m, ell, n] * self.w_mat[idx_w, i, ell] * self.effective_lengths[n, i]
        return x

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.float64_t update(self) except*:
        """ Single update of all variational parameters.
        """

        elbo_prev = self.calculate_elbo()

        threshold = -1e-6

        for m in range(1, self.num_clones):
            for ell in range(self.num_alleles):
                print 'update_p_cn', m, ell
                self.update_p_cn(m, ell)
                print np.argmax(self.posterior_marginals[m, ell, :, :], axis=-1)
                elbo = self.calculate_elbo()
                print 'elbo diff: {:.10f}'.format(elbo - elbo_prev)
                if elbo - elbo_prev < threshold:
                    raise Exception('elbo error!!!!')
                elbo_prev = elbo

        print 'update_p_breakpoint'
        self.update_p_breakpoint()
        elbo = self.calculate_elbo()
        print 'elbo diff: {:.10f}'.format(elbo - elbo_prev)
        if elbo - elbo_prev < threshold:
            raise Exception('elbo error!!!!')
        elbo_prev = elbo

        print 'update_p_allele'
        self.update_p_allele()
        elbo = self.calculate_elbo()
        print 'elbo diff: {:.10f}'.format(elbo - elbo_prev)
        if elbo - elbo_prev < threshold:
            raise Exception('elbo error!!!!')
        elbo_prev = elbo

        print 'update_p_obs_allele'
        self.update_p_obs_allele()
        elbo = self.calculate_elbo()
        print 'elbo diff: {:.10f}'.format(elbo - elbo_prev)
        if elbo - elbo_prev < threshold:
            raise Exception('elbo error!!!!')
        elbo_prev = elbo

        print 'update_p_garbage'
        self.update_p_garbage()
        elbo = self.calculate_elbo()
        print 'elbo diff: {:.10f}'.format(elbo - elbo_prev)
        if elbo - elbo_prev < threshold:
            raise Exception('elbo error!!!!')
        elbo_prev = elbo

        print 'update_h'
        self.update_h()
        elbo = self.calculate_elbo()
        print 'elbo diff: {:.10f}'.format(elbo - elbo_prev)
        if elbo - elbo_prev < threshold:
            raise Exception('elbo error!!!!')
        elbo_prev = elbo

        print 'update_phi'
        self.update_phi()
        elbo = self.calculate_elbo()
        print 'elbo diff: {:.10f}'.format(elbo - elbo_prev)
        if elbo - elbo_prev < threshold:
            raise Exception('elbo error!!!!')
        elbo_prev = elbo

        print 'update_a', self.a
        self.update_a()
        print self.a
        elbo = self.calculate_elbo()
        print 'elbo diff: {:.10f}'.format(elbo - elbo_prev)
        if elbo - elbo_prev < threshold:
            raise Exception('elbo error!!!!')
        elbo_prev = elbo

        print 'done'

        return self.calculate_elbo()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _lp_rho_1(self, int n, int c):
        """ Term 1 in prior expansion.
        """
        return -self.true_lengths[n] * (self.num_clones - 2) * c**2 / self.prior_variance

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _lp_rho_2(self, int n, int c, int c_):
        """ Term 2 in prior expansion.
        """
        return self.true_lengths[n] * 2. * c * c_ / self.prior_variance

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_0(self, int n, int i):
        """ Term 0 in likelihood expansion.
        """
        return -1. * log(self.likelihood_variance[n, i]) / 2.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_1(self, int n, int i):
        """ Term 1 in likelihood expansion.
        """
        return -1. * self.x[n, i]**2 / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_2(self, int n, int i, int m, int ell, int c, int idx_w):
        """ Term 2 in likelihood expansion.
        """
        return 2. * self.x[n, i] * self.effective_lengths[n, i] * self.h[m] * c * self.w_mat[idx_w, i, ell] / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_3(self, int n, int i, int m, int ell, int c, int idx_w):
        """ Term 3 in likelihood expansion.
        """
        return -1. * (self.effective_lengths[n, i] * self.h[m] * c * self.w_mat[idx_w, i, ell])**2 / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_4(self, int n, int i, int m, int ell, int m_, int ell_, int c, int c_, int idx_w):
        """ Term 4 in likelihood expansion.
        """
        return -2. * self.effective_lengths[n, i]**2 * self.h[m] * self.h[m_] * c * c_ * self.w_mat[idx_w, i, ell] * self.w_mat[idx_w, i, ell_] / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_2_3(self, int n, int i, int m, int ell, int c, int idx_w):
        """ Term 2 + 3 in likelihood expansion.
        """
        return self._ll_eta_2(n, i, m, ell, c, idx_w) + self._ll_eta_3(n, i, m, ell, c, idx_w)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_2_h(self, int n, int i, int m, int ell, int c, int idx_w):
        """ Term 2 h coefficient in likelihood expansion.
        """
        return 2. * self.x[n, i] * self.effective_lengths[n, i] * c * self.w_mat[idx_w, i, ell] / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_3_h(self, int n, int i, int m, int ell, int c, int idx_w):
        """ Term 3 h coefficient in likelihood expansion.
        """
        return -1. * (self.effective_lengths[n, i] * c * self.w_mat[idx_w, i, ell])**2 / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_4_h(self, int n, int i, int m, int ell, int m_, int ell_, int c, int c_, int idx_w):
        """ Term 4 h coefficient in likelihood expansion.
        """
        return -2. * self.effective_lengths[n, i]**2 * c * c_ * self.w_mat[idx_w, i, ell] * self.w_mat[idx_w, i, ell_] / (2. * self.likelihood_variance[n, i])

    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_2_l(self, int n, int i, int m, int ell, int c, int idx_w):
        """ Term 2 l coefficient in in likelihood expansion.
        """
        return 2. * self.x[n, i] * self.h[m] * c * self.w_mat[idx_w, i, ell] / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_3_l(self, int n, int i, int m, int ell, int c, int idx_w):
        """ Term 3 l coefficient in in likelihood expansion.
        """
        return -1. * (self.h[m] * c * self.w_mat[idx_w, i, ell])**2 / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_4_l(self, int n, int i, int m, int ell, int m_, int ell_, int c, int c_, int idx_w):
        """ Term 4 l coefficient in in likelihood expansion.
        """
        return -2. * self.h[m] * self.h[m_] * c * c_ * self.w_mat[idx_w, i, ell] * self.w_mat[idx_w, i, ell_] / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_0_a(self, int n, int i):
        """ Term 0 a coefficient in likelihood expansion (excluding a constant term).
        """
        return -1. / 2.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_1_a(self, int n, int i):
        """ Term 1 a coefficient in likelihood expansion.
        """
        return -1. * self.x[n, i]**2 / (2. * self.unscaled_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_2_a(self, int n, int i, int m, int ell, int c, int idx_w):
        """ Term 2 a coefficient in likelihood expansion.
        """
        return 2. * self.x[n, i] * self.effective_lengths[n, i] * self.h[m] * c * self.w_mat[idx_w, i, ell] / (2. * self.unscaled_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_3_a(self, int n, int i, int m, int ell, int c, int idx_w):
        """ Term 3 a coefficient in likelihood expansion.
        """
        return -1. * (self.effective_lengths[n, i] * self.h[m] * c * self.w_mat[idx_w, i, ell])**2 / (2. * self.unscaled_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_4_a(self, int n, int i, int m, int ell, int m_, int ell_, int c, int c_, int idx_w):
        """ Term 4 a coefficient in likelihood expansion.
        """
        return -2. * self.effective_lengths[n, i]**2 * self.h[m] * self.h[m_] * c * c_ * self.w_mat[idx_w, i, ell] * self.w_mat[idx_w, i, ell_] / (2. * self.unscaled_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_eta_2_3_a(self, int n, int i, int m, int ell, int c, int idx_w):
        """ Term 2 + 3 a coefficient in likelihood expansion.
        """
        return self._ll_eta_2_a(n, i, m, ell, c, idx_w) + self._ll_eta_3_a(n, i, m, ell, c, idx_w)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void update_p_cn(self, int m, int ell) except*:
        """ Update the parameters of the approximating HMM.
        """

        cdef np.ndarray[np.float64_t, ndim=2] alphas = np.empty((self.num_segments, self.cn_max + 1))
        cdef np.ndarray[np.float64_t, ndim=2] betas = np.empty((self.num_segments, self.cn_max + 1))

        cdef int n, c, m_, c_, i, idx_w, ell_
        cdef np.ndarray[np.float64_t, ndim=1] log_posterior_marginals = np.zeros((self.cn_max + 1,))
        cdef np.ndarray[np.float64_t, ndim=2] log_joint_posterior_marginals = np.zeros((self.cn_max + 1, self.cn_max + 1))

        # Build the frame log probabilities of this chain
        for n in range(self.num_segments):
            for c in range(self.cn_max + 1):

                self.framelogprob[m, ell, n, c] = 0.

                # Copy number prior, singleton factors
                self.framelogprob[m, ell, n, c] += self._lp_rho_1(n, c)

                # Copy number prior, pairwise factors
                for m_ in range(1, self.num_clones):
                    if m_ == m:
                        continue
                    for c_ in range(self.cn_max + 1):
                        self.framelogprob[m, ell, n, c] += self.posterior_marginals[m_, ell, n, c_] * self._lp_rho_2(n, c, c_)

                # Likelihood, singleton factors
                for i in range(self.num_measurements):
                    for idx_w in range(2):
                        self.framelogprob[m, ell, n, c] += (
                            self.p_obs_allele[n, idx_w] *
                            self.p_garbage[n, i, 0] *
                            self._ll_eta_2_3(n, i, m, ell, c, idx_w))

                # Likelihood, pairwise factors
                for i in range(self.num_measurements):
                    for m_ in range(self.num_clones):
                        for ell_ in range(self.num_alleles):
                            if m_ == m and ell_ == ell:
                                continue
                            for c_ in range(self.cn_max + 1):
                                for idx_w in range(2):
                                    self.framelogprob[m, ell, n, c] += (
                                        self.posterior_marginals[m_, ell_, n, c_] *
                                        self.p_obs_allele[n, idx_w] *
                                        self.p_garbage[n, i, 0] *
                                        self._ll_eta_4(n, i, m, ell, m_, ell_, c, c_, idx_w))

        # Build the log transition probabilities of this chain
        for n in range(0, self.num_segments - 1):
            self.calculate_log_transmat(n, m, ell, self.log_transmat[m, ell, n, :, :])

        sum_product(self.framelogprob[m, ell, :, :], self.log_transmat[m, ell, :, :, :], alphas, betas)

        assert not np.any(np.isnan(alphas))
        assert not np.any(np.isnan(betas))

        self.hmm_log_norm_const[m, ell] = _logsum(alphas[-1, :])

        for n in range(self.num_segments):
            log_posterior_marginals[:] = alphas[n, :] + betas[n, :]
            _exp_normalize(self.posterior_marginals[m, ell, n, :], log_posterior_marginals)

        assert not np.any(np.isnan(self.posterior_marginals))

        for n in range(self.num_segments - 1):
            for c in range(self.cn_max + 1):
                for c_ in range(self.cn_max + 1):
                    log_joint_posterior_marginals[c, c_] = (alphas[n, c] + self.log_transmat[m, ell, n, c, c_]
                        + self.framelogprob[m, ell, n + 1, c_] + betas[n + 1, c_])

            _exp_normalize2(self.joint_posterior_marginals[m, ell, n, :, :], log_joint_posterior_marginals)

        assert not np.any(np.isnan(self.joint_posterior_marginals))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_p_breakpoint(self) except *:
        """ Update the breakpoint approximating distributions.
        """

        cdef int t, i, j, brk_allele
        cdef np.ndarray[np.float64_t, ndim=3] log_p_breakpoint = np.zeros((self.num_breakpoints, self.num_clones, self.cn_max + 1))

        for n in range(0, self.num_segments - 1):
            if self.breakpoint_idx[n] < 0:
                continue

            for m in range(self.num_clones):
                for ell in range(self.num_alleles):
                    calculate_log_breakpoint_p_expectation_cn(
                        log_p_breakpoint[self.breakpoint_idx[n], m, :],
                        self.joint_posterior_marginals[m, ell, n, :, :],
                        self.breakpoint_orient[n],
                        self.cn_max,
                        -self.transition_penalty * self.p_allele[n, ell])

        for k in range(self.num_breakpoints):
            for m in range(self.num_clones):
                _exp_normalize(self.p_breakpoint[k, m, :], log_p_breakpoint[k, m, :])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_p_allele(self) except*:
        """ Update the rearranged allele approximating distribution.
        """

        cdef int v, n, m, ell

        cdef np.ndarray[np.float64_t, ndim=1] log_p_breakpoint = np.empty((self.cn_max + 1,))
        cdef np.ndarray[np.float64_t, ndim=2] log_transmat = np.empty((self.cn_max + 1, self.cn_max + 1))
        cdef np.ndarray[np.float64_t, ndim=1] log_p_allele = np.empty((self.num_alleles,))

        for n in range(0, self.num_segments - 1):
            if self.breakpoint_idx[n] < 0:
                continue

            log_p_allele[:] = 0.

            for v in range(self.num_alleles):
                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        if v == ell:
                            log_p_breakpoint[:] = 0.

                            calculate_log_breakpoint_p_expectation_cn(
                                log_p_breakpoint,
                                self.joint_posterior_marginals[m, ell, n, :, :],
                                self.breakpoint_orient[n],
                                self.cn_max,
                                -self.transition_penalty)

                            log_p_allele[v] += np.sum(
                                log_p_breakpoint * self.p_breakpoint[self.breakpoint_idx[n], m, :])

                        else:
                            log_transmat[:] = 0

                            calculate_log_transmat(
                                log_transmat,
                                self.cn_max,
                                -self.transition_penalty)

                            log_p_allele[v] += np.sum(
                                log_transmat * self.joint_posterior_marginals[m, ell, n, :, :])

            _exp_normalize(self.p_allele[n, :], log_p_allele)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_p_obs_allele(self) except*:
        """ Update the observed allele approximating distribution.
        """

        cdef int n, idx_w, i, m, ell, c, m_, ell_, c_
        cdef np.ndarray[np.float64_t, ndim=1] log_p_obs_allele = np.empty((self.num_alleles,))

        for n in range(self.num_segments):
            log_p_obs_allele[:] = 0.

            for idx_w in range(2):

                # Likelihood, singleton factors
                for i in range(self.num_measurements):
                    for m in range(self.num_clones):
                        for ell in range(self.num_alleles):
                            for c in range(self.cn_max + 1):
                                log_p_obs_allele[idx_w] += (
                                    self.posterior_marginals[m, ell, n, c]
                                    * self.p_garbage[n, i, 0]
                                    * self._ll_eta_2_3(n, i, m, ell, c, idx_w))

                # Likelihood, pairwise factors
                for i in range(self.num_measurements):
                    for m in range(self.num_clones):
                        for ell in range(self.num_alleles):
                            for m_ in range(self.num_clones):
                                for ell_ in range(self.num_alleles):
                                    if not (m_ > m or (m_ == m and ell_ > ell)):
                                        continue
                                    for c in range(self.cn_max + 1):
                                        for c_ in range(self.cn_max + 1):
                                            log_p_obs_allele[idx_w] += (
                                                self.posterior_marginals[m, ell, n, c]
                                                * self.posterior_marginals[m_, ell_, n, c_]
                                                * self.p_garbage[n, i, 0]
                                                * self._ll_eta_4(n, i, m, ell, m_, ell_, c, c_, idx_w))

            _exp_normalize(self.p_obs_allele[n, :], log_p_obs_allele)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_p_garbage(self) except*:
        """ Update the read count garbage state approximating distribution.
        """

        cdef int n, i, m, ell, c, idx_w, m_, ell_, c_
        cdef np.ndarray[np.float64_t, ndim=2] log_p_garbage = np.empty((3, 2))
        cdef np.ndarray[np.float64_t, ndim=1] log_p_allele_garbage = np.empty((2,))

        for n in range(self.num_segments):
            log_p_garbage[:] = 0.

            for i in range(self.num_measurements):

                # Likelihood, constant factors
                log_p_garbage[i, 0] += self._ll_eta_0(n, i)
                log_p_garbage[i, 0] += self._ll_eta_1(n, i)

                # Likelihood, singleton factors
                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        for c in range(self.cn_max + 1):
                            for idx_w in range(2):
                                log_p_garbage[i, 0] += (
                                    self.posterior_marginals[m, ell, n, c]
                                    * self.p_obs_allele[n, idx_w]
                                    * self._ll_eta_2_3(n, i, m, ell, c, idx_w))

                # Likelihood, pairwise factors
                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        for m_ in range(self.num_clones):
                            for ell_ in range(self.num_alleles):
                                if not (m_ > m or (m_ == m and ell_ > ell)):
                                    continue
                                for c in range(self.cn_max + 1):
                                    for c_ in range(self.cn_max + 1):
                                        for idx_w in range(2):
                                            log_p_garbage[i, 0] += (
                                                self.posterior_marginals[m, ell, n, c]
                                                * self.posterior_marginals[m_, ell_, n, c_]
                                                * self.p_obs_allele[n, idx_w]
                                                * self._ll_eta_4(n, i, m, ell, m_, ell_, c, c_, idx_w))

            # Total reads
            log_p_garbage[2, 0] += log(1.0 - self.prior_total_garbage)
            log_p_garbage[2, 1] += log(self.prior_total_garbage)
            _exp_normalize(self.p_garbage[n, 2, :], log_p_garbage[2, :])

            # Allele specific reads
            log_p_allele_garbage[:] = log_p_garbage[0, :] + log_p_garbage[1, :]
            log_p_allele_garbage[0] += log(1.0 - self.prior_allele_garbage)
            log_p_allele_garbage[1] += log(self.prior_allele_garbage)
            _exp_normalize(self.p_garbage[n, 0, :], log_p_allele_garbage)
            _exp_normalize(self.p_garbage[n, 1, :], log_p_allele_garbage)

    def _h_opt_objective(self, h, coeff_1, coeff_2):
        return -np.sum(h * coeff_1) - np.sum(np.outer(h, h) * coeff_2)

    def _h_opt_jacobian(self, h, coeff_1, coeff_2):
        return -coeff_1 - np.dot(coeff_2 + coeff_2.T, h)

    def _h_opt_hessian(self, h, coeff_1, coeff_2):
        return -coeff_2 - coeff_2.T

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_h(self) except *:


        cdef int i, t, m_1, m_2, m, w, ell
        cdef np.ndarray[np.float64_t, ndim=1] coeff_1
        cdef np.ndarray[np.float64_t, ndim=2] coeff_2

        coeff_1 = np.zeros((self.num_clones,))
        coeff_2 = np.zeros((self.num_clones, self.num_clones))

        for n in range(self.num_segments):
            for m in range(self.num_clones):
                for ell in range(self.num_alleles):
                    for i in range(self.num_measurements):
                        for c in range(self.cn_max + 1):
                            for idx_w in range(2):
                                coeff_1[m] += (
                                    self.posterior_marginals[m, ell, n, c] *
                                    self.p_obs_allele[n, idx_w] *
                                    self.p_garbage[n, i, 0] *
                                    self._ll_eta_2_h(n, i, m, ell, c, idx_w))

                                coeff_2[m, m] += (
                                    self.posterior_marginals[m, ell, n, c] *
                                    self.p_obs_allele[n, idx_w] *
                                    self.p_garbage[n, i, 0] *
                                    self._ll_eta_3_h(n, i, m, ell, c, idx_w))

        for n in range(self.num_segments):
            for m in range(self.num_clones):
                for ell in range(self.num_alleles):
                    for m_ in range(self.num_clones):
                        for ell_ in range(self.num_alleles):
                            if not (m_ > m or (m_ == m and ell_ > ell)):
                                continue
                            for i in range(self.num_measurements):
                                for c in range(self.cn_max + 1):
                                    for c_ in range(self.cn_max + 1):
                                        for idx_w in range(2):
                                            coeff_2[m, m_] += (
                                                self.posterior_marginals[m, ell, n, c] *
                                                self.posterior_marginals[m_, ell_, n, c_] *
                                                self.p_obs_allele[n, idx_w] *
                                                self.p_garbage[n, i, 0] *
                                                self._ll_eta_4_h(n, i, m, ell, m_, ell_, c, c_, idx_w))

        try:
            self.h = np.linalg.solve(coeff_2 + coeff_2.T, -coeff_1)
            success = True
        except Exception as e:
            success = False

        if success and not np.any(np.asarray(self.h) < 0.):
            return

        opt_result = scipy.optimize.minimize(
            self._h_opt_objective,
            self.h,
            args=(coeff_1, coeff_2),
            jac=self._h_opt_jacobian,
            hess=None,
            bounds=[(0., np.inf)] * self.num_clones)

        if not opt_result.success:
            raise Exception('error for optimizing h: ' + opt_result.message)

        self.h = opt_result.x

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_phi(self) except *:
        """ Update effective length for allele read counts.
        """

        cdef int n, i, m, ell, c, idx_w, m_, ell_, c_
        cdef np.float64_t coeff_1, coeff_2, phi

        for n in range(self.num_segments):
            coeff_1 = 0.
            coeff_2 = 0.

            for i in range(2):
                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        for c in range(self.cn_max + 1):
                            for idx_w in range(2):
                                coeff_1 += (
                                    self.posterior_marginals[m, ell, n, c] *
                                    self.p_obs_allele[n, idx_w] *
                                    self.p_garbage[n, i, 0] *
                                    self._ll_eta_2_l(n, i, m, ell, c, idx_w))

                                coeff_2 += (
                                    self.posterior_marginals[m, ell, n, c] *
                                    self.p_obs_allele[n, idx_w] *
                                    self.p_garbage[n, i, 0] *
                                    self._ll_eta_3_l(n, i, m, ell, c, idx_w))

            for i in range(2):
                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        for m_ in range(self.num_clones):
                            for ell_ in range(self.num_alleles):
                                if not (m_ > m or (m_ == m and ell_ > ell)):
                                    continue
                                for c in range(self.cn_max + 1):
                                    for c_ in range(self.cn_max + 1):
                                        for idx_w in range(2):
                                            coeff_2 += (
                                                self.posterior_marginals[m, ell, n, c] *
                                                self.posterior_marginals[m_, ell_, n, c_] *
                                                self.p_obs_allele[n, idx_w] *
                                                self.p_garbage[n, i, 0] *
                                                self._ll_eta_4_l(n, i, m, ell, m_, ell_, c, c_, idx_w))

            if coeff_2 == 0.:
                phi = 0.
            else:
                phi = -coeff_1 / (2. * coeff_2)

            self.effective_lengths[n, 0] = phi
            self.effective_lengths[n, 1] = phi

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_a(self) except *:
        """ Update variance multipler.
        """

        cdef int n, i, m, ell, c, idx_w, m_, ell_, c_
        cdef np.float64_t coeff_1, coeff_2

        coeff_1 = 0.
        coeff_2 = 0.

        for n in range(self.num_segments):
            for i in range(self.num_measurements):

                coeff_1 += self.p_garbage[n, i, 0] * self._ll_eta_0_a(n, i)
                coeff_2 += self.p_garbage[n, i, 0] * self._ll_eta_1_a(n, i)

                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        for c in range(self.cn_max + 1):
                            for idx_w in range(2):
                                coeff_2 += (
                                    self.posterior_marginals[m, ell, n, c] *
                                    self.p_obs_allele[n, idx_w] *
                                    self.p_garbage[n, i, 0] *
                                    self._ll_eta_2_3_a(n, i, m, ell, c, idx_w))

                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        for m_ in range(self.num_clones):
                            for ell_ in range(self.num_alleles):
                                if not (m_ > m or (m_ == m and ell_ > ell)):
                                    continue
                                for c in range(self.cn_max + 1):
                                    for c_ in range(self.cn_max + 1):
                                        for idx_w in range(2):
                                            coeff_2 += (
                                                self.posterior_marginals[m, ell, n, c] *
                                                self.posterior_marginals[m_, ell_, n, c_] *
                                                self.p_obs_allele[n, idx_w] *
                                                self.p_garbage[n, i, 0] *
                                                self._ll_eta_4_a(n, i, m, ell, m_, ell_, c, c_, idx_w))

        self.a = coeff_2 / coeff_1
        self.likelihood_variance = np.asarray(self.unscaled_variance) * self.a

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.float64_t calculate_variational_entropy(self):
        """ Calculate the entropy of the approximating distribution.
        """

        cdef np.float64_t entropy = 0.

        entropy += -np.sum(self.hmm_log_norm_const)
        entropy += np.sum(np.asarray(self.posterior_marginals) * np.asarray(self.framelogprob))
        entropy += np.sum(np.asarray(self.joint_posterior_marginals) * np.asarray(self.log_transmat))
        entropy += _entropy(np.asarray(self.p_breakpoint).flatten())
        entropy += _entropy(np.asarray(self.p_allele).flatten())
        entropy += _entropy(np.asarray(self.p_obs_allele).flatten())
        entropy += _entropy(np.asarray(self.p_garbage)[:, 0, :].flatten())
        entropy += _entropy(np.asarray(self.p_garbage)[:, 2, :].flatten())

        # print 'entropy', entropy

        return entropy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.float64_t calculate_variational_energy(self) except*:
        """ Calculate the expectation of the true distribution wrt the
        approximating distribution.
        """

        cdef np.ndarray[np.float64_t, ndim=2] log_transmat = np.empty((self.cn_max + 1, self.cn_max + 1))

        cdef int n, i, m, ell, c, m_, ell_, c_
        cdef np.float64_t energy = 0.

        for n in range(self.num_segments):
            for m in range(1, self.num_clones):
                for ell in range(self.num_alleles):
                    for c in range(self.cn_max + 1):

                        # Copy number prior, singleton factors
                        energy += self.posterior_marginals[m, ell, n, c] * self._lp_rho_1(n, c)

                        # Copy number prior, pairwise factors
                        for m_ in range(1, self.num_clones):
                            if m_ <= m:
                                continue
                            for c_ in range(self.cn_max + 1):
                                energy += self.posterior_marginals[m, ell, n, c] * self.posterior_marginals[m_, ell, n, c_] * self._lp_rho_2(n, c, c_)

        for n in range(self.num_segments):
            for i in range(self.num_measurements):

                # Likelihood, constant factors
                energy += self.p_garbage[n, i, 0] * self._ll_eta_0(n, i)
                energy += self.p_garbage[n, i, 0] * self._ll_eta_1(n, i)

                # Likelihood, singleton factor
                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        for c in range(self.cn_max + 1):
                            for idx_w in range(2):
                                energy += (
                                    self.posterior_marginals[m, ell, n, c] *
                                    self.p_obs_allele[n, idx_w] *
                                    self.p_garbage[n, i, 0] *
                                    self._ll_eta_2_3(n, i, m, ell, c, idx_w))

                # Likelihood, pairwise factor
                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        for m_ in range(self.num_clones):
                            for ell_ in range(self.num_alleles):
                                if not (m_ > m or (m_ == m and ell_ > ell)):
                                    continue
                                for c in range(self.cn_max + 1):
                                    for c_ in range(self.cn_max + 1):
                                        for idx_w in range(2):
                                            energy += (
                                                self.posterior_marginals[m, ell, n, c] *
                                                self.posterior_marginals[m_, ell_, n, c_] *
                                                self.p_obs_allele[n, idx_w] *
                                                self.p_garbage[n, i, 0] *
                                                self._ll_eta_4(n, i, m, ell, m_, ell_, c, c_, idx_w))

        # Transitions factor
        for n in range(0, self.num_segments - 1):
            for m in range(self.num_clones):
                for ell in range(self.num_alleles):
                    self.calculate_log_transmat(n, m, ell, log_transmat)
                    energy += np.sum(self.joint_posterior_marginals[m, ell, n, :, :] * log_transmat)

        # Garbage state prior
        for n in range(self.num_segments):
            energy += self.p_garbage[n, 0, 0] * log(1. - self.prior_total_garbage)
            energy += self.p_garbage[n, 0, 1] * log(self.prior_total_garbage)
            energy += self.p_garbage[n, 2, 0] * log(1. - self.prior_allele_garbage)
            energy += self.p_garbage[n, 2, 1] * log(self.prior_allele_garbage)

        # print 'energy', energy

        return energy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.float64_t calculate_elbo(self) except*:
        """ Calculate the evidence lower bound.
        """

        return self.calculate_variational_energy() - self.calculate_variational_entropy()

    @cython.wraparound(False)
    cpdef void infer_cn(self, np.ndarray[np.int64_t, ndim=3] cn) except*:
        """ Infer optimal copy number state sequence.
        """

        for m in range(self.num_clones):
            for ell in range(self.num_alleles):
                max_product(self.framelogprob[m, ell, :, :], self.log_transmat[m, ell, :, :, :], cn[:, m, ell])


@cython.boundscheck(False)
cpdef void sum_product(
        np.float64_t[:, :] framelogprob,
        np.float64_t[:, :, :] log_transmat,
        np.float64_t[:, :] alphas,
        np.float64_t[:, :] betas) except*:
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


@cython.boundscheck(False)
cpdef np.float64_t max_product(
        np.float64_t[:, :] framelogprob,
        np.float64_t[:, :, :] log_transmat,
        np.int64_t[:] state_sequence) except*:

    cdef int t, max_pos
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
