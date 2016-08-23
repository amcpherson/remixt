from libc.math cimport exp, log, fabs
import numpy as np
import scipy
import itertools
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


cdef class RemixtModel:
    cdef public int num_clones
    cdef public int num_segments
    cdef public int num_breakpoints
    cdef public int num_alleles
    cdef public int cn_max
    cdef public int cn_diff_max
    cdef public int num_measurements
    cdef public int num_cn_states
    cdef public np.int64_t[:, :, :] cn_states

    cdef public np.int64_t[:, :] x
    cdef public np.int64_t[:] is_telomere
    cdef public np.int64_t[:] breakpoint_idx
    cdef public np.int64_t[:] breakpoint_orient
    cdef public np.float64_t[:, :] effective_lengths
    cdef public np.float64_t[:] true_lengths
    cdef public np.float64_t transition_penalty

    cdef public np.float64_t[:] h
    cdef public np.float64_t[:] a
    cdef public np.float64_t[:, :] unscaled_variance
    cdef public np.float64_t[:, :] likelihood_variance
    cdef public np.float64_t prior_variance
    cdef public np.float64_t prior_total_garbage
    cdef public np.float64_t prior_allele_garbage

    cdef public np.float64_t[:, :, :] p_breakpoint
    cdef public np.float64_t[:, :, :] p_allele
    cdef public np.float64_t[:, :, :] p_garbage

    cdef public np.float64_t hmm_log_norm_const
    cdef public np.float64_t[:, :] framelogprob
    cdef public np.float64_t[:, :, :] log_transmat
    cdef public np.float64_t[:, :] posterior_marginals
    cdef public np.float64_t[:, :, :] joint_posterior_marginals

    cdef public np.int64_t[:, :] w_mat

    def __cinit__(self,
        int num_clones, int num_segments,
        int num_breakpoints, int cn_max, int cn_diff_max,
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
        self.cn_diff_max = cn_diff_max
        self.num_alleles = 2
        self.cn_states = self.create_cn_states(self.num_clones, self.num_alleles, self.cn_max, self.cn_diff_max)
        self.num_cn_states = self.cn_states.shape[0]

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
        phi = (x[:, :2].sum(axis=1).astype(float) + 1.) / (x[:, 2].astype(float) + 1.)
        measurement_effective_lengths = np.zeros((self.num_segments, 3))
        measurement_effective_lengths[:, 0] = effective_lengths * phi
        measurement_effective_lengths[:, 1] = effective_lengths * phi
        measurement_effective_lengths[:, 2] = effective_lengths
        self.effective_lengths = measurement_effective_lengths

        # Must be initialized by user
        self.h = np.zeros((self.num_clones,))
        self.unscaled_variance = (x + 1)**1.75
        self.a = np.array([0.062]*3)
        self.likelihood_variance = self.a * np.asarray(self.unscaled_variance)
        self.prior_variance = 1e5
        self.prior_total_garbage = 1e-10
        self.prior_allele_garbage = 1e-10

        # Initialize to prefer positive breakpoint copy number
        p_breakpoint = np.zeros((self.num_breakpoints, self.num_clones, self.cn_max + 1))
        p_breakpoint[:, :, 0] = 0.1
        p_breakpoint[:, :, 1] = 0.9
        self.p_breakpoint = p_breakpoint

        # Initialize to prefer no allele swaps
        self.p_allele = np.ones((self.num_segments, 2, 2))
        self.p_allele[:, 0, 0] = 0.49
        self.p_allele[:, 0, 1] = 0.01
        self.p_allele[:, 1, 0] = 0.49
        self.p_allele[:, 1, 1] = 0.01

        # Initialize to prior
        self.p_garbage = np.ones((self.num_segments, 3, 2))
        self.p_garbage[:, 0, 0] = (1. - self.prior_total_garbage)
        self.p_garbage[:, 0, 1] = self.prior_total_garbage
        self.p_garbage[:, 1, 0] = (1. - self.prior_allele_garbage)
        self.p_garbage[:, 1, 1] = self.prior_allele_garbage
        self.p_garbage[:, 2, 0] = (1. - self.prior_allele_garbage)
        self.p_garbage[:, 2, 1] = self.prior_allele_garbage

        self.hmm_log_norm_const = 0.
        self.framelogprob = np.ones((self.num_segments, self.num_cn_states))
        self.log_transmat = np.zeros((self.num_segments - 1, self.num_cn_states, self.num_cn_states))
        self.posterior_marginals = np.zeros((self.num_segments, self.num_cn_states))
        self.joint_posterior_marginals = np.zeros((self.num_segments - 1, self.num_cn_states, self.num_cn_states))

        self.w_mat = np.array(
            [[1, 0],
             [0, 1],
             [1, 1]])

        self.num_measurements = 3

        # Initialize to something valid
        self.posterior_marginals[:] = 1.
        self.posterior_marginals /= np.sum(self.posterior_marginals, axis=-1)[:, np.newaxis]
        self.joint_posterior_marginals[:] = 1.
        self.joint_posterior_marginals /= np.sum(self.joint_posterior_marginals, axis=(-1, -2))[:, np.newaxis, np.newaxis]

    def create_cn_states(self, num_clones, num_alleles, cn_max, cn_diff_max):
        """ Create a list of allele specific copy number states.
        """
        num_tumour_vars = (num_clones - 1) * num_alleles

        cn_states = list()
        for cn in itertools.product(range(cn_max + 1), repeat=num_tumour_vars):
            cn = np.array((1, 1) + cn).reshape((num_clones, num_alleles))

            if not np.all(cn.sum(axis=1) <= cn_max):
                continue

            if not np.all(cn[1:, :].max(axis=0) - cn[1:, :].min(axis=0) <= cn_diff_max):
                continue

            cn_states.append(cn)

        cn_states = np.array(cn_states)

        return cn_states

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_log_transmat(self, np.float64_t[:, :] log_transmat, int m, int ell, int w, np.float64_t mult_const):
        """ Add log transition matrix for no breakpoint.
        """

        cdef int s_1, s_2, ell_

        ell_ = w * (1 - ell) + (1 - w) * ell
        for s_1 in range(self.num_cn_states):
            for s_2 in range(self.num_cn_states):
                log_transmat[s_1, s_2] += mult_const * fabs(self.cn_states[s_1, m, ell] - self.cn_states[s_2, m, ell_])

    @cython.boundscheck(False)
    @cython.wraparound(True)
    def add_log_transmat_expectation_brk(self, np.float64_t[:, :] log_transmat, int m, int ell, int w,
                                         np.float64_t[:] p_breakpoint, int breakpoint_orient,
                                         np.float64_t mult_const):
        """ Add expected log transition matrix wrt breakpoint
        probability for the total copy number case.
        """

        cdef int d, cn_b, s_1, s_2
        cdef np.ndarray[np.float64_t, ndim=1] p_b

        p_b = np.zeros((self.cn_max + 1) * 2)

        for d in range(-self.cn_max - 1, self.cn_max + 2):
            for cn_b in range(self.cn_max + 1):
                p_b[d] += p_breakpoint[cn_b] * abs(d - breakpoint_orient * cn_b)

        ell_ = w * (1 - ell) + (1 - w) * ell
        for s_1 in range(self.num_cn_states):
            for s_2 in range(self.num_cn_states):
                log_transmat[s_1, s_2] += mult_const * p_b[self.cn_states[s_1, m, ell] - self.cn_states[s_2, m, ell_]]

    @cython.boundscheck(False)
    @cython.wraparound(True)
    def add_log_breakpoint_p_expectation_cn(self, np.float64_t[:] log_breakpoint_p, np.float64_t[:, :] p_cn,
                                            int m, int ell, int w, int breakpoint_orient, np.float64_t mult_const):
        """ Calculate the expected log transition matrix wrt pairwise
        copy number probability.
        """

        cdef int d, s_1, s_2, cn_b, ell_
        cdef np.ndarray[np.float64_t, ndim=1] p_d

        p_d = np.zeros(((self.cn_max + 1) * 2,))

        ell_ = w * (1 - ell) + (1 - w) * ell
        for s_1 in range(self.num_cn_states):
            for s_2 in range(self.num_cn_states):
                d = self.cn_states[s_1, m, ell] - self.cn_states[s_2, m, ell_]
                p_d[d] += p_cn[s_1, s_2]

        for cn_b in range(self.cn_max + 1):
            for d in range(-self.cn_max - 1, self.cn_max + 2):
                log_breakpoint_p[cn_b] += mult_const * p_d[d] * fabs(d - breakpoint_orient * cn_b)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void calculate_log_transmat(self, int n, np.float64_t[:, :] log_transmat) except*:
        """ Calculate the log transition matrix given current breakpoint and
        allele probabilities.
        """

        cdef int v, w, m
        cdef np.float64_t mult_const

        if self.is_telomere[n] > 0:
            log_transmat[:] = 0.

        elif self.breakpoint_idx[n] < 0:
            log_transmat[:] = 0.

            for v in range(2):
                for w in range(2):
                    mult_const = -self.transition_penalty * self.p_allele[n, v, w]

                    for m in range(self.num_clones):
                        for ell in range(self.num_alleles):
                            self.add_log_transmat(log_transmat, m, ell, w, mult_const)

        else:
            log_transmat[:] = 0.

            for v in range(2):
                for w in range(2):
                    for m in range(self.num_clones):
                        mult_const = -self.transition_penalty * self.p_allele[n, v, w]

                        for ell in range(self.num_alleles):
                            if ell == v:
                                self.add_log_transmat_expectation_brk(
                                    log_transmat, m, ell, w,
                                    self.p_breakpoint[self.breakpoint_idx[n], m, :],
                                    self.breakpoint_orient[n],
                                    mult_const)

                            else:
                                self.add_log_transmat(log_transmat, m, ell, w, mult_const)

    cpdef void init_p_cn(self) except*:
        """ Initialize chains to something valid.
        """

        cdef int m, ell

        for _ in range(10):
            for m in range(1, self.num_clones):
                print np.asarray(self.x)
                print self.calculate_expected_x2()
                print m
                print self.get_cn()[:, m, :].T
                self.update_p_cn()
                # print self.x
                # print self.calculate_expected_x2()
                print self.get_cn()[:, m, :].T

            # self.update_phi()

    cpdef calculate_expected_x(self):
        cdef int n, i, m, ell, s
        cdef np.ndarray[np.float64_t, ndim=2] x = np.zeros((self.num_segments, self.num_measurements))
        for n in range(self.num_segments):
            for i in range(self.num_measurements):
                for m in range(self.num_clones):
                    for s in range(self.num_cn_states):
                        for ell in range(self.num_alleles):
                            x[n, i] += self.posterior_marginals[n, s] * self.h[m] * self.cn_states[s, m, ell] * self.w_mat[i, ell] * self.effective_lengths[n, i]
        return x

    cpdef calculate_expected_x2(self):
        cdef int n, i, m, ell
        cdef np.ndarray[np.float64_t, ndim=2] x = np.ones((self.num_segments, self.num_measurements))
        cs = self.get_cn()
        cs[:, 0, :] = 1
        for n in range(self.num_segments):
            for i in range(self.num_measurements):
                for m in range(self.num_clones):
                    for ell in range(self.num_alleles):
                        x[n, i] += self.h[m] * cs[n, m, ell] * self.w_mat[i, ell] * self.effective_lengths[n, i]
        return x

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef calculate_expected(self, int n, int s):
        """ Single update of all variational parameters.
        """
        cdef int i, m
        cdef np.ndarray[np.float64_t, ndim=1] expected = np.zeros((self.num_measurements,))

        for i in range(self.num_measurements):
            for m in range(self.num_clones):
                expected[i] += self.effective_lengths[n, i] * self.h[m] * self._ll_measured_copies(i, m, s)

        return expected

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.float64_t update(self) except*:
        """ Single update of all variational parameters.
        """

        elbo_prev = self.calculate_elbo()

        threshold = -1e-6

        print 'update_p_cn'
        self.update_p_cn()
        print self.get_cn()[:, :, :]
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

        print 'update_p_garbage'
        print np.asarray(self.p_garbage[:, :, 0]).sum(axis=0)
        self.update_p_garbage()
        print np.asarray(self.p_garbage[:, :, 0]).sum(axis=0)
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
        print (np.asarray(self.effective_lengths[:, 0]) / (np.asarray(self.effective_lengths[:, 2]) + 1)).mean()
        self.update_phi()
        print (np.asarray(self.effective_lengths[:, 0]) / (np.asarray(self.effective_lengths[:, 2]) + 1)).mean()
        elbo = self.calculate_elbo()
        print 'elbo diff: {:.10f}'.format(elbo - elbo_prev)
        if elbo - elbo_prev < threshold:
            raise Exception('elbo error!!!!')
        elbo_prev = elbo

        print 'update_a', np.asarray(self.a)
        self.update_a()
        print np.asarray(self.a)
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
    cdef np.float64_t _lp_term_1(self, int n, int m, int s):
        """ Term 1 in prior expansion.
        """
        cdef int ell
        cdef np.float64_t ll = 0.
        for ell in range(self.num_alleles):
            ll += -self.true_lengths[n] * (self.num_clones - 2) * self.cn_states[s, m, ell]**2 / self.prior_variance
        return ll

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _lp_term_2(self, int n, int m, int m_, int s):
        """ Term 2 in prior expansion.
        """
        cdef int ell
        cdef np.float64_t ll = 0.
        for ell in range(self.num_alleles):
            ll += self.true_lengths[n] * 2. * self.cn_states[s, m, ell] * self.cn_states[s, m_, ell] / self.prior_variance
        return ll

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_measured_copies(self, int i, int m, int s):
        """ Term 2 in likelihood expansion.
        """
        cdef int ell
        cdef np.float64_t kappa = 0.
        for ell in range(self.num_alleles):
            kappa += self.cn_states[s, m, ell] * self.w_mat[i, ell]
        return kappa

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_0(self, int n, int i):
        """ Term 0 in likelihood expansion.
        """
        return -1. * log(self.likelihood_variance[n, i]) / 2.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_1(self, int n, int i):
        """ Term 1 in likelihood expansion.
        """
        return -1. * self.x[n, i]**2 / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_2(self, int n, int i, int m, int s):
        """ Term 2 in likelihood expansion.
        """
        return 2. * self.x[n, i] * self.effective_lengths[n, i] * self.h[m] * self._ll_measured_copies(i, m, s) / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_3(self, int n, int i, int m, int s):
        """ Term 3 in likelihood expansion.
        """
        return -1. * (self.effective_lengths[n, i] * self.h[m] * self._ll_measured_copies(i, m, s))**2 / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_4(self, int n, int i, int m, int m_, int s):
        """ Term 4 in likelihood expansion.
        """
        return -2. * self.effective_lengths[n, i]**2 * self.h[m] * self.h[m_] * self._ll_measured_copies(i, m, s) * self._ll_measured_copies(i, m_, s) / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_2_3(self, int n, int i, int m, int s):
        """ Term 2 + 3 in likelihood expansion.
        """
        return self._ll_term_2(n, i, m, s) + self._ll_term_3(n, i, m, s)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_2_h(self, int n, int i, int m, int s):
        """ Term 2 h coefficient in likelihood expansion.
        """
        return 2. * self.x[n, i] * self.effective_lengths[n, i] * self._ll_measured_copies(i, m, s) / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_3_h(self, int n, int i, int m, int s):
        """ Term 3 h coefficient in likelihood expansion.
        """
        return -1. * (self.effective_lengths[n, i] * self._ll_measured_copies(i, m, s))**2 / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_4_h(self, int n, int i, int m, int m_, int s):
        """ Term 4 h coefficient in likelihood expansion.
        """
        return -2. * self.effective_lengths[n, i]**2 * self._ll_measured_copies(i, m, s) * self._ll_measured_copies(i, m_, s) / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_2_l(self, int n, int i, int m, int s):
        """ Term 2 l coefficient in in likelihood expansion.
        """
        return 2. * self.x[n, i] * self.h[m] * self._ll_measured_copies(i, m, s) / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_3_l(self, int n, int i, int m, int s):
        """ Term 3 l coefficient in in likelihood expansion.
        """
        return -1. * (self.h[m] * self._ll_measured_copies(i, m, s))**2 / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_4_l(self, int n, int i, int m, int m_, int s):
        """ Term 4 l coefficient in in likelihood expansion.
        """
        return -2. * self.h[m] * self.h[m_] * self._ll_measured_copies(i, m, s) * self._ll_measured_copies(i, m_, s) / (2. * self.likelihood_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_0_a(self, int n, int i):
        """ Term 0 a coefficient in likelihood expansion (excluding a constant term).
        """
        return -1. / 2.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_1_a(self, int n, int i):
        """ Term 1 a coefficient in likelihood expansion.
        """
        return -1. * self.x[n, i]**2 / (2. * self.unscaled_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_2_a(self, int n, int i, int m, int s):
        """ Term 2 a coefficient in likelihood expansion.
        """
        return 2. * self.x[n, i] * self.effective_lengths[n, i] * self.h[m] * self._ll_measured_copies(i, m, s) / (2. * self.unscaled_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_3_a(self, int n, int i, int m, int s):
        """ Term 3 a coefficient in likelihood expansion.
        """
        return -1. * (self.effective_lengths[n, i] * self.h[m] * self._ll_measured_copies(i, m, s))**2 / (2. * self.unscaled_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_4_a(self, int n, int i, int m, int m_, int s):
        """ Term 4 a coefficient in likelihood expansion.
        """
        return -2. * self.effective_lengths[n, i]**2 * self.h[m] * self.h[m_] * self._ll_measured_copies(i, m, s) * self._ll_measured_copies(i, m_, s) / (2. * self.unscaled_variance[n, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef np.float64_t _ll_term_2_3_a(self, int n, int i, int m, int s):
        """ Term 2 + 3 a coefficient in likelihood expansion.
        """
        return self._ll_term_2_a(n, i, m, s) + self._ll_term_3_a(n, i, m, s)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void update_p_cn(self) except*:
        """ Update the parameters of the approximating HMM.
        """

        cdef np.ndarray[np.float64_t, ndim=2] alphas = np.empty((self.num_segments, self.num_cn_states))
        cdef np.ndarray[np.float64_t, ndim=2] betas = np.empty((self.num_segments, self.num_cn_states))

        cdef int n, m, s, m_, i
        cdef np.ndarray[np.float64_t, ndim=1] log_posterior_marginals = np.zeros((self.num_cn_states,))
        cdef np.ndarray[np.float64_t, ndim=2] log_joint_posterior_marginals = np.zeros((self.num_cn_states, self.num_cn_states))

        # Build the frame log probabilities of this chain
        for n in range(self.num_segments):
            for s in range(self.num_cn_states):
                self.framelogprob[n, s] = 0.

                for m in range(1, self.num_clones):

                    # Copy number prior, singleton factors
                    self.framelogprob[n, s] += self._lp_term_1(n, m, s)

                    # Copy number prior, pairwise factors
                    for m_ in range(1, self.num_clones):
                        if m_ <= m:
                            continue
                        self.framelogprob[n, s] += self._lp_term_2(n, m, m_, s)

                for m in range(self.num_clones):

                    # Likelihood, singleton factors
                    for i in range(self.num_measurements):
                        self.framelogprob[n, s] += (
                            self.p_garbage[n, i, 0] *
                            self._ll_term_2_3(n, i, m, s))

                    # Likelihood, pairwise factors
                    for i in range(self.num_measurements):
                        for m_ in range(self.num_clones):
                            if m_ <= m:
                                continue
                            self.framelogprob[n, s] += (
                                self.p_garbage[n, i, 0] *
                                self._ll_term_4(n, i, m, m_, s))

        # Build the log transition probabilities of this chain
        for n in range(0, self.num_segments - 1):
            self.calculate_log_transmat(n, self.log_transmat[n, :, :])

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
                for v in range(2):
                    for w in range(2):
                        mult_const = -self.transition_penalty * self.p_allele[n, v, w]

                        for ell in range(self.num_clones):
                            if ell == v:
                                self.add_log_breakpoint_p_expectation_cn(
                                    log_p_breakpoint[self.breakpoint_idx[n], m, :],
                                    self.joint_posterior_marginals[n, :, :],
                                    m, ell, w, self.breakpoint_orient[n], mult_const)

        for k in range(self.num_breakpoints):
            for m in range(self.num_clones):
                _exp_normalize(self.p_breakpoint[k, m, :], log_p_breakpoint[k, m, :])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_p_allele(self) except*:
        """ Update the rearranged allele approximating distribution.
        """

        cdef int v, n, m, ell

        cdef np.ndarray[np.float64_t, ndim=2] log_transmat = np.empty((self.num_cn_states, self.num_cn_states))
        cdef np.ndarray[np.float64_t, ndim=2] log_p_allele = np.empty((2, 2))

        for n in range(0, self.num_segments - 1):
            if self.breakpoint_idx[n] < 0:
                continue

            log_p_allele[:] = 0.

            for v in range(2):
                for w in range(2):
                    for m in range(self.num_clones):
                        for ell in range(self.num_alleles):
                            log_transmat[:] = 0.

                            if v == ell:
                                self.add_log_transmat_expectation_brk(
                                    log_transmat, m, ell, w,
                                    self.p_breakpoint[self.breakpoint_idx[n], m, :],
                                    self.breakpoint_orient[n],
                                    -self.transition_penalty)

                            else:
                                self.add_log_transmat(log_transmat, m, ell, w, -self.transition_penalty)

                            log_p_allele[v, w] += np.sum(
                                log_transmat * self.joint_posterior_marginals[n, :, :])

            _exp_normalize2(self.p_allele[n, :, :], log_p_allele)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_p_garbage(self) except*:
        """ Update the read count garbage state approximating distribution.
        """

        cdef int n, i, m, s, m_
        cdef np.ndarray[np.float64_t, ndim=2] log_p_garbage = np.empty((3, 2))
        cdef np.ndarray[np.float64_t, ndim=1] log_p_allele_garbage = np.empty((2,))

        for n in range(self.num_segments):
            log_p_garbage[:] = 0.

            for i in range(self.num_measurements):

                # Likelihood, constant factors
                log_p_garbage[i, 0] += self._ll_term_0(n, i)
                log_p_garbage[i, 0] += self._ll_term_1(n, i)

                # Likelihood, singleton factors
                for m in range(self.num_clones):
                    for s in range(self.num_cn_states):
                        log_p_garbage[i, 0] += (
                            self.posterior_marginals[n, s]
                            * self._ll_term_2_3(n, i, m, s))

                # Likelihood, pairwise factors
                for m in range(self.num_clones):
                    for m_ in range(self.num_clones):
                        if m_ <= m:
                            continue
                        for s in range(self.num_cn_states):
                            log_p_garbage[i, 0] += (
                                self.posterior_marginals[n, s]
                                * self._ll_term_4(n, i, m, m_, s))

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
        """ Update haploid read depths.
        """

        cdef int n, m, i, s, m_
        cdef np.ndarray[np.float64_t, ndim=1] coeff_1
        cdef np.ndarray[np.float64_t, ndim=2] coeff_2

        coeff_1 = np.zeros((self.num_clones,))
        coeff_2 = np.zeros((self.num_clones, self.num_clones))

        for n in range(self.num_segments):
            for m in range(self.num_clones):
                for i in range(self.num_measurements):
                    for s in range(self.num_cn_states):
                        coeff_1[m] += (
                            self.posterior_marginals[n, s] *
                            self.p_garbage[n, i, 0] *
                            self._ll_term_2_h(n, i, m, s))

                        coeff_2[m, m] += (
                            self.posterior_marginals[n, s] *
                            self.p_garbage[n, i, 0] *
                            self._ll_term_3_h(n, i, m, s))

        for n in range(self.num_segments):
            for m in range(self.num_clones):
                for m_ in range(self.num_clones):
                    if m_ <= m:
                        continue
                    for i in range(self.num_measurements):
                        for s in range(self.num_cn_states):
                            coeff_2[m, m_] += (
                                self.posterior_marginals[n, s] *
                                self.p_garbage[n, i, 0] *
                                self._ll_term_4_h(n, i, m, m_, s))

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

        cdef int n, i, m, s, m_
        cdef np.float64_t coeff_1, coeff_2, phi

        for n in range(self.num_segments):
            coeff_1 = 0.
            coeff_2 = 0.

            for i in range(2):
                for m in range(self.num_clones):
                    for s in range(self.num_cn_states):
                        coeff_1 += (
                            self.posterior_marginals[n, s] *
                            self.p_garbage[n, i, 0] *
                            self._ll_term_2_l(n, i, m, s))

                        coeff_2 += (
                            self.posterior_marginals[n, s] *
                            self.p_garbage[n, i, 0] *
                            self._ll_term_3_l(n, i, m, s))

            for i in range(2):
                for m in range(self.num_clones):
                    for m_ in range(self.num_clones):
                        if m_ <= m:
                            continue
                        for s in range(self.num_cn_states):
                            coeff_2 += (
                                self.posterior_marginals[n, s] *
                                self.p_garbage[n, i, 0] *
                                self._ll_term_4_l(n, i, m, m_, s))

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

        cdef int n, i, m, s, m_
        cdef np.float64_t[:] coeff_1 = np.zeros((self.num_measurements,))
        cdef np.float64_t[:] coeff_2 = np.zeros((self.num_measurements,))

        for i in range(self.num_measurements):
            for n in range(self.num_segments):

                coeff_1[i] += self.p_garbage[n, i, 0] * self._ll_term_0_a(n, i)
                coeff_2[i] += self.p_garbage[n, i, 0] * self._ll_term_1_a(n, i)

                for m in range(self.num_clones):
                    for s in range(self.num_cn_states):
                        coeff_2[i] += (
                            self.posterior_marginals[n, s] *
                            self.p_garbage[n, i, 0] *
                            self._ll_term_2_3_a(n, i, m, s))

                for m in range(self.num_clones):
                    for m_ in range(self.num_clones):
                        if m_ <= m:
                            continue
                        for s in range(self.num_cn_states):
                            coeff_2[i] += (
                                self.posterior_marginals[n, s] *
                                self.p_garbage[n, i, 0] *
                                self._ll_term_4_a(n, i, m, m_, s))

        self.a[0] = (coeff_2[0] + coeff_2[1]) / (coeff_1[0] + coeff_1[1])
        self.a[1] = (coeff_2[0] + coeff_2[1]) / (coeff_1[0] + coeff_1[1])
        self.a[2] = coeff_2[2] / coeff_1[2]

        for i in range(self.num_measurements):
            for n in range(self.num_segments):
                self.likelihood_variance[n, i] = self.unscaled_variance[n, i] * self.a[i]

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

        cdef np.ndarray[np.float64_t, ndim=2] log_transmat = np.empty((self.num_cn_states, self.num_cn_states))

        cdef int n, i, m, s, m_
        cdef np.float64_t energy = 0.

        for n in range(self.num_segments):
            for s in range(self.num_cn_states):
                for m in range(1, self.num_clones):

                    # Copy number prior, singleton factors
                    energy += self.posterior_marginals[n, s] * self._lp_term_1(n, m, s)

                    # Copy number prior, pairwise factors
                    for m_ in range(1, self.num_clones):
                        if m_ <= m:
                            continue
                        energy += self.posterior_marginals[n, s] * self._lp_term_2(n, m, m_, s)

        for n in range(self.num_segments):
            for i in range(self.num_measurements):

                # Likelihood, constant factors
                energy += self.p_garbage[n, i, 0] * self._ll_term_0(n, i)
                energy += self.p_garbage[n, i, 0] * self._ll_term_1(n, i)

                # Likelihood, singleton factor
                for m in range(self.num_clones):
                    for s in range(self.num_cn_states):
                        energy += (
                            self.posterior_marginals[n, s] *
                            self.p_garbage[n, i, 0] *
                            self._ll_term_2_3(n, i, m, s))

                # Likelihood, pairwise factor
                for m in range(self.num_clones):
                    for m_ in range(self.num_clones):
                        if m_ <= m:
                            continue
                        for s in range(self.num_cn_states):
                            energy += (
                                self.posterior_marginals[n, s] *
                                self.p_garbage[n, i, 0] *
                                self._ll_term_4(n, i, m, m_, s))

        # Transitions factor
        for n in range(0, self.num_segments - 1):
            self.calculate_log_transmat(n, log_transmat)
            energy += np.sum(self.joint_posterior_marginals[n, :, :] * log_transmat)

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void infer_cn(self, np.ndarray[np.int64_t, ndim=3] cn) except*:
        """ Infer optimal copy number state sequence.
        """
        cdef np.ndarray[np.int64_t, ndim=1] state_sequence = np.zeros((self.num_segments,), dtype=np.int64)

        max_product(self.framelogprob[:, :], self.log_transmat[:, :, :], state_sequence)

        for n in range(self.num_segments):
            for m in range(self.num_clones):
                for ell in range(self.num_alleles):
                    cn[n, m, ell] = self.cn_states[state_sequence[n], m, ell]

    def get_cn(self):
        """ Get optimal copy number.
        """
        cdef np.ndarray[np.int64_t, ndim=3] cn = np.zeros((self.num_segments, self.num_clones, self.num_alleles), dtype=np.int64)
        self.infer_cn(cn)
        return cn


@cython.boundscheck(False)
@cython.wraparound(False)
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
@cython.wraparound(True)
cpdef np.float64_t max_product(
        np.float64_t[:, :] framelogprob,
        np.float64_t[:, :, :] log_transmat,
        np.int64_t[:] state_sequence) except*:

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
