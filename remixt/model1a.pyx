# cython: profile=True
# cython: initializedcheck=False
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
cdef np.float64_t _max3(np.float64_t[:, :, :] values):
    cdef np.float64_t vmax = _NINF
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            for k in range(values.shape[2]):
                if values[i, j, k] > vmax:
                    vmax = values[i, j, k]
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


@cython.boundscheck(True)
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
cdef np.float64_t _logsum3(np.float64_t[:, :, :] X):
    cdef np.float64_t vmax = _max3(X)
    cdef np.float64_t power_sum = 0

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                power_sum += exp(X[i, j, k]-vmax)

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


@cython.boundscheck(False)
@cython.cdivision(True)
cdef np.float64_t _lognormpdf(np.float64_t x, np.float64_t mu, np.float64_t v):
    return -0.5 * log(2 * _PI * v) - (x - mu)**2 / (2. * v)


def create_remixt_model(
    num_clones,
    num_segments,
    num_breakpoints,
    num_alleles,
    cn_max,
    cn_diff_max,
    normal_contamination,
    num_cn_states,
    cn_states,
    num_brk_states,
    brk_states,
    is_telomere,
    breakpoint_idx,
    breakpoint_orient,
    breakpoint_side,
    transition_penalty,
    p_breakpoint,
    hmm_log_norm_const,
    framelogprob,
    log_transmat,
    posterior_marginals,
    joint_posterior_marginals,
):
    model = RemixtModel(
        num_clones,
        num_segments,
        num_breakpoints,
        cn_max,
        cn_diff_max,
        normal_contamination,
        is_telomere,
        breakpoint_idx,
        breakpoint_orient,
        transition_penalty,
    )
    
    model.num_clones = num_clones
    model.num_segments = num_segments
    model.num_breakpoints = num_breakpoints
    model.num_alleles = num_alleles
    model.cn_max = cn_max
    model.cn_diff_max = cn_diff_max
    model.normal_contamination = normal_contamination
    model.num_cn_states = num_cn_states
    model.cn_states = cn_states
    model.num_brk_states = num_brk_states
    model.brk_states = brk_states
    model.is_telomere = is_telomere
    model.breakpoint_idx = breakpoint_idx
    model.breakpoint_orient = breakpoint_orient
    model.breakpoint_side = breakpoint_side
    model.transition_penalty = transition_penalty
    model.p_breakpoint = p_breakpoint
    model.hmm_log_norm_const = hmm_log_norm_const
    model.framelogprob = framelogprob
    model.log_transmat = log_transmat
    model.posterior_marginals = posterior_marginals
    model.joint_posterior_marginals = joint_posterior_marginals
    
    return model


cdef class RemixtModel:
    cdef public int num_clones
    cdef public int num_segments
    cdef public int num_breakpoints
    cdef public int num_alleles
    cdef public int cn_max
    cdef public int cn_diff_max
    cdef public bint normal_contamination
    cdef public int num_cn_states
    cdef public np.int64_t[:, :, :] cn_states
    cdef public np.int64_t[:, :] cn_states_total
    cdef public int num_brk_states
    cdef public np.int64_t[:, :] brk_states

    cdef public np.int64_t[:] is_telomere
    cdef public np.int64_t[:] breakpoint_idx
    cdef public np.int64_t[:] breakpoint_orient
    cdef public np.int64_t[:] breakpoint_side
    cdef public np.float64_t transition_penalty

    cdef public np.float64_t[:, :] p_breakpoint

    cdef public np.float64_t hmm_log_norm_const
    cdef public np.float64_t[:, :] framelogprob
    cdef public np.float64_t[:, :, :] log_transmat
    cdef public np.float64_t[:, :] posterior_marginals
    cdef public np.float64_t[:, :, :] joint_posterior_marginals

    def __cinit__(self,
        int num_clones, int num_segments,
        int num_breakpoints, int cn_max,
        int cn_diff_max, bint normal_contamination,
        np.ndarray[np.int64_t, ndim=1] is_telomere,
        np.ndarray[np.int64_t, ndim=1] breakpoint_idx,
        np.ndarray[np.int64_t, ndim=1] breakpoint_orient,
        np.float64_t transition_penalty):

        self.num_clones = num_clones
        self.num_segments = num_segments
        self.num_breakpoints = num_breakpoints
        self.cn_max = cn_max
        self.cn_diff_max = cn_diff_max
        self.normal_contamination = normal_contamination
        self.num_alleles = 2
        self.cn_states = self.create_cn_states(self.num_clones, self.num_alleles, self.cn_max, self.cn_diff_max, self.normal_contamination)
        self.num_cn_states = self.cn_states.shape[0]
        self.brk_states = self.create_brk_states(self.num_clones, self.cn_max, self.cn_diff_max)
        self.num_brk_states = self.brk_states.shape[0]

        # Create total states for convenience
        self.cn_states_total = np.zeros((self.num_cn_states, self.num_clones), dtype=np.int64)
        for s in range(self.num_cn_states):
            for m in range(self.num_clones):
                for ell in range(self.num_alleles):
                    self.cn_states_total[s, m] += self.cn_states[s, m, ell]

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
                if not (brk.max() == 1 and brk.min() == 0):
                    continue
                self.p_breakpoint[k, s_b] = 1.
        self.p_breakpoint /= np.sum(self.p_breakpoint, axis=-1)[:, np.newaxis]

        self.hmm_log_norm_const = 0.
        self.framelogprob = np.ones((self.num_segments, self.num_cn_states))
        self.log_transmat = np.zeros((self.num_segments - 1, self.num_cn_states, self.num_cn_states))
        self.posterior_marginals = np.zeros((self.num_segments, self.num_cn_states))
        self.joint_posterior_marginals = np.zeros((self.num_segments - 1, self.num_cn_states, self.num_cn_states))

        # Initialize to something valid
        self.posterior_marginals[:] = 1.
        self.posterior_marginals /= np.sum(self.posterior_marginals, axis=-1)[:, np.newaxis]
        self.joint_posterior_marginals[:] = 1.
        self.joint_posterior_marginals /= np.sum(self.joint_posterior_marginals, axis=(-2, -1))[:, np.newaxis, np.newaxis]
        
    def __reduce__(self):
        state = (
            self.num_clones,
            self.num_segments,
            self.num_breakpoints,
            self.num_alleles,
            self.cn_max,
            self.cn_diff_max,
            self.normal_contamination,
            self.num_cn_states,
            np.asarray(self.cn_states),
            self.num_brk_states,
            np.asarray(self.brk_states),
            np.asarray(self.is_telomere),
            np.asarray(self.breakpoint_idx),
            np.asarray(self.breakpoint_orient),
            np.asarray(self.breakpoint_side),
            np.asarray(self.transition_penalty),
            np.asarray(self.p_breakpoint),
            self.hmm_log_norm_const,
            np.asarray(self.framelogprob),
            np.asarray(self.log_transmat),
            np.asarray(self.posterior_marginals),
            np.asarray(self.joint_posterior_marginals),
        )

        return (create_remixt_model, state)

    def create_cn_states(self, num_clones, num_alleles, cn_max, cn_diff_max, normal_contamination):
        """ Create a list of allele specific copy number states.
        """
        num_tumour_vars = (num_clones - 1) * num_alleles
        
        if normal_contamination:
            normal_cn = (1, 1)
        else:
            normal_cn = (0, 0)

        cn_states = list()
        for cn in itertools.product(range(cn_max + 1), repeat=num_tumour_vars):
            cn = np.array(normal_cn + cn).reshape((num_clones, num_alleles))

            if not np.all(cn.sum(axis=1) <= cn_max):
                continue

            if not np.all(cn[1:, :].max(axis=0) - cn[1:, :].min(axis=0) <= cn_diff_max):
                continue

            # Discard states for which minor copy number is greater than or
            # equal to major copy number for all clones, and minor is strictly
            # greater than major for at least one clone
            if np.all(cn[1:, 1] >= cn[1:, 0]) and np.any(cn[1:, 1] > cn[1:, 0]):
                continue

            cn_states.append(cn)

        cn_states = np.array(cn_states)

        return cn_states

    def create_brk_states(self, num_clones, cn_max, cn_diff_max):
        """ Create a list of allele specific breakpoint copy number states.
        """
        num_tumour_vars = num_clones - 1
            
        normal_cn = (0,)

        brk_states = list()
        for cn in itertools.product(range(cn_max + 1), repeat=num_tumour_vars):
            cn = np.array(normal_cn + cn).reshape((num_clones,))

            if not np.all(cn <= cn_max):
                continue

            if not np.all(cn[1:].max() - cn[1:].min() <= cn_diff_max):
                continue

            brk_states.append(cn)

        brk_states = np.array(brk_states)

        return brk_states

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void add_log_transmat_total(self, np.float64_t[:, :] log_transmat, int m, np.float64_t mult_const) except*:
        """ Add total copy number log transition matrix for no breakpoint.
        """

        cdef int s_1, s_2

        for s_1 in range(self.num_cn_states):
            for s_2 in range(self.num_cn_states):
                log_transmat[s_1, s_2] += mult_const * fabs(self.cn_states_total[s_1, m] - self.cn_states_total[s_2, m])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void add_log_transmat_allele(self, np.float64_t[:, :] log_transmat, np.float64_t mult_const) except*:
        """ Add allele specific copy number log transition matrix for no breakpoint.
        """

        cdef int s_1, s_2, flip, m, allele
        cdef np.ndarray[np.float64_t, ndim=1] allele_cn_change = np.zeros((2,))

        for s_1 in range(self.num_cn_states):
            for s_2 in range(self.num_cn_states):
                for flip in range(2):
                    allele_cn_change[flip] = 0.
                    for m in range(self.num_clones):
                        for allele in range(self.num_alleles):
                            if flip == 1:
                                other_allele = 1 - allele
                            else:
                                other_allele = allele
                            allele_cn_change[flip] += fabs(self.cn_states[s_1, m, allele] - self.cn_states[s_2, m, other_allele])
                        allele_cn_change[flip] -= fabs(self.cn_states_total[s_1, m] - self.cn_states_total[s_2, m])
                log_transmat[s_1, s_2] += mult_const * min(allele_cn_change[0], allele_cn_change[1])

    @cython.boundscheck(False)
    @cython.wraparound(True)
    cdef void add_log_transmat_expectation_brk(self, np.float64_t[:, :] log_transmat, int m,
                                          np.float64_t[:] p_breakpoint, int breakpoint_orient,
                                          np.float64_t mult_const) except*:
        """ Add expected log transition matrix wrt breakpoint
        probability for the total copy number case.
        """

        cdef int d, s_b, s_1, s_2
        cdef np.ndarray[np.float64_t, ndim=1] p_b

        p_b = np.zeros((self.cn_max + 1) * 2)

        for d in range(-self.cn_max - 1, self.cn_max + 2):
            for s_b in range(self.num_brk_states):
                p_b[d] += p_breakpoint[s_b] * fabs(d - breakpoint_orient * self.brk_states[s_b, m])

        for s_1 in range(self.num_cn_states):
            for s_2 in range(self.num_cn_states):
                log_transmat[s_1, s_2] += mult_const * p_b[self.cn_states_total[s_1, m] - self.cn_states_total[s_2, m]]

    @cython.boundscheck(False)
    @cython.wraparound(True)
    cdef void add_log_breakpoint_p_expectation_cn(self, np.float64_t[:] log_breakpoint_p, np.float64_t[:, :] p_cn,
                                             int m, int breakpoint_orient, np.float64_t mult_const) except*:
        """ Calculate the expected log transition matrix wrt pairwise
        copy number probability.
        """

        cdef int d, s_1, s_2, s_b
        cdef np.ndarray[np.float64_t, ndim=1] p_d

        p_d = np.zeros(((self.cn_max + 1) * 2,))

        for s_1 in range(self.num_cn_states):
            for s_2 in range(self.num_cn_states):
                d = self.cn_states_total[s_1, m] - self.cn_states_total[s_2, m]
                p_d[d] += p_cn[s_1, s_2]

        for s_b in range(self.num_brk_states):
            for d in range(-self.cn_max - 1, self.cn_max + 2):
                log_breakpoint_p[s_b] += mult_const * p_d[d] * fabs(d - breakpoint_orient * self.brk_states[s_b, m])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calculate_log_transmat_regular(self, int n, np.float64_t[:, :] log_transmat) except*:
        cdef int m

        log_transmat[:] = 0.

        for m in range(self.num_clones):
            self.add_log_transmat_total(log_transmat, m, -self.transition_penalty)

        self.add_log_transmat_allele(log_transmat, -self.transition_penalty)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calculate_log_transmat_breakpoint(self, int n, np.float64_t[:, :] log_transmat) except*:
        cdef int m
        cdef np.float64_t mult_const

        log_transmat[:] = 0.

        for m in range(self.num_clones):
            self.add_log_transmat_expectation_brk(
                log_transmat, m,
                self.p_breakpoint[self.breakpoint_idx[n], :],
                self.breakpoint_orient[n],
                -self.transition_penalty)

        self.add_log_transmat_allele(log_transmat, -self.transition_penalty)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calculate_log_transmat(self, int n, np.float64_t[:, :] log_transmat) except*:
        """ Calculate the log transition matrix given current breakpoint and
        allele probabilities.
        """

        cdef int m

        if self.is_telomere[n] > 0:
            log_transmat[:] = 0.

        elif self.breakpoint_idx[n] < 0:
            self.calculate_log_transmat_regular(n, log_transmat)

        else:
            self.calculate_log_transmat_breakpoint(n, log_transmat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void update_p_cn(self) except*:
        """ Update the parameters of the approximating HMM.
        """

        cdef np.ndarray[np.float64_t, ndim=2] alphas = np.empty((self.num_segments, self.num_cn_states))
        cdef np.ndarray[np.float64_t, ndim=2] betas = np.empty((self.num_segments, self.num_cn_states))

        cdef int n, s, s_
        cdef np.ndarray[np.float64_t, ndim=1] log_posterior_marginals = np.zeros((self.num_cn_states,))
        cdef np.ndarray[np.float64_t, ndim=2] log_joint_posterior_marginals = np.zeros((self.num_cn_states, self.num_cn_states))

        # Assume frame log probabilities of this chain
        assert self.framelogprob.shape[0] == self.num_segments
        assert self.framelogprob.shape[1] == self.num_cn_states

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

        cdef int n, m, k
        cdef np.ndarray[np.float64_t, ndim=2] log_p_breakpoint = np.zeros((self.num_breakpoints, self.num_brk_states))

        for n in range(0, self.num_segments - 1):
            if self.breakpoint_idx[n] < 0:
                continue

            for m in range(self.num_clones):
                self.add_log_breakpoint_p_expectation_cn(
                    log_p_breakpoint[self.breakpoint_idx[n], :],
                    self.joint_posterior_marginals[n, :, :],
                    m, self.breakpoint_orient[n],
                    -self.transition_penalty)

        for k in range(self.num_breakpoints):
            for m in range(self.num_clones):
                _exp_normalize(self.p_breakpoint[k, :], log_p_breakpoint[k, :])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.float64_t calculate_variational_entropy(self) except*:
        """ Calculate the entropy of the approximating distribution.
        """

        cdef np.float64_t entropy = 0.

        entropy += -np.sum(self.hmm_log_norm_const)
        entropy += np.sum(np.asarray(self.joint_posterior_marginals) * np.asarray(self.log_transmat))
        entropy += _entropy(np.asarray(self.p_breakpoint).flatten())

        return entropy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.float64_t calculate_variational_energy(self) except*:
        """ Calculate the expectation of the true distribution wrt the
        approximating distribution.
        """

        cdef np.ndarray[np.float64_t, ndim=2] log_transmat = np.empty((self.num_cn_states, self.num_cn_states))

        cdef int n, s, s_
        cdef np.float64_t energy = 0.

        # Transitions factor
        for n in range(0, self.num_segments - 1):
            self.calculate_log_transmat(n, log_transmat)
            for s in range(self.num_cn_states):
                for s_ in range(self.num_cn_states):
                    energy += self.joint_posterior_marginals[n, s, s_] * log_transmat[s, s_]

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
