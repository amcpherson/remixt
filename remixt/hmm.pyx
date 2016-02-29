# hmmlearn cython hmm procedures

# Copyright (c) 2014, hmmlearn authors and contributors:
#     * Ron Weiss
#     * Shiqiao Du
#     * Jaques Grobler
#     * David Cournapeau
#     * Fabian Pedregosa
#     * Gael Varoquaux
#     * Andreas Mueller
#     * Bertrand Thirion
#     * Daniel Nouri
#     * Gilles Louppe
#     * Jake Vanderplas
#     * John Benediktsson
#     * Lars Buitinck
#     * Mikhail Korobov
#     * Robert McGibbon
#     * Stefano Lattarini
#     * Vlad Niculae
#     * csytracy
#     * Alexandre Gramfort
#     * Sergei Lebedev
#     * Daniela Huppenkothen
#     * Christopher Farrow
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the {organization} nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



from libc.math cimport exp, log
import numpy as np
cimport numpy as np
cimport cython

np.import_array()

ctypedef np.float64_t dtype_t

cdef dtype_t _NINF = -np.inf

@cython.boundscheck(False)
cdef dtype_t _max(dtype_t[:] values):
    # find maximum value (builtin 'max' is unrolled for speed)
    cdef dtype_t value
    cdef dtype_t vmax = _NINF
    for i in range(values.shape[0]):
        value = values[i]
        if value > vmax:
            vmax = value
    return vmax

@cython.boundscheck(False)
cpdef dtype_t _logsum(dtype_t[:] X):
    cdef dtype_t vmax = _max(X)
    cdef dtype_t power_sum = 0

    for i in range(X.shape[0]):
        power_sum += exp(X[i]-vmax)

    return log(power_sum) + vmax


@cython.boundscheck(False)
def _forward(int n_observations, int n_components,
        np.ndarray[dtype_t, ndim=1] log_startprob,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=2] framelogprob,
        np.ndarray[dtype_t, ndim=2] fwdlattice):

    cdef int t, i, j
    cdef double logprob
    cdef np.ndarray[dtype_t, ndim = 1] work_buffer
    work_buffer = np.zeros(n_components)

    for i in range(n_components):
        fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]

    for t in range(1, n_observations):
        for j in range(n_components):
            for i in range(n_components):
                work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j]
            fwdlattice[t, j] = _logsum(work_buffer) + framelogprob[t, j]


@cython.boundscheck(False)
def _backward(int n_observations, int n_components,
        np.ndarray[dtype_t, ndim=1] log_startprob,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=2] framelogprob,
        np.ndarray[dtype_t, ndim=2] bwdlattice):

    cdef int t, i, j
    cdef double logprob
    cdef np.ndarray[dtype_t, ndim = 1] work_buffer
    work_buffer = np.zeros(n_components)

    for i in range(n_components):
        bwdlattice[n_observations - 1, i] = 0.0

    for t in range(n_observations - 2, -1, -1):
        for i in range(n_components):
            for j in range(n_components):
                work_buffer[j] = log_transmat[i, j] + framelogprob[t + 1, j] \
                    + bwdlattice[t + 1, j]
            bwdlattice[t, i] = _logsum(work_buffer)


@cython.boundscheck(False)
def _compute_lneta(int n_observations, int n_components,
        np.ndarray[dtype_t, ndim=2] fwdlattice,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=2] bwdlattice,
        np.ndarray[dtype_t, ndim=2] framelogprob,
        double logprob,
        np.ndarray[dtype_t, ndim=3] lneta):

    cdef int i, j, t
    for t in range(n_observations - 1):
        for i in range(n_components):
            for j in range(n_components):
                lneta[t, i, j] = fwdlattice[t, i] + log_transmat[i, j] \
                    + framelogprob[t + 1, j] + bwdlattice[t + 1, j] - logprob


@cython.boundscheck(False)
def _viterbi(int n_observations, int n_components,
        np.ndarray[dtype_t, ndim=1] log_startprob,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=2] framelogprob):

    cdef int t, max_pos
    cdef np.ndarray[dtype_t, ndim = 2] viterbi_lattice
    cdef np.ndarray[np.int_t, ndim = 1] state_sequence
    cdef dtype_t logprob
    cdef np.ndarray[dtype_t, ndim = 2] work_buffer

    # Initialization
    state_sequence = np.empty(n_observations, dtype=np.int)
    viterbi_lattice = np.zeros((n_observations, n_components))
    viterbi_lattice[0] = log_startprob + framelogprob[0]

    # Induction
    for t in range(1, n_observations):
        work_buffer = viterbi_lattice[t-1] + log_transmat.T
        viterbi_lattice[t] = np.max(work_buffer, axis=1) + framelogprob[t]

    # Observation traceback
    max_pos = np.argmax(viterbi_lattice[n_observations - 1, :])
    state_sequence[n_observations - 1] = max_pos
    logprob = viterbi_lattice[n_observations - 1, max_pos]

    for t in range(n_observations - 2, -1, -1):
        max_pos = np.argmax(viterbi_lattice[t, :] \
                + log_transmat[:, state_sequence[t + 1]])
        state_sequence[t] = max_pos

    return state_sequence, logprob
