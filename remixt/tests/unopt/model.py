import numpy as np


def total_transition_expectation_unopt_1(p_breakpoint, brk_orient, cn_max):
    tr_mat = np.zeros((cn_max + 1, cn_max + 1))

    for cn_1 in range(cn_max + 1):
        for cn_2 in range(cn_max + 1):
            for b in range(cn_max + 1):
                tr_mat[cn_1, cn_2] += p_breakpoint[b] * abs(cn_1 - cn_2 - brk_orient * b)

    return tr_mat


def total_transition_expectation_unopt_2(p_breakpoint, brk_orient, cn_max):
    tr_mat = np.zeros((cn_max + 1, cn_max + 1))

    p_b_geq = np.zeros(p_breakpoint.shape[0] * 2)
    p_b_b_geq = np.zeros(p_breakpoint.shape[0] * 2)

    p_b_lt = np.zeros(p_breakpoint.shape[0] * 2)
    p_b_b_lt = np.zeros(p_breakpoint.shape[0] * 2)

    for d in range(-cn_max - 1, cn_max + 2):
        for b in range(cn_max + 1):
            if d - brk_orient * b >= 0:
                p_b_geq[d] += p_breakpoint[b]
                p_b_b_geq[d] += p_breakpoint[b] * b

            if d - brk_orient * b < 0:
                p_b_lt[d] += p_breakpoint[b]
                p_b_b_lt[d] += p_breakpoint[b] * b

    for cn_1 in range(cn_max + 1):
        for cn_2 in range(cn_max + 1):
            tr_mat[cn_1, cn_2] = (
                (cn_1 - cn_2) * p_b_geq[cn_1 - cn_2] - brk_orient * p_b_b_geq[cn_1 - cn_2] +
                (-cn_1 + cn_2) * p_b_lt[cn_1 - cn_2] + brk_orient * p_b_b_lt[cn_1 - cn_2]
            )

    return tr_mat


def total_transition_expectation_unopt_3(p_breakpoint, brk_orient, cn_max):
    tr_mat = np.zeros((cn_max + 1, cn_max + 1))

    p_b = np.zeros((cn_max + 1) * 2)

    for d in range(-cn_max - 1, cn_max + 2):
        for b in range(cn_max + 1):
            p_b[d] += p_breakpoint[b] * abs(d - brk_orient * b)

    for cn_1 in range(cn_max + 1):
        for cn_2 in range(cn_max + 1):
            tr_mat[cn_1, cn_2] = p_b[cn_1 - cn_2]

    return tr_mat


def allele_transition_expectation_unopt_1(p_breakpoint, p_allele, brk_orient, cn_states, brk_states):
    num_states = cn_states.shape[0]
    num_clones = cn_states.shape[1]
    num_alleles = cn_states.shape[2]

    tr_mat = np.zeros((num_states, num_states))

    for idx_1, cn_1 in enumerate(cn_states):
        for idx_2, cn_2 in enumerate(cn_states):
            for idx_b, cn_b in enumerate(brk_states):
                for m in range(num_clones):
                    for ell in range(num_alleles):
                        for brk_allele in range(num_alleles):
                            if ell == brk_allele:
                                tr_mat[idx_1, idx_2] += p_breakpoint[idx_b] * p_allele[brk_allele] * abs(cn_1[m, ell] - cn_2[m, ell] - brk_orient * cn_b[m])
                            else:
                                tr_mat[idx_1, idx_2] += p_breakpoint[idx_b] * p_allele[brk_allele] * abs(cn_1[m, ell] - cn_2[m, ell])

    return tr_mat


def allele_transition_expectation_allele_unopt_2(p_breakpoint, brk_orient, brk_allele, cn_states, brk_states, cn_max):
    num_states = cn_states.shape[0]
    num_clones = cn_states.shape[1]
    num_alleles = cn_states.shape[2]

    tr_mat = np.zeros((num_states, num_states))

    p_b_geq = np.zeros(((cn_max + 1) * 2, num_clones))
    p_b_b_geq = np.zeros(((cn_max + 1) * 2, num_clones))

    p_b_lt = np.zeros(((cn_max + 1) * 2, num_clones))
    p_b_b_lt = np.zeros(((cn_max + 1) * 2, num_clones))

    for m in range(num_clones):
        for d in range(-cn_max - 1, cn_max + 2):
            for idx_b, cn_b in enumerate(brk_states):
                if d - brk_orient * cn_b[m] >= 0:
                    p_b_geq[d, m] += p_breakpoint[idx_b]
                    p_b_b_geq[d, m] += p_breakpoint[idx_b] * cn_b[m]

                if d - brk_orient * cn_b[m] < 0:
                    p_b_lt[d, m] += p_breakpoint[idx_b]
                    p_b_b_lt[d, m] += p_breakpoint[idx_b] * cn_b[m]

    for idx_1, cn_1 in enumerate(cn_states):
        for idx_2, cn_2 in enumerate(cn_states):
            for m in range(num_clones):
                for ell in range(num_alleles):
                    d = cn_1[m, ell] - cn_2[m, ell]
                    if ell == brk_allele:
                        tr_mat[idx_1, idx_2] += (
                            d * p_b_geq[d, m] - brk_orient * p_b_b_geq[d, m] +
                            (-d) * p_b_lt[d, m] + brk_orient * p_b_b_lt[d, m]
                        )

                    else:
                        tr_mat[idx_1, idx_2] += abs(d)

    return tr_mat


def allele_transition_expectation_unopt_2(p_breakpoint, p_allele, brk_orient, cn_states, brk_states, cn_max):
    return (
        allele_transition_expectation_allele_unopt_2(p_breakpoint, brk_orient, 0, cn_states, brk_states, cn_max) * p_allele[0] +
        allele_transition_expectation_allele_unopt_2(p_breakpoint, brk_orient, 1, cn_states, brk_states, cn_max) * p_allele[1])


def calculate_log_breakpoint_p_expectation_cn_unopt_1(p_cn, p_allele, brk_orient, cn_states, brk_states):
    num_brk_states = brk_states.shape[0]
    num_clones = cn_states.shape[1]
    num_alleles = cn_states.shape[2]

    brk_expect = np.zeros((num_brk_states,))

    for idx_1, cn_1 in enumerate(cn_states):
        for idx_2, cn_2 in enumerate(cn_states):
            for idx_b, cn_b in enumerate(brk_states):
                for m in range(num_clones):
                    for ell in range(num_alleles):
                        for brk_allele in range(num_alleles):
                            if ell == brk_allele:
                                brk_expect[idx_b] += p_cn[idx_1, idx_2] * p_allele[brk_allele] * abs(cn_1[m, ell] - cn_2[m, ell] - brk_orient * cn_b[m])
                            else:
                                brk_expect[idx_b] += p_cn[idx_1, idx_2] * p_allele[brk_allele] * abs(cn_1[m, ell] - cn_2[m, ell])

    return brk_expect


def calculate_log_breakpoint_p_expectation_cn_unopt_2(p_cn, p_allele, brk_orient, cn_states, brk_states, cn_max):
    num_brk_states = brk_states.shape[0]
    num_clones = cn_states.shape[1]
    num_alleles = cn_states.shape[2]

    p_d = np.zeros(((cn_max + 1) * 2, num_clones, num_alleles))

    for idx_1, cn_1 in enumerate(cn_states):
        for idx_2, cn_2 in enumerate(cn_states):
            for m in range(num_clones):
                for ell in range(num_alleles):
                    d = cn_1[m, ell] - cn_2[m, ell]
                    p_d[d, m, ell] += p_cn[idx_1, idx_2]

    brk_expect = np.zeros((num_brk_states,))

    for idx_b, cn_b in enumerate(brk_states):
        for d in range(-cn_max - 1, cn_max + 2):
            for m in range(num_clones):
                for ell in range(num_alleles):
                    for brk_allele in range(num_alleles):
                        if ell == brk_allele:
                            brk_expect[idx_b] += p_d[d, m, ell] * p_allele[brk_allele] * abs(d - brk_orient * cn_b[m])
                        else:
                            brk_expect[idx_b] += p_d[d, m, ell] * p_allele[brk_allele] * abs(d)

    return brk_expect



