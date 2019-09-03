import unittest
import itertools
import numpy as np
import scipy.misc

import remixt.tests.unopt.model
import remixt.vhmm
import remixt.model3
import remixt.tests.test_remixt
import remixt.likelihood
import remixt.cn_model


np.random.seed(2014)


cn_max = 3
num_clones = 3
num_alleles = 2


def create_cn_states():
    cn_states = list()

    num_tumour_vars = (num_clones - 1) * num_alleles

    for cn in itertools.product(range(cn_max + 1), repeat=num_tumour_vars):
        cn = np.array((1, 1) + cn).reshape((num_clones, num_alleles))

        if not np.all(cn.sum(axis=1) <= cn_max):
            continue

        if not np.all(cn[1:, :].max(axis=0) - cn[1:, :].min(axis=0) <= 1):
            continue

        cn_states.append(cn)

    cn_states = np.array(cn_states)

    return cn_states


def create_brk_states():
    brk_states = list()

    num_tumour_brk_vars = num_clones - 1

    for cn in itertools.product(range(cn_max + 1), repeat=num_tumour_brk_vars):
        cn = np.array((0,) + cn).reshape((num_clones,))

        brk_states.append(cn)

    brk_states = np.array(brk_states)

    return brk_states


def naive_norm_const(cn_states, framelogprob, model):
    num_segments = framelogprob.shape[0]
    num_cn_states = cn_states.shape[0]

    log_transmat = np.empty((num_segments, num_cn_states, num_cn_states))
    for seg_idx in range(num_segments - 1):
        model.calculate_log_transmat(seg_idx, log_transmat[seg_idx, :, :])

    norm_const = 0.
    for seg_states in itertools.product(range(num_cn_states), repeat=num_segments):
        log_prob = framelogprob[range(num_segments), seg_states].sum()
        for seg_idx in range(num_segments - 1):
            log_prob += log_transmat[seg_idx, seg_states[seg_idx], seg_states[seg_idx + 1]]
        norm_const += np.exp(log_prob)

    return norm_const


def naive_posterior_marginal(cn_states, framelogprob, model, query_seg_idx):
    num_segments = framelogprob.shape[0]
    num_cn_states = cn_states.shape[0]

    log_transmat = np.empty((num_segments, num_cn_states, num_cn_states))
    for seg_idx in range(num_segments - 1):
        model.calculate_log_transmat(seg_idx, log_transmat[seg_idx, :, :])

    posterior_marginal = np.zeros((num_cn_states,))
    for seg_states in itertools.product(range(num_cn_states), repeat=num_segments):
        log_prob = framelogprob[range(num_segments), seg_states].sum()
        for seg_idx in range(num_segments - 1):
            log_prob += log_transmat[seg_idx, seg_states[seg_idx], seg_states[seg_idx + 1]]
        posterior_marginal[seg_states[query_seg_idx]] += np.exp(log_prob)

    posterior_marginal /= posterior_marginal.sum()

    return posterior_marginal


num_replicates = 1


class model_unittest(unittest.TestCase):

    def test_total_transition_expectation_opt(self):
        for i in range(num_replicates):
            for orientation in (1, -1):
                p_breakpoint = np.random.dirichlet((4.,) * (cn_max + 1))

                tr_mat_1 = remixt.tests.unopt.model.total_transition_expectation_unopt_1(p_breakpoint, orientation, cn_max)
                tr_mat_2 = remixt.tests.unopt.model.total_transition_expectation_unopt_2(p_breakpoint, orientation, cn_max)
                tr_mat_3 = remixt.tests.unopt.model.total_transition_expectation_unopt_3(p_breakpoint, orientation, cn_max)
                tr_mat_4 = np.zeros((cn_max + 1, cn_max + 1))
                remixt.vhmm.calculate_log_transmat_expectation_brk_total(p_breakpoint, orientation, cn_max, tr_mat_4)

                error = np.sum(np.square(tr_mat_1 - tr_mat_2))
                self.assertAlmostEqual(error, 0.0, places=3)

                error = np.sum(np.square(tr_mat_1 - tr_mat_3))
                self.assertAlmostEqual(error, 0.0, places=3)

                error = np.sum(np.square(tr_mat_1 - tr_mat_4))
                self.assertAlmostEqual(error, 0.0, places=3)

    def test_allele_transition_expectation_opt(self):
        cn_states = create_cn_states()
        brk_states = create_brk_states()

        num_cn_states = cn_states.shape[0]

        for i in range(num_replicates):
            for brk_orient in (1, -1):
                p_breakpoint = np.random.dirichlet((4.,) * brk_states.shape[0])
                p_allele = np.random.dirichlet((4.,) * 2)

                tr_mat_1 = -1.0 * remixt.tests.unopt.model.allele_transition_expectation_unopt_1(p_breakpoint, p_allele, brk_orient, cn_states, brk_states)

                tr_mat_2 = -1.0 * remixt.tests.unopt.model.allele_transition_expectation_unopt_2(p_breakpoint, p_allele, brk_orient, cn_states, brk_states, cn_max)

                tr_mat_3 = np.zeros((num_cn_states, num_cn_states))
                for brk_allele in range(2):
                    remixt.vhmm.calculate_log_transmat_expectation_brk_allele(
                        tr_mat_3,
                        cn_states,
                        brk_states,
                        p_breakpoint,
                        brk_orient,
                        brk_allele,
                        cn_max,
                        -1.0 * p_allele[brk_allele])

                model = remixt.vhmm.RemixtModel(
                    3, 3, 1,
                    cn_states,
                    brk_states,
                    np.array([0, 0, 1], dtype=np.uint8),
                    np.array([0, 0, 0]),
                    np.array([brk_orient, brk_orient, brk_orient], dtype=np.int8),
                    cn_max,
                    1.0,
                )

                model.p_breakpoint[:] = p_breakpoint[np.newaxis, :]
                model.p_allele[:] = p_allele[np.newaxis, :]

                tr_mat_4 = np.zeros((num_cn_states, num_cn_states))
                model.calculate_log_transmat(0, tr_mat_4)

                error = np.sum(np.square(tr_mat_1 - tr_mat_2))
                self.assertAlmostEqual(error, 0.0, places=3)

                error = np.sum(np.square(tr_mat_1 - tr_mat_3))
                self.assertAlmostEqual(error, 0.0, places=3)

                error = np.sum(np.square(tr_mat_1 - tr_mat_4))
                self.assertAlmostEqual(error, 0.0, places=3)

    def test_allele_brk_expectation_opt(self):
        cn_states = create_cn_states()
        brk_states = create_brk_states()

        num_brk_states = brk_states.shape[0]

        for i in range(num_replicates):
            for brk_orient in (1, -1):
                p_cn = np.random.dirichlet((4.,) * cn_states.shape[0], size=cn_states.shape[0])
                p_allele = np.random.dirichlet((4.,) * 2)

                brk_expect_1 = -1.0 * remixt.tests.unopt.model.calculate_log_breakpoint_p_expectation_cn_unopt_1(p_cn, p_allele, brk_orient, cn_states, brk_states)

                brk_expect_2 = -1.0 * remixt.tests.unopt.model.calculate_log_breakpoint_p_expectation_cn_unopt_2(p_cn, p_allele, brk_orient, cn_states, brk_states, cn_max)

                brk_expect_3 = np.zeros((num_brk_states,))
                for brk_allele in range(2):
                    remixt.vhmm.calculate_log_breakpoint_p_expectation_cn_allele(
                        brk_expect_3,
                        cn_states,
                        brk_states,
                        p_cn,
                        brk_orient,
                        brk_allele,
                        cn_max,
                        -1.0 * p_allele[brk_allele])
                
                error = np.sum(np.square(brk_expect_1 - brk_expect_2))
                self.assertAlmostEqual(error, 0.0, places=3)

                error = np.sum(np.square(brk_expect_1 - brk_expect_3))
                self.assertAlmostEqual(error, 0.0, places=3)

    def test_sum_product(self):
        cn_states = create_cn_states()
        brk_states = create_brk_states()

        num_cn_states = cn_states.shape[0]

        num_segments = 3

        for i in range(num_replicates):

            model = remixt.vhmm.RemixtModel(
                3, num_segments, 0,
                cn_states,
                brk_states,
                np.array([0, 0, 1], dtype=np.uint8),
                np.array([-1] * 3),
                np.array([-1] * 3, dtype=np.int8),
                cn_max,
                1.0,
            )

            framelogprob = np.random.rand(num_segments, num_cn_states)

            alphas = np.zeros((num_segments, num_cn_states))
            betas = np.zeros((num_segments, num_cn_states))
            remixt.vhmm.sum_product(model, framelogprob, alphas, betas)

            for seg_idx in range(num_segments):
                posterior_marginal_1 = naive_posterior_marginal(cn_states, framelogprob, model, seg_idx)

                gamma = alphas[seg_idx] + betas[seg_idx]
                posterior_marginal_2 = np.exp(gamma - scipy.misc.logsumexp(gamma))
                posterior_marginal_2 /= posterior_marginal_2.sum()

                error = np.sum(np.square(posterior_marginal_1 - posterior_marginal_2))
                self.assertAlmostEqual(error, 0.0, places=3)

    def create_random_model(self, num_segments, num_clones, num_alleles):
        num_breakpoints = 5
        cn_max = 3

        l = np.random.uniform(low=100000, high=1000000, size=num_segments)

        cn = np.random.randint(cn_max, size=(num_segments, num_clones, num_alleles))
        h = np.random.uniform(low=0.5, high=2.0, size=num_clones)
        phi = np.random.uniform(low=0.2, high=0.4, size=num_segments)
        mu = remixt.likelihood.expected_read_count(l, cn, h, phi)

        is_telomere = np.zeros((num_segments,), dtype=np.int64)
        is_telomere[-1] = 1

        breakpoint_idx = np.zeros((num_segments,), dtype=np.int64) - 1
        breakends = np.random.choice(num_segments, size=num_breakpoints * 2)
        for i in range(num_breakpoints):
            breakpoint_idx[breakends[2 * i]] = i
            breakpoint_idx[breakends[2 * i + 1]] = i

        breakpoint_orient = np.random.choice([+1, -1], size=num_segments)

        model = remixt.model3.RemixtModel(
            num_clones, num_segments, num_breakpoints, cn_max,
            is_telomere,
            breakpoint_idx,
            breakpoint_orient,
            l,
            l,
            1.0,
        )

        model.h = h
        model.a = 0.05
        model.unscaled_variance = mu
        model.likelihood_variance = model.a * model.unscaled_variance
        model.prior_variance = 1e-5

        # Uniform initialization
        model.p_breakpoint = np.random.random(size=(num_breakpoints, num_clones, cn_max + 1))
        model.p_breakpoint /= model.p_breakpoint.sum(axis=2)[:, :, np.newaxis]

        model.p_allele = np.random.random(size=(num_segments, num_alleles))
        model.p_allele /= model.p_allele.sum(axis=1)[:, np.newaxis]

        model.p_obs_allele = np.random.random(size=(num_segments, num_alleles))
        model.p_obs_allele /= model.p_obs_allele.sum(axis=1)[:, np.newaxis]

        model.p_garbage = np.random.random(size=(num_segments, 3, 2))
        model.p_garbage[:, 0, 0] = (1. - model.prior_total_garbage)
        model.p_garbage[:, 0, 1] = model.prior_total_garbage
        model.p_garbage[:, 0, :] /= model.p_garbage[:, 0, :].sum(axis=1)[:, np.newaxis]

        model.p_garbage[:, 1, 0] = (1. - model.prior_allele_garbage)
        model.p_garbage[:, 1, 1] = model.prior_allele_garbage
        model.p_garbage[:, 1, :] /= model.p_garbage[:, 1, :].sum(axis=1)[:, np.newaxis]

        model.p_garbage[:, 2, 0] = (1. - model.prior_allele_garbage)
        model.p_garbage[:, 2, 1] = model.prior_allele_garbage
        model.p_garbage[:, 2, :] /= model.p_garbage[:, 2, :].sum(axis=1)[:, np.newaxis]

        for m in range(num_clones):
            for ell in range(num_alleles):
                model.posterior_marginals[m, ell, :, :] = np.random.random(size=(num_segments, cn_max + 1))
                model.posterior_marginals[m, ell, :, :] /= model.posterior_marginals[m, ell, :, :].sum(axis=1)[:, np.newaxis]

                model.joint_posterior_marginals[m, ell, :, :, :] = np.random.random(size=(num_segments, cn_max + 1, cn_max + 1))
                model.joint_posterior_marginals[m, ell, :, :, :] /= model.joint_posterior_marginals[m, ell, :, :, :].sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

        return model

    def test_random_model(self):
        num_segments = 5
        num_clones = 3
        num_alleles = 2

        model = self.create_random_model(num_segments, num_clones, num_alleles)
        x = np.random.randint(100000, size=(num_segments, 3))

        for m in range(num_clones):
            for ell in range(num_alleles):
                elbo_1 = model.calculate_elbo(x)
                model.update(x)
                elbo_2 = model.calculate_elbo(x)
                print (elbo_2 - elbo_1)
                self.assertTrue(elbo_2 - elbo_1 > -1e-10)

    def test_update(self):
        num_segments = 4

        for i in range(num_replicates):

            cn = np.array([[[1, 1],
                            [1, 1],
                            [1, 1]],
                           [[1, 1],
                            [1, 0],
                            [1, 1]],
                           [[1, 1],
                            [1, 1],
                            [1, 1]],
                           [[1, 1],
                            [0, 0],
                            [0, 0]]])

            h = np.array([0.2, 0.3, 0.1])
            l = np.array([10000., 30000., 20000., 10000.])
            phi = np.array([0.2, 0.2, 0.2, 0.2])
            mu = remixt.likelihood.expected_read_count(l, cn, h, phi)

            x = np.array([np.random.poisson(a) for a in mu])
            x = x.reshape(mu.shape)

            model = remixt.model3.RemixtModel(
                3, num_segments, 1, cn_max,
                x,
                np.array([0, 0, 0, 1]),
                np.array([0, 0, -1, -1]),
                np.array([+1, -1, 0, 0]),
                l,
                l,
                1.0,
            )

            print (h)

            model.prior_variance = 1e5

            model.h = np.array([0.2, 0.3, 0.1])

            print (np.asarray(model.cn_states))

            model.init_p_cn()

            print (model.posterior_marginals)
            print (model.get_cn())

            # model.posterior_marginals[0, :, :, :] = (
            #     np.array([[[0.01, 0.99],
            #                [0.01, 0.99],
            #                [0.01, 0.99],
            #                [0.01, 0.99]],
            #               [[0.01, 0.99],
            #                [0.01, 0.99],
            #                [0.01, 0.99],
            #                [0.01, 0.99]]]))

            elbo_prev = None
            for i in range(20):
                elbo = model.update()
                print ('elbo', elbo)
                print ('h', np.asarray(model.h))
                if elbo_prev is not None:
                    print ('diff:', elbo - elbo_prev)
                    self.assertTrue(elbo - elbo_prev > -1e-5)
                elbo_prev = elbo

            print (model.get_cn())
            print (np.asarray(model.p_garbage))
            print (np.argmax(model.p_garbage, axis=-1))

            brk_cn = np.argmax(model.p_breakpoint, axis=-1)
            self.assertTrue(np.all(brk_cn == np.array([0, 1, 0])))


    def test_update_swap(self):
        num_segments = 4

        for i in range(num_replicates):

            cn = np.array([[[1, 1],
                            [0, 1]],
                           [[1, 1],
                            [2, 1]],
                           [[1, 1],
                            [0, 1]],
                           [[1, 1],
                            [0, 1]]])

            h = np.array([0.2, 0.3])
            l = np.array([10000., 30000., 20000., 10000.])
            phi = np.array([0.2, 0.2, 0.2, 0.2])
            mu = remixt.likelihood.expected_read_count(l, cn, h, phi)

            x = np.array([np.random.poisson(a) for a in mu])
            x.sort(axis=1)
            x = x.reshape(mu.shape)

            model = remixt.model3.RemixtModel(
                2, num_segments, 1, cn_max,
                x,
                np.array([0, 0, 0, 1]),
                np.array([0, 0, -1, -1]),
                np.array([-1, +1, 0, 0]),
                l,
                l,
                1.0,
            )

            print (h)

            model.prior_variance = 1e5

            model.h = np.array([0.2, 0.3])

            model.init_p_cn()

            print (model.get_cn())

            # model.posterior_marginals[0, :, :, :] = (
            #     np.array([[[0.01, 0.99],
            #                [0.01, 0.99],
            #                [0.01, 0.99],
            #                [0.01, 0.99]],
            #               [[0.01, 0.99],
            #                [0.01, 0.99],
            #                [0.01, 0.99],
            #                [0.01, 0.99]]]))

            elbo_prev = None
            for i in range(20):
                elbo = model.update()
                print ('elbo', elbo)
                print ('h', model.h)
                if elbo_prev is not None:
                    print ('diff:', elbo - elbo_prev)
                    self.assertTrue(elbo - elbo_prev > -1e-5)
                elbo_prev = elbo

            print (model.get_cn())

            print (np.asarray(model.p_allele))

            for n in range(2):
                for v in range(2):
                    for w in range(2):
                        print (n, v, w, model.p_allele[n, v, w])

            print (np.asarray(model.p_breakpoint))

            brk_cn = np.argmax(model.p_breakpoint, axis=-1)
            print (brk_cn)

            self.assertTrue(np.all(brk_cn == np.array([0, 1, 0])))


if __name__ == '__main__':
    unittest.main()


