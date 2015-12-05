import sys
import os
import unittest
import copy
import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import statsmodels.tools.numdiff

import remixt.simulations.simple as sim_simple
import remixt.simulations.experiment as sim_experiment
import remixt.likelihood as likelihood
import remixt.tests.unopt.likelihood as likelihood_unopt


np.random.seed(2014)


def assert_grad_correct(func, grad, x0, *args, **kwargs):
    """ Assert correct gradiant compared to finite difference approximation
    """

    analytic_fprime = grad(x0, *args)
    approx_fprime = statsmodels.tools.numdiff.approx_fprime_cs(x0, func, args=args)

    np.testing.assert_almost_equal(analytic_fprime, approx_fprime, 5)


class likelihood_unittest(unittest.TestCase):

    def generate_simple_data(self, total_cn):

        N = 100
        M = 3
        r = np.array([75, 75, 75])

        l = np.random.uniform(low=100000, high=1000000, size=N)
        phi = np.random.uniform(low=0.2, high=0.4, size=N)
        p = np.vstack([phi, phi, np.ones(phi.shape)]).T

        cn = sim_simple.generate_cn(N, M, 2.0, 0.5, 0.5, 2)
        h = np.random.uniform(low=0.5, high=2.0, size=M)

        likelihood_model = likelihood.ReadCountLikelihood()
        likelihood_model.h = h
        likelihood_model.phi = phi

        mu = likelihood_model.expected_read_count(l, cn)

        nb_p = mu / (r + mu)

        x = np.array([np.random.negative_binomial(r, 1.-a) for a in nb_p])
        x = x.reshape(nb_p.shape)

        if not total_cn:
            p = p[:,0:2]
            r = r[0:2]
            x = x[:,0:2]

        return cn, h, l, phi, r, x


    def generate_count_data(self):

        N = 100
        r = 75.

        mu = np.random.uniform(low=100000, high=1000000, size=N)

        nb_p = mu / (r + mu)
        x = np.random.negative_binomial(r, 1.-nb_p)

        return mu, x


    def test_expected_read_count_opt(self):

        for total_cn in (True, False):
            cn, h, l, phi, r, x = self.generate_simple_data(total_cn)

            emission = likelihood_unopt.ReadCountLikelihood(total_cn=total_cn)
            emission.h = h
            emission.phi = phi

            unopt = emission.expected_read_count_unopt(l, cn)
            opt = emission.expected_read_count(l, cn)

            error = np.sum(unopt - opt)

            self.assertAlmostEqual(error, 0.0, places=3)


    def test_log_likelihood_cn_partial_phi_opt(self):

        for total_cn in (True, False):
            cn, h, l, phi, r, x = self.generate_simple_data(total_cn)

            emission = likelihood_unopt.NegBinLikelihood(total_cn=total_cn)
            emission.h = h
            emission.phi = phi
            emission.r = r

            unopt = emission._log_likelihood_partial_phi_unopt(x, l, cn)
            opt = emission._log_likelihood_partial_phi(x, l, cn)

            error = np.sum(unopt - opt)

            self.assertAlmostEqual(error, 0.0, places=3)


    def test_log_likelihood_cn_negbin_opt(self):

        mu, x = self.generate_count_data()

        dist = likelihood_unopt.NegBinDistribution()

        unopt = dist.log_likelihood_unopt(x, mu)
        opt = dist.log_likelihood(x, mu)

        error = np.sum(unopt - opt)

        self.assertAlmostEqual(error, 0.0, places=3)


    def test_log_likelihood_cn_poisson_opt(self):

        mu, x = self.generate_count_data()

        dist = likelihood_unopt.PoissonDistribution()

        unopt = dist.log_likelihood_unopt(x, mu)
        opt = dist.log_likelihood(x, mu)

        error = np.sum(unopt - opt)

        self.assertAlmostEqual(error, 0.0, places=3)


    def test_log_likelihood_cn_negbin_partial_h_opt(self):

        for total_cn in (True, False):
            cn, h, l, phi, r, x = self.generate_simple_data(total_cn)

            emission = likelihood_unopt.NegBinLikelihood(total_cn=total_cn)
            emission.h = h
            emission.phi = phi
            emission.r = r

            unopt = emission._log_likelihood_partial_h_unopt(x, l, cn)
            opt = emission._log_likelihood_partial_h(x, l, cn)

            error = np.sum(unopt - opt)

            self.assertAlmostEqual(error, 0.0, places=3)


    def test_log_likelihood_cn_negbin_partial_r_opt(self):

        mu, x = self.generate_count_data()

        dist = likelihood_unopt.NegBinDistribution()

        unopt = dist.log_likelihood_partial_r_unopt(x, mu)
        opt = dist.log_likelihood_partial_r(x, mu)

        error = np.sum(unopt - opt)

        self.assertAlmostEqual(error, 0.0, places=3)


    def test_log_likelihood_cn_poisson_partial_h_opt(self):

        for total_cn in (True, False):
            cn, h, l, phi, r, x = self.generate_simple_data(total_cn)

            emission = likelihood_unopt.PoissonLikelihood(total_cn=total_cn)
            emission.h = h
            emission.phi = phi

            unopt = emission._log_likelihood_partial_h_unopt(x, l, cn)
            opt = emission._log_likelihood_partial_h(x, l, cn)

            error = np.sum(unopt - opt)

            self.assertAlmostEqual(error, 0.0, places=3)


    def test_log_likelihood_cn_partial_phi(self):

        for total_cn in (True, False):
            cn, h, l, phi, r, x = self.generate_simple_data(total_cn)

            emission = likelihood.NegBinLikelihood(total_cn=total_cn)
            emission.h = h
            emission.phi = phi
            emission.r = r

            def evaluate_log_likelihood(phi, x, l, cn):
                emission.phi = phi
                return np.sum(emission.log_likelihood(x, l, cn))

            def evaluate_log_likelihood_partial_phi(phi, x, l, cn):
                emission.phi = phi
                return emission._log_likelihood_partial_phi(x, l, cn)[:,0]

            assert_grad_correct(evaluate_log_likelihood,
                evaluate_log_likelihood_partial_phi, phi,
                x, l, cn)


    def test_log_likelihood_cn_negbin_partial_h(self):

        for total_cn in (True, False):
            cn, h, l, phi, r, x = self.generate_simple_data(total_cn)

            def evaluate_log_likelihood(h, x, l, cn, phi):
                emission = likelihood.NegBinLikelihood(total_cn=total_cn)
                emission.h = h
                emission.phi = phi
                emission.r = r
                return emission.log_likelihood(x, l, cn)

            def evaluate_log_likelihood_partial_h(h, x, l, cn, phi):
                emission = likelihood.NegBinLikelihood(total_cn=total_cn)
                emission.h = h
                emission.phi = phi
                emission.r = r
                return emission._log_likelihood_partial_h(x, l, cn)

            assert_grad_correct(evaluate_log_likelihood,
                evaluate_log_likelihood_partial_h, h,
                x, l, cn, phi)


    def test_log_likelihood_cn_negbin_partial_r(self):

        mu, x = self.generate_count_data()

        dist = likelihood_unopt.NegBinDistribution()

        def evaluate_log_likelihood(r):
            dist.r = r
            return dist.log_likelihood(x, mu)

        def evaluate_log_likelihood_partial_r(r):
            dist.r = r
            return dist.log_likelihood_partial_r(x, mu)[:,None]

        r = np.array([75.])

        assert_grad_correct(evaluate_log_likelihood,
            evaluate_log_likelihood_partial_r, r)


    def test_log_likelihood_cn_poisson_partial_h(self):

        for total_cn in (True, False):
            cn, h, l, phi, r, x = self.generate_simple_data(total_cn)

            def evaluate_log_likelihood(h, x, l, cn, phi):
                emission = likelihood.PoissonLikelihood(total_cn=total_cn)
                emission.h = h
                emission.phi = phi
                return emission.log_likelihood(x, l, cn)

            def evaluate_log_likelihood_partial_h(h, x, l, cn, phi):
                emission = likelihood.PoissonLikelihood(total_cn=total_cn)
                emission.h = h
                emission.phi = phi
                return emission._log_likelihood_partial_h(x, l, cn)

            assert_grad_correct(evaluate_log_likelihood,
                evaluate_log_likelihood_partial_h, h,
                x, l, cn, phi)


if __name__ == '__main__':
    unittest.main()


