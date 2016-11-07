import sys
import os
import unittest
import copy
import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
from scipy.special import gammaln, betaln
import statsmodels.tools.numdiff

import remixt.simulations.simple
import remixt.tests.unopt.likelihood as likelihood_unopt
import remixt.likelihood
import remixt.paramlearn
import remixt.tests.utils


np.random.seed(2014)


class likelihood_unittest(unittest.TestCase):

    def generate_simple_data(self):

        N = 100
        M = 3
        r = np.array([75, 75, 75])

        l = np.random.uniform(low=100000, high=1000000, size=N)
        phi = np.random.uniform(low=0.2, high=0.4, size=N)
        p = np.vstack([phi, phi, np.ones(phi.shape)]).T

        cn = remixt.simulations.simple.generate_cn(N, M, 2.0, 0.5, 0.5, 2)
        h = np.random.uniform(low=0.5, high=2.0, size=M)

        # Add a 0 copy segment
        cn[0,1:,:] = 0

        mu = remixt.likelihood.expected_read_count(l, cn, h, phi)

        nb_p = mu / (r + mu)

        x = np.array([np.random.negative_binomial(r, 1.-a) for a in nb_p])
        x = x.reshape(nb_p.shape)

        return cn, h, l, phi, r, x


    def generate_count_data(self):

        N = 100
        r = 75.

        mu = np.random.uniform(low=100000, high=1000000, size=N)

        nb_p = mu / (r + mu)
        x = np.random.negative_binomial(r, 1.-nb_p)

        return mu, x


    def generate_allele_data(self):

        N = 100

        p = np.random.uniform(low=0.01, high=0.99, size=N)
        n = np.random.randint(low=10000, high=50000, size=N)

        k = np.random.binomial(n, p)

        return p, n, k


    def test_expected_read_count_opt(self):

        cn, h, l, phi, r, x = self.generate_simple_data()

        emission = likelihood_unopt.ReadCountLikelihood()
        emission.h = h
        emission.phi = phi

        unopt = emission.expected_read_count_unopt(l, cn)
        opt = emission.expected_read_count(l, cn)

        error = np.sum(unopt - opt)

        self.assertAlmostEqual(error, 0.0, places=3)


    def test_log_likelihood_cn_partial_phi_opt(self):

        cn, h, l, phi, r, x = self.generate_simple_data()

        emission = likelihood_unopt.NegBinLikelihood()
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

        cn, h, l, phi, r, x = self.generate_simple_data()

        emission = likelihood_unopt.NegBinLikelihood()
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

        cn, h, l, phi, r, x = self.generate_simple_data()

        emission = likelihood_unopt.PoissonLikelihood()
        emission.h = h
        emission.phi = phi

        unopt = emission._log_likelihood_partial_h_unopt(x, l, cn)
        opt = emission._log_likelihood_partial_h(x, l, cn)

        error = np.sum(unopt - opt)

        self.assertAlmostEqual(error, 0.0, places=3)


    def test_log_likelihood_cn_partial_phi(self):

        cn, h, l, phi, r, x = self.generate_simple_data()

        emission = likelihood.NegBinLikelihood()
        emission.h = h
        emission.phi = phi
        emission.r = r

        def evaluate_log_likelihood(phi, x, l, cn):
            emission.phi = phi
            return np.sum(emission.log_likelihood(x, l, cn))

        def evaluate_log_likelihood_partial_phi(phi, x, l, cn):
            emission.phi = phi
            return emission._log_likelihood_partial_phi(x, l, cn)[:,0]

        remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
            evaluate_log_likelihood_partial_phi, phi,
            x, l, cn)


    def test_log_likelihood_cn_negbin_partial_h(self):

        cn, h, l, phi, r, x = self.generate_simple_data()

        def evaluate_log_likelihood(h, x, l, cn, phi):
            emission = likelihood.NegBinLikelihood()
            emission.h = h
            emission.phi = phi
            emission.r = r
            return emission.log_likelihood(x, l, cn)

        def evaluate_log_likelihood_partial_h(h, x, l, cn, phi):
            emission = likelihood.NegBinLikelihood()
            emission.h = h
            emission.phi = phi
            emission.r = r
            return emission._log_likelihood_partial_h(x, l, cn)

        remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
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

        remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
            evaluate_log_likelihood_partial_r, r)


    def test_log_likelihood_cn_poisson_partial_h(self):

        cn, h, l, phi, r, x = self.generate_simple_data()

        def evaluate_log_likelihood(h, x, l, cn, phi):
            emission = likelihood.PoissonLikelihood()
            emission.h = h
            emission.phi = phi
            return emission.log_likelihood(x, l, cn)

        def evaluate_log_likelihood_partial_h(h, x, l, cn, phi):
            emission = likelihood.PoissonLikelihood()
            emission.h = h
            emission.phi = phi
            return emission._log_likelihood_partial_h(x, l, cn)

        remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
            evaluate_log_likelihood_partial_h, h,
            x, l, cn, phi)


    def test_log_likelihood_cn_betabinnegbin_partial_h(self):

        cn, h, l, phi, r, x = self.generate_simple_data()

        emission = likelihood.NegBinBetaBinLikelihood()

        def evaluate_log_likelihood(h, x, l, cn, phi):
            emission.h = h
            return emission.log_likelihood(x, l, cn)

        def evaluate_log_likelihood_partial_h(h, x, l, cn, phi):
            emission.h = h
            return emission._log_likelihood_partial_h(x, l, cn)

        remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
            evaluate_log_likelihood_partial_h, h,
            x, l, cn, phi)


    def test_learn_negbin_r_partial(self):

        N = 1000

        l = np.random.uniform(low=100000, high=1000000, size=N)
        x = np.random.uniform(low=0.02, high=0.1, size=N) * l

        negbin = remixt.likelihood.NegBinDistribution()

        g0 = remixt.paramlearn._sum_adjacent(x / l) / 2.
        r0 = 100.
        param0 = np.concatenate([g0, [r0]])

        remixt.tests.utils.assert_grad_correct(remixt.paramlearn.nll_negbin,
            remixt.paramlearn.nll_negbin_partial_param, param0,
            negbin, x, l)


    def test_log_likelihood_cn_betabin_partial_p(self):

        p, n, k = self.generate_allele_data()

        dist = likelihood.BetaBinDistribution()

        def evaluate_log_likelihood(p):
            return dist.log_likelihood(k, n, p).sum()

        def evaluate_log_likelihood_partial_p(p):
            return dist.log_likelihood_partial_p(k, n, p)

        remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
            evaluate_log_likelihood_partial_p, p)


    def test_log_likelihood_cn_betabin_partial_M(self):

        p, n, k = self.generate_allele_data()

        dist = likelihood.BetaBinDistribution()

        def evaluate_log_likelihood(M):
            dist.M = M
            return dist.log_likelihood(k, n, p)

        def evaluate_log_likelihood_partial_M(M):
            dist.M = M
            return dist.log_likelihood_partial_M(k, n, p)[:,None]

        M = np.array([50.])

        remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
            evaluate_log_likelihood_partial_M, M)


    def test_log_likelihood_cn_betabin_uniform_partial_p(self):

        p, n, k = self.generate_allele_data()

        for wrapped_dist in (likelihood.BetaBinDistribution, likelihood.BetaBinDistribution):

            dist = likelihood.BetaBinUniformDistribution(dist_type=wrapped_dist)

            def evaluate_log_likelihood(p):
                return dist.log_likelihood(k, n, p).sum()

            def evaluate_log_likelihood_partial_p(p):
                return dist.log_likelihood_partial_p(k, n, p)

            remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
                evaluate_log_likelihood_partial_p, p)


    def test_log_likelihood_cn_betabin_uniform_partial_M(self):

        p, n, k = self.generate_allele_data()

        for wrapped_dist in (likelihood.BetaBinDistribution, likelihood.BetaBinDistribution):

            dist = likelihood.BetaBinUniformDistribution(dist_type=wrapped_dist)

            def evaluate_log_likelihood(M):
                dist.M = M
                return dist.log_likelihood(k, n, p)

            def evaluate_log_likelihood_partial_M(M):
                dist.M = M
                return dist.log_likelihood_partial_M(k, n, p)[:,None]

            M = np.array([50.])

            remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
                evaluate_log_likelihood_partial_M, M)


    def test_log_likelihood_cn_betabin_uniform_partial_z(self):

        p, n, k = self.generate_allele_data()

        for wrapped_dist in (likelihood.BetaBinDistribution, likelihood.BetaBinDistribution):

            dist = likelihood.BetaBinUniformDistribution(dist_type=wrapped_dist)

            def evaluate_log_likelihood(z):
                dist.z = z
                return dist.log_likelihood(k, n, p)

            def evaluate_log_likelihood_partial_z(z):
                dist.z = z
                return dist.log_likelihood_partial_z(k, n, p)[:,None]

            z = np.array([0.01])

            remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
                evaluate_log_likelihood_partial_z, z)


    def test_log_likelihood_cn_betabin_reflected_partial_p(self):

        p, n, k = self.generate_allele_data()

        dist = likelihood.BetaBinReflectedDistribution()

        def evaluate_log_likelihood(p):
            return dist.log_likelihood(k, n, p).sum()

        def evaluate_log_likelihood_partial_p(p):
            return dist.log_likelihood_partial_p(k, n, p)

        remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
            evaluate_log_likelihood_partial_p, p)


    def test_log_likelihood_cn_betabin_reflected_partial_M(self):

        p, n, k = self.generate_allele_data()

        dist = likelihood.BetaBinReflectedDistribution()

        def evaluate_log_likelihood(M):
            dist.M = M
            return dist.log_likelihood(k, n, p)

        def evaluate_log_likelihood_partial_M(M):
            dist.M = M
            return dist.log_likelihood_partial_M(k, n, p)[:,None]

        M = np.array([50.])

        remixt.tests.utils.assert_grad_correct(evaluate_log_likelihood,
            evaluate_log_likelihood_partial_M, M)


    def test_learn_betabin_r_partial(self):

        p, n, k = self.generate_allele_data()

        betabin = remixt.likelihood.BetaBinDistribution()

        p0 = remixt.paramlearn._sum_adjacent(k.astype(float) / n.astype(float)) / 2.
        M0 = 100.
        param0 = np.concatenate([p0, [M0]])

        remixt.tests.utils.assert_grad_correct(remixt.paramlearn.nll_betabin,
            remixt.paramlearn.nll_betabin_partial_param, param0,
            betabin, k, n)

    def test_log_likelihood_cn_betabinnegbin_cornercases(self):

        cn, h, l, phi, r, x = self.generate_simple_data()

        emission = likelihood.NegBinBetaBinLikelihood()
        emission.h = np.array([1e-16, 10., 1e-16])

        cn[:,:,:] = np.array([[1., 1.], [0., 1.], [1., 0.]])

        ll = emission.log_likelihood(x, l, cn)
        ll_partial_h = emission.log_likelihood_partial_param(x, l, cn, 'h')

        self.assertTrue(not np.any(np.isnan(ll)))
        self.assertTrue(not np.any(np.isnan(ll_partial_h)))


if __name__ == '__main__':
    unittest.main()
