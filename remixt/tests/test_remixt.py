import sys
import os
import unittest
import copy
import itertools
import pickle
import numpy as np
import pandas as pd
import scipy
import scipy.optimize

import remixt.simulations.simple as sim_simple
import remixt.simulations.experiment
import remixt.cn_model as cn_model
import remixt.em as em
import remixt.likelihood as likelihood
import remixt.genome_graph as genome_graph
import remixt.seqdataio

np.random.seed(2014)


class remixt_unittest(unittest.TestCase):

    def generate_simple_data(self, total_cn, N=100, M=3):

        r = np.array([75, 75, 75])

        l = np.random.uniform(low=100000, high=1000000, size=N)
        phi = np.random.uniform(low=0.2, high=0.4, size=N)
        p = np.vstack([phi, phi, np.ones(phi.shape)]).T

        cn = sim_simple.generate_cn(N, M, 2.0, 0.5, 0.5, 1)
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


    def load_test_experiment(self):

        script_directory = os.path.realpath(os.path.dirname(__file__))
        experiment_filename = os.path.join(script_directory, 'test_experiment.pickle')

        with open(experiment_filename) as experiment_file:
            experiment = pickle.load(experiment_file)

        return experiment


    def uniform_cn_prior(self):

        cn_max = 4

        num_haploid_states = cn_max + 1
        num_cn_states = num_haploid_states * (num_haploid_states + 1) / 2
        num_total_states = num_cn_states + 1

        cn_prior = np.ones((cn_max + 2, cn_max + 2)) / float(num_total_states)

        return cn_prior


    def perfect_cn_prior(self, cn):

        cn_max = 4

        cn_prior = np.zeros((cn_max + 2, cn_max + 2))

        for n in xrange(cn.shape[0]):
            for m in xrange(1, cn.shape[1]):
                cn_1 = int(cn[n,m,0])
                cn_2 = int(cn[n,m,1])
                if cn_1 > cn_max or cn_2 > cn_max:
                    cn_prior[cn_max+1, :] += 1.0
                    cn_prior[:, cn_max+1] += 1.0
                    cn_prior[cn_max+1, cn_max+1] -= 1.0
                elif cn_1 != cn_2:
                    cn_prior[cn_1, cn_2] += 1.0
                    cn_prior[cn_2, cn_1] += 1.0
                else:
                    cn_prior[cn_1, cn_2] += 1.0

        cn_prior /= cn.shape[0] * cn.shape[1]

        return cn_prior


    def generate_tiny_data(self):

        cn = np.array([[[1, 1],
                        [2, 2],
                        [2, 2]],
                       [[1, 1],
                        [2, 1],
                        [2, 2]],
                       [[1, 1],
                        [2, 2],
                        [2, 2]]])

        h = np.array([0.2, 0.3, 0.1])

        l = np.array([10000, 20000, 10000])

        phi = np.array([0.2, 0.2, 0.2])

        p = np.vstack([phi, phi, np.ones(phi.shape)]).T

        likelihood_model = likelihood.ReadCountLikelihood()
        likelihood_model.h = h
        likelihood_model.phi = phi

        mu = likelihood_model.expected_read_count(l, cn)

        x = np.array([np.random.poisson(a) for a in mu])
        x = x.reshape(mu.shape)

        return x, cn, h, l, phi

        
    def test_evaluate_q_derivative(self):

        for total_cn in (True, False):
            cn, h, l, phi, r, x = self.generate_simple_data(total_cn)

            emission = likelihood.NegBinLikelihood(total_cn=total_cn)
            emission.h = h
            emission.phi = phi
            emission.r = r

            N = l.shape[0]
            M = h.shape[0]

            prior = cn_model.CopyNumberPrior(N, M, self.perfect_cn_prior(cn))
            prior.set_lengths(l)
            
            model = cn_model.HiddenMarkovModel(N, M, emission, prior)
            model.set_observed_data(x, l)

            log_posterior, cns, resps = model.posterior_marginals()

            estimator = em.ExpectationMaximizationEstimator()

            grad_error = scipy.optimize.check_grad(estimator.evaluate_q, estimator.evaluate_q_derivative,
                h, model, 'h', cns, resps)

            self.assertAlmostEqual(grad_error, 0.0, places=-1)


    def test_build_cn(self):

        prior = cn_model.CopyNumberPrior(1, 3, self.uniform_cn_prior())
        model = cn_model.HiddenMarkovModel(1, 3, None, prior)

        model.cn_max = 6
        model.cn_dev_max = 2

        # Build cn list using the method from CopyNumberModel
        built_cns = set()
        for cn in model.build_cn_states():
            cn_tuple = list()
            for m in xrange(3):
                for ell in xrange(2):
                    cn_tuple.append(cn[0,m,ell])
            cn_tuple = tuple(cn_tuple)
            self.assertNotIn(cn_tuple, built_cns)
            built_cns.add(cn_tuple)

        # Build the naive way
        num_cns = 0
        for b1 in xrange(6+1):
            for b2 in xrange(6+1):
                for d1 in xrange(-2, 2+1):
                    for d2 in xrange(-2, 2+1):
                        if b1 + d1 < 0 or b2 + d2 < 0 or b1 + d1 > 6 or b2 + d2 > 6:
                            continue
                        cn_tuple = (1, 1, b1, b2, b1+d1, b2+d2)
                        self.assertIn(cn_tuple, built_cns)
                        num_cns += 1

        self.assertTrue(num_cns == len(built_cns))


    def test_simple_genome_graph(self):

        x, cn, h, l, phi = self.generate_tiny_data()

        cn_true = cn.copy()

        emission = likelihood.PoissonLikelihood(total_cn=False)
        emission.h = h
        emission.phi = phi

        N = l.shape[0]
        M = h.shape[0]

        prior = cn_model.CopyNumberPrior(N, M, self.uniform_cn_prior())
        prior.set_lengths(l)
        
        adjacencies = [(0, 1), (1, 2)]
        breakpoints = [frozenset([(0, 1), (2, 0)])]

        # Modify copy number from known true
        cn[1,1,1] = 2

        graph = genome_graph.GenomeGraph(emission, prior, adjacencies, breakpoints)
        graph.set_observed_data(x, l)
        graph.init_copy_number(cn)

        graph.optimize()

        self.assertTrue(np.all(graph.cn == cn_true))

        bond_cn = graph.bond_cn.set_index(['n_1', 'ell_1', 'side_1', 'n_2', 'ell_2', 'side_2'])
        bond_cn = bond_cn[['cn_1', 'cn_2']]

        # Check reference edges
        self.assertTrue(np.all(bond_cn.loc[0,0,1,1,0,0].values == np.array([2, 2])))
        self.assertTrue(np.all(bond_cn.loc[0,1,1,1,1,0].values == np.array([1, 2])))
        self.assertTrue(np.all(bond_cn.loc[1,0,1,2,0,0].values == np.array([2, 2])))
        self.assertTrue(np.all(bond_cn.loc[1,1,1,2,1,0].values == np.array([1, 2])))

        # Check variant edge
        self.assertTrue(np.all(bond_cn.loc[0,1,1,2,1,0].values == np.array([1, 0])))


    def test_learn_h(self):

        experiment = self.load_test_experiment()

        print 'h', experiment.h

        emission = likelihood.NegBinLikelihood(total_cn=False)
        emission.h = experiment.h
        emission.phi = experiment.phi
        emission.r = experiment.negbin_r[0:2]

        N = experiment.l.shape[0]
        M = experiment.h.shape[0]

        prior = cn_model.CopyNumberPrior(N, M, self.perfect_cn_prior(experiment.cn))
        prior.set_lengths(experiment.l)

        model = cn_model.HiddenMarkovModel(N, M, emission, prior)
        model.set_observed_data(experiment.x, experiment.l)
        
        _, cn_init = model.optimal_state()

        graph = genome_graph.GenomeGraph(emission, prior, experiment.adjacencies, experiment.breakpoints)
        graph.set_observed_data(experiment.x, experiment.l)
        graph.init_copy_number(cn_init)

        h_init = experiment.h + experiment.h * 0.05 * np.random.randn(*experiment.h.shape)

        print 'h_init', h_init

        estimator = em.HardAssignmentEstimator(num_em_iter=1)
        estimator.learn_param(graph, 'h', h_init)


    def test_learn_r(self):

        experiment = self.load_test_experiment()

        print 'r', experiment.negbin_r

        emission = likelihood.NegBinLikelihood(total_cn=True)
        emission.h = experiment.h
        emission.phi = experiment.phi
        emission.r = experiment.negbin_r

        N = experiment.l.shape[0]
        M = experiment.h.shape[0]

        prior = cn_model.CopyNumberPrior(N, M, self.perfect_cn_prior(experiment.cn))
        prior.set_lengths(experiment.l)

        model = cn_model.HiddenMarkovModel(N, M, emission, prior)
        model.set_observed_data(experiment.x, experiment.l)
        
        r_init = experiment.negbin_r + experiment.negbin_r * 0.10 * np.random.randn(*experiment.negbin_r.shape)

        print 'r_init', r_init

        estimator = em.ExpectationMaximizationEstimator(num_em_iter=1)
        estimator.learn_param(model, 'r', r_init)


    def test_learn_phi(self):

        experiment = self.load_test_experiment()

        print 'phi', experiment.phi

        emission = likelihood.NegBinLikelihood(total_cn=True)
        emission.h = experiment.h
        emission.phi = experiment.phi
        emission.r = experiment.negbin_r

        N = experiment.l.shape[0]
        M = experiment.h.shape[0]

        prior = cn_model.CopyNumberPrior(N, M, self.perfect_cn_prior(experiment.cn))
        prior.set_lengths(experiment.l)

        model = cn_model.HiddenMarkovModel(N, M, emission, prior)
        model.set_observed_data(experiment.x, experiment.l)

        phi_init = experiment.phi + experiment.phi * 0.02 * np.random.randn(*experiment.phi.shape)

        print 'phi_init', phi_init

        estimator = em.ExpectationMaximizationEstimator(num_em_iter=1)
        estimator.learn_param(model, 'phi', phi_init)


    def test_recreate(self):

        rparams = remixt.simulations.experiment.RearrangedGenome.default_params.copy()

        genome = remixt.simulations.experiment.RearrangedGenome(100)

        genome.create(rparams)

        for _ in xrange(20):
            genome.rearrange(rparams)

        cn_1 = copy.deepcopy(genome.segment_copy_number)
        brks_1 = copy.deepcopy(genome.breakpoints)

        genome.recreate()

        cn_2 = copy.deepcopy(genome.segment_copy_number)
        brks_2 = copy.deepcopy(genome.breakpoints)

        self.assertTrue(np.all(cn_1 == cn_2))
        self.assertTrue(brks_1 == brks_2)


    def test_rewind(self):

        rparams = remixt.simulations.experiment.RearrangedGenome.default_params.copy()

        genome = remixt.simulations.experiment.RearrangedGenome(100)

        genome.create(rparams)

        for _ in xrange(10):
            genome.rearrange(rparams)

        cn_1 = copy.deepcopy(genome.segment_copy_number)
        brks_1 = copy.deepcopy(genome.breakpoints)

        for _ in xrange(10):
            genome.rearrange(rparams)

        genome.rewind(10)

        cn_2 = copy.deepcopy(genome.segment_copy_number)
        brks_2 = copy.deepcopy(genome.breakpoints)

        self.assertTrue(np.all(cn_1 == cn_2))
        self.assertTrue(brks_1 == brks_2)


    def test_create_rearranged_sequence(self):

        rparams = remixt.simulations.experiment.RearrangedGenome.default_params.copy()

        rparams['chromosome_lengths'] = {'1':20, '2':10}

        genome = remixt.simulations.experiment.RearrangedGenome(4)

        np.random.seed(2014)

        genome.create(rparams)

        for _ in xrange(10):
            genome.rearrange(rparams)

        germline_genome = {
            ('1', 0): 'AAAAACCCCCGGGGGTTTTT',
            ('1', 1): 'AATAACCCCCGGTGGTTTTT',
            ('2', 0): 'ACACACACAC',
            ('2', 1): 'ACACAGACAC',
        }

        true_sequences = ['AAAAACCACCGGGGGTTATTGTTATTACACAGACACACACAGACACAAAAACCACCGGGGACAGACACAAAAACCACCGGGGACAGACAC']

        test_sequences = genome.create_chromosome_sequences(germline_genome)

        self.assertTrue(test_sequences == true_sequences)


if __name__ == '__main__':
    unittest.main()


