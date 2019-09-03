import sys
import os
import unittest
import copy
import itertools
import pickle
import numpy as np
import pandas as pd
import random
import joblib
import scipy
import scipy.optimize

import remixt.simulations.simple as sim_simple
import remixt.simulations.experiment
import remixt.simulations.pipeline
import remixt.cn_model as cn_model
import remixt.likelihood as likelihood
import remixt.seqdataio
import remixt.tests.utils
import remixt.cn_plot
import remixt.analysis.pipeline
import remixt.visualize

np.random.seed(2014)


def generate_simple_data(N=100, M=3):

    r = np.array([75, 75, 75])

    l = np.random.uniform(low=100000, high=1000000, size=N)
    phi = np.random.uniform(low=0.2, high=0.4, size=N)

    cn = sim_simple.generate_cn(N, M, 2.0, 0.5, 0.5, 1)
    h = np.random.uniform(low=0.5, high=2.0, size=M)

    likelihood_model = likelihood.ReadCountLikelihood(None, l)
    likelihood_model.h = h
    likelihood_model.phi = phi

    mu = likelihood_model.expected_read_count(l, cn)

    nb_p = mu / (r + mu)

    x = np.array([np.random.negative_binomial(r, 1.-a) for a in nb_p])
    x = x.reshape(nb_p.shape)
    x[:,0:2].sort(axis=1)
    x[:,0:2] = x[:,2:0:-1]

    return cn, h, l, phi, r, x


def load_test_experiment():

    script_directory = os.path.realpath(os.path.dirname(__file__))
    experiment_filename = os.path.join(script_directory, 'test_experiment.pickle')

    with open(experiment_filename) as experiment_file:
        experiment = pickle.load(experiment_file)

    return experiment
    
    
def uniform_cn_prior():

    cn_max = 4

    num_haploid_states = cn_max + 1
    num_cn_states = num_haploid_states * (num_haploid_states + 1) / 2
    num_total_states = num_cn_states + 1

    cn_prior = np.ones((cn_max + 2, cn_max + 2)) / float(num_total_states)

    return cn_prior


def perfect_cn_prior(cn):

    cn_max = 4

    cn_prior = np.zeros((cn_max + 2, cn_max + 2))

    for n in range(cn.shape[0]):
        for m in range(1, cn.shape[1]):
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

    cn_prior[:] += 1.0

    cn_prior /= cn.shape[0] * cn.shape[1]

    return cn_prior


def generate_tiny_data():

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

    mu = likelihood.expected_read_count(l, cn, h, phi)

    x = np.array([np.random.poisson(a) for a in mu])
    x = x.reshape(mu.shape)

    return x, cn, h, l, phi


class remixt_unittest(unittest.TestCase):

    def test_evaluate_q_derivative(self):

        cn, h, l, phi, r, x = generate_simple_data()

        emission = likelihood.NegBinBetaBinLikelihood(x, l)
        emission.h = h

        N = l.shape[0]
        M = h.shape[0]

        prior = cn_model.CopyNumberPrior(perfect_cn_prior(cn))
        prior.set_lengths(l)
        
        model = cn_model.HiddenMarkovModel(N, M, emission, prior, [(0, N)])

        log_posterior, cns, resps = model.posterior_marginals()

        estimator = em.ExpectationMaximizationEstimator()

        params = [emission.h_param, emission.r_param, emission.M_param, emission.z_param, emission.hdel_mu_param, emission.loh_p_param]
        value = np.concatenate([p.value for p in params])
        idxs = np.array([p.length for p in params]).cumsum() - params[0].length

        remixt.tests.utils.assert_grad_correct(
            estimator.evaluate_q, estimator.evaluate_q_derivative,
            value, model, cns, resps, params, idxs)


    def test_build_cn(self):

        prior = cn_model.CopyNumberPrior(1, 3, uniform_cn_prior())
        model = cn_model.HiddenMarkovModel(1, 3, None, prior)

        cn_max = 6
        cn_dev_max = 1

        model.cn_max = cn_max
        model.cn_dev_max = cn_dev_max

        # Build cn list using the method from CopyNumberModel
        built_cns = set()
        for cn in model.build_cn_states():
            cn_tuple = list()
            for m in range(3):
                for ell in range(2):
                    cn_tuple.append(cn[0,m,ell])
            cn_tuple = tuple(cn_tuple)
            self.assertNotIn(cn_tuple, built_cns)
            built_cns.add(cn_tuple)

        # Build the naive way
        expected_cns = set()
        for b1 in range(cn_max+1):
            for b2 in range(cn_max+1):
                for d1 in range(-cn_dev_max, cn_dev_max+1):
                    for d2 in range(-cn_dev_max, cn_dev_max+1):
                        if b1 + d1 < 0 or b2 + d2 < 0 or b1 + d1 > cn_max or b2 + d2 > cn_max:
                            continue
                        if (b1 != b2 or b1+d1 != b2+d2) and (b1 <= b2 and b1+d1 <= b2+d2):
                            continue
                        cn_tuple = (1, 1, b1, b2, b1+d1, b2+d2)
                        self.assertIn(cn_tuple, built_cns)
                        expected_cns.add(cn_tuple)

        self.assertTrue(expected_cns == built_cns)


    def test_simple_genome_graph(self):

        x, cn, h, l, phi = generate_tiny_data()

        emission = likelihood.NegBinBetaBinLikelihood(x, l)
        emission.h = h
        emission.phi = phi

        N = l.shape[0]
        M = h.shape[0]

        prior = cn_model.CopyNumberPrior(l)
        
        adjacencies = [(0, 1), (1, 2)]
        breakpoints = [frozenset([(0, 1), (2, 0)])]

        for i in range(1000):
            print (i)
            random.seed(i)
            np.random.seed(i)

            # Modify copy number from known true
            cn_init = cn.copy()
            for n in range(N):
                m = np.random.randint(low=1, high=M)
                for allele in range(2):
                    cn_init[n, m, allele] += np.random.randint(low=-1, high=1)

            # print np.random.randint(low=-1, high=1, size=cn_init[:, 1:, :].shape)
            # cn_init[:, 1:, :] += np.random.randint(low=-1, high=1, size=cn_init[:, 1:, :].shape)

            graph = genome_graph.GenomeGraphModel(N, M, emission, prior, adjacencies, breakpoints)
            graph.init_copy_number(cn_init, init_breakpoints=True)

            graph.optimize()

            if not np.all(graph.breakpoint_copy_number[['cn_1', 'cn_2']].values == np.array([1, 0])):
                print (cn)
                print (cn_init)
                print (graph.segment_cn)
                print (graph.breakpoint_copy_number)
                import IPython; IPython.embed(); raise

            self.assertTrue(np.all(graph.breakpoint_copy_number[['cn_1', 'cn_2']].values == np.array([1, 0])))


    def test_learn_h_graph(self):

        experiment = load_test_experiment()

        emission = likelihood.NegBinBetaBinLikelihood()
        emission.h = experiment.h
        emission.learn_parameters(experiment.x, experiment.l)

        N = experiment.l.shape[0]
        M = experiment.h.shape[0]

        prior = cn_model.CopyNumberPrior(N, M, perfect_cn_prior(experiment.cn))
        prior.set_lengths(experiment.l)

        model = cn_model.HiddenMarkovModel(N, M, emission, prior)
        
        _, cn_init = model.optimal_state()

        graph = genome_graph.GenomeGraph(emission, prior, experiment.adjacencies, experiment.breakpoints)
        graph.init_copy_number(cn_init)

        h_init = experiment.h + experiment.h * 0.05 * np.random.randn(*experiment.h.shape)

        estimator = em.HardAssignmentEstimator(num_em_iter=1)
        estimator.learn_param(graph, 'h', h_init)


    def test_learn_h(self):

        experiment = load_test_experiment()

        h_init = experiment.h * (1. + 0.05 * np.random.randn(*experiment.h.shape))

        emission = likelihood.NegBinBetaBinLikelihood(experiment.x, experiment.l)
        emission.h = h_init

        print (experiment.h, h_init)

        N = experiment.l.shape[0]
        M = experiment.h.shape[0]

        prior = cn_model.CopyNumberPrior(experiment.l)

        model = cn_model.HiddenMarkovModel(N, M, emission, prior, experiment.chains, normal_contamination=False)
        
        estimator = em.ExpectationMaximizationEstimator()
        estimator.learn_param(model, emission.h_param)


    def test_learn_h_variational(self):

        experiment = load_test_experiment()
        
        h_init = experiment.h * (1. + 0.05 * np.random.randn(*experiment.h.shape))

        init_params = {}
        init_params['h_normal'] = h_init[0]
        init_params['h_tumour'] = h_init[1:].sum()
        init_params['mix_frac'] = h_init[1] / h_init[1:].sum()
        init_params['divergence_weight'] = 1e-7
        init_params['mode_idx'] = 0
        
        config = {}
        
        # import pstats, cProfile
        # cProfile.runctx("fit_results = remixt.analysis.pipeline.fit(experiment, init_params, config)", globals(), locals(), "Profile.prof")
        # s = pstats.Stats("Profile.prof")
        # s.strip_dirs().sort_stats("cumtime").print_stats()
        # raise
        fit_results = remixt.analysis.pipeline.fit(experiment, init_params, config)
        
        # with open('test_model.gz', 'w') as f:
        #     pickle.dump(fit_results, f)
        
        h = fit_results['h']
        cn = fit_results['cn']
        brk_cn = fit_results['brk_cn']

        cn_table = remixt.analysis.experiment.create_cn_table(experiment, cn, h)
        brk_cn_table = remixt.analysis.experiment.create_brk_cn_table(experiment, brk_cn)

        evaluation = remixt.simulations.pipeline.evaluate_results(experiment.genome_mixture, cn_table, brk_cn_table, h / h.sum())
        with pd.HDFStore('test.h5', 'w') as store:
            for key, data in evaluation.items():
                store['/' + key] = data

        print ('creating visualization')
        remixt.visualize.create_genome_visualization(cn_table, brk_cn_table, 'test.html')

        print ('plotting result')
        fig = remixt.cn_plot.experiment_plot(experiment, cn, h, maxcopies=8)
        fig.savefig('test_result.pdf')


    def test_learn_r(self):

        experiment = load_test_experiment()

        emission = likelihood.NegBinLikelihood()
        emission.h = experiment.h
        emission.phi = experiment.phi
        emission.r = experiment.negbin_r

        N = experiment.l.shape[0]
        M = experiment.h.shape[0]

        prior = cn_model.CopyNumberPrior(N, M, perfect_cn_prior(experiment.cn))
        prior.set_lengths(experiment.l)

        model = cn_model.HiddenMarkovModel(N, M, emission, prior)
        
        r_init = experiment.negbin_r + experiment.negbin_r * 0.10 * np.random.randn(*experiment.negbin_r.shape)

        estimator = em.ExpectationMaximizationEstimator(num_em_iter=1)
        estimator.learn_param(model, 'r', r_init)


    def test_learn_phi(self):

        experiment = load_test_experiment()

        emission = likelihood.NegBinLikelihood()
        emission.h = experiment.h
        emission.phi = experiment.phi
        emission.r = experiment.negbin_r

        N = experiment.l.shape[0]
        M = experiment.h.shape[0]

        prior = cn_model.CopyNumberPrior(N, M, perfect_cn_prior(experiment.cn))
        prior.set_lengths(experiment.l)

        model = cn_model.HiddenMarkovModel(N, M, emission, prior)

        phi_init = experiment.phi + experiment.phi * 0.02 * np.random.randn(*experiment.phi.shape)

        estimator = em.ExpectationMaximizationEstimator(num_em_iter=1)
        estimator.learn_param(model, 'phi', phi_init)


    def test_recreate(self):

        rparams = remixt.simulations.experiment.RearrangedGenome.default_params.copy()

        genome = remixt.simulations.experiment.RearrangedGenome(100)

        genome.create(rparams)

        for _ in range(20):
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

        for _ in range(10):
            genome.rearrange(rparams)

        cn_1 = copy.deepcopy(genome.segment_copy_number)
        brks_1 = copy.deepcopy(genome.breakpoints)

        for _ in range(10):
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

        for _ in range(10):
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


