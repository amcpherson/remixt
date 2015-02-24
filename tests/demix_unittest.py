import sys
import os
import unittest
import copy
import itertools
import numpy as np
import scipy
import scipy.optimize

demix_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(demix_directory)

import demix.simulations.simple as sim_simple
import demix.simulations.experiment as sim_experiment
import demix.cn_model as cn_model
import demix.genome_graph as genome_graph

np.random.seed(2014)


if __name__ == '__main__':

    class demix_unittest(unittest.TestCase):

        def generate_simple_data(self, total_cn):

            N = 100
            M = 3
            r = np.array([75, 75, 75])

            l = np.random.uniform(low=100000, high=1000000, size=N)
            p = np.random.uniform(low=0.2, high=0.4, size=N)
            p = np.vstack([p, p, np.ones(p.shape)]).T

            cn = sim_simple.generate_cn(N, M, 2.0, 0.5, 0.5, 2)
            h = np.random.uniform(low=0.5, high=2.0, size=M)

            model = cn_model.CopyNumberModel(M, set(), set())

            mu = model.expected_read_count(l, cn, h, p)

            nb_p = mu / (r + mu)

            x = np.array([np.random.negative_binomial(r, 1.-a) for a in nb_p])
            x = x.reshape(nb_p.shape)

            if not total_cn:
                r = r[0:2]

            return cn, h, l, p, r, x


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

            p = np.array([[0.2, 0.2, 1.0],
                          [0.2, 0.2, 1.0],
                          [0.2, 0.2, 1.0]])

            mu = cn_model.CopyNumberModel.expected_read_count(l, cn, h, p)

            x = np.array([np.random.poisson(a) for a in mu])
            x = x.reshape(mu.shape)

            return x, cn, h, l, p


        def test_expected_read_count_opt(self):

            for total_cn in (True, False):
                cn, h, l, p, r, x = self.generate_simple_data(total_cn)

                model = cn_model.CopyNumberModel(cn.shape[1], set(), set())
                model.total_cn = total_cn

                unopt = model.expected_read_count_unopt(l, cn, h, p)

                opt = model.expected_read_count(l, cn, h, p)

                error = np.sum(unopt - opt)

                self.assertAlmostEqual(error, 0.0, places=3)


        def test_log_likelihood_cn_negbin_opt(self):

            for total_cn in (True, False):
                cn, h, l, p, r, x = self.generate_simple_data(total_cn)

                model = cn_model.CopyNumberModel(cn.shape[1], set(), set())
                model.total_cn = total_cn

                mu = model.expected_read_count_unopt(l, cn, h, p)

                unopt = model.log_likelihood_cn_negbin_unopt(x, mu, r)
                opt = model.log_likelihood_cn_negbin(x, mu, r)

                error = np.sum(unopt - opt)

                self.assertAlmostEqual(error, 0.0, places=3)


        def test_log_likelihood_cn_poisson_opt(self):

            for total_cn in (True, False):
                cn, h, l, p, r, x = self.generate_simple_data(total_cn)

                model = cn_model.CopyNumberModel(cn.shape[1], set(), set())
                model.total_cn = total_cn

                mu = model.expected_read_count_unopt(l, cn, h, p)

                unopt = model.log_likelihood_cn_poisson_unopt(x, mu)
                opt = model.log_likelihood_cn_poisson(x, mu)

                error = np.sum(unopt - opt)

                self.assertAlmostEqual(error, 0.0, places=3)


        def test_log_likelihood_cn_negbin_partial_h_opt(self):

            for total_cn in (True, False):
                cn, h, l, p, r, x = self.generate_simple_data(total_cn)

                model = cn_model.CopyNumberModel(cn.shape[1], set(), set())
                model.total_cn = total_cn

                unopt = model.log_likelihood_cn_negbin_partial_h_unopt(x, l, cn, h, p, r)
                opt = model.log_likelihood_cn_negbin_partial_h(x, l, cn, h, p, r)

                error = np.sum(unopt - opt)

                self.assertAlmostEqual(error, 0.0, places=3)


        def test_log_likelihood_cn_poisson_partial_h_opt(self):

            for total_cn in (True, False):
                cn, h, l, p, r, x = self.generate_simple_data(total_cn)

                model = cn_model.CopyNumberModel(cn.shape[1], set(), set())
                model.total_cn = total_cn

                unopt = model.log_likelihood_cn_poisson_partial_h_unopt(x, l, cn, h, p)
                opt = model.log_likelihood_cn_poisson_partial_h(x, l, cn, h, p)

                error = np.sum(unopt - opt)

                self.assertAlmostEqual(error, 0.0, places=3)


        def test_log_likelihood_cn_negbin_partial_h(self):

            for total_cn in (True, False):
                cn, h, l, p, r, x = self.generate_simple_data(total_cn)

                model = cn_model.CopyNumberModel(cn.shape[1], set(), set())
                model.total_cn = total_cn

                def evaluate_log_likelihood(h):
                    mu = model.expected_read_count_unopt(l, cn, h, p)
                    return np.sum(model.log_likelihood_cn_negbin(x, mu, r))

                def evaluate_log_likelihood_partial_h(h):
                    return np.sum(model.log_likelihood_cn_negbin_partial_h(x, l, cn, h, p, r), axis=0)

                approx = scipy.optimize.approx_fprime(h, evaluate_log_likelihood, 1e-8) 
                calculated = evaluate_log_likelihood_partial_h(h)
                error = np.sum(np.square(approx - calculated)) / np.sum(np.square((approx + calculated)/2.0))

                self.assertAlmostEqual(error, 0.0, places=3)


        def test_log_likelihood_cn_poisson_partial_h(self):

            for total_cn in (True, False):
                cn, h, l, p, r, x = self.generate_simple_data(total_cn)

                model = cn_model.CopyNumberModel(cn.shape[1], set(), set())
                model.total_cn = total_cn

                def evaluate_log_likelihood(h):
                    mu = model.expected_read_count_unopt(l, cn, h, p)
                    return np.sum(model.log_likelihood_cn_poisson(x, mu))

                def evaluate_log_likelihood_partial_h(h):
                    return np.sum(model.log_likelihood_cn_poisson_partial_h(x, l, cn, h, p), axis=0)

                approx = scipy.optimize.approx_fprime(h, evaluate_log_likelihood, 1e-8) 
                calculated = evaluate_log_likelihood_partial_h(h)
                error = np.sum(np.square(approx - calculated)) / np.sum(np.square((approx + calculated)/2.0))
                
                self.assertAlmostEqual(error, 0.0, places=3)


        def test_evaluate_q_derivative(self):

            for total_cn in (True, False):
                cn, h, l, p, r, x = self.generate_simple_data(total_cn)

                model = cn_model.CopyNumberModel(cn.shape[1], set(), set())
                model.total_cn = total_cn

                model.infer_offline_parameters(x, l)

                cns, resps, log_posterior = model.e_step_independent(x, l, h)

                grad_error = scipy.optimize.check_grad(model.evaluate_q, model.evaluate_q_derivative, h, x, l, cns, resps)
                
                self.assertAlmostEqual(grad_error, 0.0, places=-1)


        def test_build_cn(self):

            model = cn_model.CopyNumberModel(3, set(), set())
            model.cn_max = 6
            model.cn_dev_max = 2

            # Build cn list using the method from CopyNumberModel
            built_cns = set()
            for cn in model.build_hmm_cns(1):
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


        def test_wildcard_cn(self):

            x, cn, h, l, p = self.generate_tiny_data()

            model = cn_model.CopyNumberModel(3, set(), set())
            model.wildcard_cn_max = 2
            model.cn_dev_max = 1
            model.cn_max = 0

            model.infer_p(x)

            # Build cn list using the method from CopyNumberModel
            built_cns = [set() for _ in xrange(x.shape[0])]
            for cn in model.build_wildcard_cns(x, l, h):
                for n in xrange(x.shape[0]):
                    built_cns[n].add(tuple(cn[n].flatten().astype(int)))

            # Build the naive way
            for n in xrange(x.shape[0]):
                num_cns = 0
                h_t = ((x[n,0:2] / p[n,0:2]).T / l[n]).T
                dom_cn = (h_t - h[0]) / h[1:].sum()
                dom_cn = dom_cn.round().astype(int)
                for b1 in xrange(2+1):
                    b1 += dom_cn[0] - 1
                    for b2 in xrange(2+1):
                        b2 += dom_cn[1] - 1
                        for d1 in xrange(-1, 1+1):
                            for d2 in xrange(-1, 1+1):
                                cn_test = np.array([1, 1, b1, b2, b1+d1, b2+d2])
                                cn_test[cn_test < 0] += 100
                                self.assertIn(tuple(cn_test.flatten()), built_cns[n])
                                num_cns += 1

                self.assertTrue(num_cns == len(built_cns[n]))


        def test_simple_genome_graph(self):

            x, cn, h, l, p = self.generate_tiny_data()

            cn_true = cn.copy()

            model = cn_model.CopyNumberModel(3, set(), set())
            model.total_cn = False
            model.false_cn = False

            wt_adj = set()
            for seg in xrange(3):
                for allele in (0, 1):
                    wt_adj.add(frozenset([((seg, allele), 1), (((seg + 1) % 3, allele), 0)]))

            tmr_adj = set()
            for allele_1, allele_2 in itertools.product(xrange(2), repeat=2):
                tmr_adj.add(frozenset([((0, allele_1), 1), ((2, allele_2),0)]))

            # Modify copy number from known true
            cn[1,1,1] = 2

            model.infer_p(x)
            
            model.r = np.array([100., 100.])

            graph = genome_graph.GenomeGraph(model, x, l, cn, wt_adj, tmr_adj)

            graph.optimize(h)

            self.assertTrue(np.all(graph.cn == cn_true))

            bond_cn = graph.bond_cn.set_index(['n_1', 'ell_1', 'side_1', 'n_2', 'ell_2', 'side_2'])
            bond_cn = bond_cn[['cn_1', 'cn_2']]

            # Check reference edges
            self.assertTrue(np.all(bond_cn.loc[0,0,1,1,0,0].values == np.array([2, 2])))
            self.assertTrue(np.all(bond_cn.loc[0,1,1,1,1,0].values == np.array([1, 2])))
            self.assertTrue(np.all(bond_cn.loc[1,0,1,2,0,0].values == np.array([2, 2])))
            self.assertTrue(np.all(bond_cn.loc[1,1,1,2,1,0].values == np.array([1, 2])))
            self.assertTrue(np.all(bond_cn.loc[0,0,0,2,0,1].values == np.array([2, 2])))
            self.assertTrue(np.all(bond_cn.loc[0,1,0,2,1,1].values == np.array([2, 2])))

            # Check variant edge
            self.assertTrue(np.all(bond_cn.loc[0,1,1,2,1,0].values == np.array([1, 0])))

            model.total = False


        def test_recreate(self):

            rparams = sim_experiment.RearrangedGenome.default_params.copy()

            genome = sim_experiment.RearrangedGenome(100)

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

            rparams = sim_experiment.RearrangedGenome.default_params.copy()

            genome = sim_experiment.RearrangedGenome(100)

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

            rparams = sim_experiment.RearrangedGenome.default_params.copy()

            rparams['chromosome_lengths'] = {'1':20, '2':10}

            genome = sim_experiment.RearrangedGenome(4)

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


    unittest.main()


