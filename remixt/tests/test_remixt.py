import copy
import os
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

import remixt.analysis.experiment
import remixt.analysis.pipeline
import remixt.cn_model
import remixt.cn_plot
import remixt.simulations.experiment
import remixt.simulations.pipeline
import remixt.ui.write_results
import remixt.visualize


def simulate_experiment(seed, N=300, M=3):
    """ Simulate a small experiment with a normal and M - 1 tumour clones.
    """
    np.random.seed(seed)

    params = {
        'N': N,
        'M': M,
        'frac_normal': 0.4,
        'frac_clone_1': 0.4,
        'num_ancestral_events': 20,
        'num_descendent_events': 10,
        'num_false_breakpoints': 0,
        'ploidy': 2.5,
    }

    rh_sampler = remixt.simulations.experiment.RearrangementHistorySampler(params)
    gc_sampler = remixt.simulations.experiment.GenomeCollectionSampler(rh_sampler, params)
    gc = gc_sampler.sample_genome_collection()

    gm_sampler = remixt.simulations.experiment.GenomeMixtureSampler(params)
    gm = gm_sampler.sample_genome_mixture(gc)

    experiment_sampler = remixt.simulations.experiment.ExperimentSampler(params)
    return experiment_sampler.sample_experiment(gm)


class remixt_unittest(unittest.TestCase):

    def test_create_cn_states(self):
        """ Copy number states match a naive enumeration.
        """
        num_clones, num_alleles, cn_max, cn_diff_max = 3, 2, 4, 1

        cn_states = remixt.cn_model.BreakpointModel.create_cn_states(
            None, num_clones, num_alleles, cn_max, cn_diff_max)

        built = set()
        for cn in cn_states:
            self.assertEqual(tuple(cn[0]), (1, 1))
            key = tuple(cn[1:].flatten())
            self.assertNotIn(key, built)
            built.add(key)

        expected = set()
        for b1, b2, c1, c2 in np.ndindex((cn_max + 1,) * 4):
            if b1 + b2 > cn_max or c1 + c2 > cn_max:
                continue
            if abs(b1 - c1) > cn_diff_max or abs(b2 - c2) > cn_diff_max:
                continue
            expected.add(frozenset([(b1, b2, c1, c2), (b2, b1, c2, c1)]))

        self.assertEqual(len(built), len(expected))
        for state in expected:
            self.assertTrue(any(key in built for key in state))

    def test_simulate_experiment(self):
        """ Simulated experiments have consistent read counts and copy number.
        """
        experiment = simulate_experiment(2014)

        self.assertEqual(experiment.x.shape, (experiment.N, 3))
        self.assertEqual(experiment.cn.shape, (experiment.N, 3, 2))
        self.assertEqual(len(experiment.h), 3)
        self.assertTrue(np.all(experiment.x[:, 0] >= experiment.x[:, 1]))
        self.assertTrue(np.all(experiment.x[:, 0] + experiment.x[:, 1] <= experiment.x[:, 2]))
        self.assertTrue(np.all(experiment.cn[:, 0, :] == 1))
        self.assertTrue(np.all(
            experiment.allele_a_readcount + experiment.allele_b_readcount == experiment.x[:, 0] + experiment.x[:, 1]))

        table = remixt.analysis.experiment.create_cn_table(experiment, experiment.cn, experiment.h)
        self.assertEqual(len(table.index), experiment.N)

        remixt.cn_plot.experiment_plot(experiment, experiment.cn, experiment.h, maxcopies=6)

    def test_fit_simulated(self):
        """ Init, fit, collate and report on a simulated experiment recover the mixture.
        """
        experiment = simulate_experiment(2015)

        with tempfile.TemporaryDirectory() as tmp:
            experiment_filename = os.path.join(tmp, 'experiment.pickle')
            with open(experiment_filename, 'wb') as f:
                pickle.dump(experiment, f)

            init_results_filename = os.path.join(tmp, 'init.h5')
            init_params = remixt.analysis.pipeline.init(init_results_filename, experiment_filename)
            self.assertGreater(len(init_params), 0)

            # Fit the haploid depths closest to the truth with the strongest and
            # weakest divergence penalties, the weakest should be selected by elbo
            def h_error(params):
                h = np.array([
                    params['h_normal'],
                    params['h_tumour'] * params['mix_frac'],
                    params['h_tumour'] * (1. - params['mix_frac'])])
                return np.abs(np.sort(h[1:]) - np.sort(experiment.h[1:])).sum() + abs(h[0] - experiment.h[0])
            best_id = min(init_params, key=lambda i: h_error(init_params[i]))
            best_h = tuple(init_params[best_id][k] for k in ('h_normal', 'h_tumour', 'mix_frac'))
            same_h = [
                i for i, params in init_params.items()
                if tuple(params[k] for k in ('h_normal', 'h_tumour', 'mix_frac')) == best_h]
            strong_id = max(same_h, key=lambda i: init_params[i]['divergence_weight'])
            weak_id = min(same_h, key=lambda i: init_params[i]['divergence_weight'])
            self.assertNotEqual(strong_id, weak_id)

            fit_results_filenames = {}
            for init_id in (strong_id, weak_id):
                fit_results_filenames[init_id] = os.path.join(tmp, 'fit_{}.pickle'.format(init_id))
                remixt.analysis.pipeline.fit_task(
                    fit_results_filenames[init_id], experiment_filename, init_params[init_id])

            results_filename = os.path.join(tmp, 'results.h5')
            remixt.analysis.pipeline.collate(
                results_filename, experiment_filename, init_results_filename, fit_results_filenames)

            with pd.HDFStore(results_filename, 'r') as store:
                self.assertEqual(store['failed_fits'].shape[0], 0)
                stats = store['stats'].set_index('init_id')
                params = store['params']
                cn_table = store['/cn']
                brk_cn_table = store['/brk_cn']
                mix = store['/mix'].values

            init_id = int(params['optimal_init_id'])
            self.assertEqual(init_id, weak_id)
            self.assertGreater(stats.loc[weak_id, 'elbo'], stats.loc[strong_id, 'elbo'])
            self.assertAlmostEqual(params['max_depth'], init_params[weak_id]['max_depth'])
            for col in ('is_short', 'is_high_depth', 'is_low_genotyped', 'prob_is_outlier_total', 'num_hap_blocks'):
                self.assertIn(col, cn_table)
            self.assertTrue(np.all(
                (cn_table['total_likelihood_mask'] == 1) == ((cn_table['is_short'] == 0) & (cn_table['is_high_depth'] == 0))))

            sample = remixt.visualize.load_sample(results_filename)
            self.assertEqual(sample['optimal_id'], init_id)
            self.assertEqual(len(sample['segments']['start']), experiment.N)

            # Tables of the optimal solution, and of the other solution by init id
            cn_filename = os.path.join(tmp, 'cn.tsv')
            brk_cn_filename = os.path.join(tmp, 'brk_cn.tsv')
            meta_filename = os.path.join(tmp, 'meta.yaml')
            remixt.ui.write_results.write_results_tables(
                results_filename, cn_filename, brk_cn_filename, meta_filename)
            written_cn = pd.read_csv(cn_filename, sep='\t')
            self.assertEqual(len(written_cn.index), experiment.N)
            self.assertTrue(np.array_equal(written_cn['major_1'].values, cn_table['major_1'].values))
            with open(meta_filename) as f:
                meta = yaml.safe_load(f)
            self.assertEqual(meta['init_id'], init_id)
            self.assertTrue(meta['is_optimal'])
            self.assertAlmostEqual(meta['params']['max_depth'], init_params[weak_id]['max_depth'])
            self.assertEqual(len(meta['h']), 3)

            remixt.ui.write_results.write_results_tables(
                results_filename, cn_filename, brk_cn_filename, meta_filename, solution=strong_id)
            with open(meta_filename) as f:
                meta = yaml.safe_load(f)
            self.assertEqual(meta['init_id'], strong_id)
            self.assertFalse(meta['is_optimal'])

        evaluation = remixt.simulations.pipeline.evaluate_results(
            experiment.genome_mixture, cn_table, brk_cn_table, mix)

        cn_evaluation = evaluation['cn_evaluation']
        mix_results = evaluation['mix_results']
        print(cn_evaluation.to_string())
        print(mix_results.to_string())

        for m in range(3):
            self.assertAlmostEqual(mix_results['mix_true_' + str(m)], mix_results['mix_pred_' + str(m)], delta=0.05)
        self.assertGreater(cn_evaluation['proportion_cn_correct'], 0.75)
        self.assertGreater(cn_evaluation['proportion_dom_cn_correct'], 0.8)
        self.assertGreater(cn_evaluation['pred_proportion_divergent'], 0.1)

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

        # Expected sequences from the segment copies of each rearranged chromosome
        complement = str.maketrans('ACGT', 'TGCA')
        true_sequences = []
        for chromosome in genome.chromosomes:
            sequence = ''
            for (segment_idx, allele), orientation in chromosome:
                chromosome_id = genome.segment_chromosome_id[segment_idx]
                start, end = genome.segment_start[segment_idx], genome.segment_end[segment_idx]
                segment_sequence = germline_genome[(chromosome_id, allele)][start:end]
                if orientation < 0:
                    segment_sequence = segment_sequence[::-1].translate(complement)
                sequence += segment_sequence
            true_sequences.append(sequence)

        self.assertGreater(len(true_sequences), 0)
        self.assertTrue(any(orientation < 0 for chromosome in genome.chromosomes for _, orientation in chromosome))

        test_sequences = genome.create_chromosome_sequences(germline_genome)

        self.assertEqual(test_sequences, true_sequences)


if __name__ == '__main__':
    unittest.main()
