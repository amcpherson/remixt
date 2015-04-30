import itertools
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pypeliner
import pypeliner.managed as mgd

import demix_paths
import demix.simulations.experiment as sim_experiment
import demix.cn_model as cn_model
import demix.cn_plot as cn_plot
import sim_pipeline

if __name__ == '__main__':

    import run_inference_sim

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pypeliner.app.add_arguments(argparser)
    argparser.add_argument('simdef', help='Simulation Definition Filename')
    argparser.add_argument('table', help='Output Table Filename')

    args = vars(argparser.parse_args())

    pyp = pypeliner.app.Pypeline([run_inference_sim], args)

    pyp.sch.transform('read_sim_defs', (), {'mem':1,'local':True},
        sim_pipeline.read_sim_defs,
        mgd.TempOutputObj('settings', 'bysim'),
        mgd.InputFile(args['simdef']))

    pyp.sch.transform('simulate_genomes', ('bysim',), {'mem':4},
        sim_pipeline.simulate_genomes,
        None,
        mgd.TempOutputFile('genomes', 'bysim'),
        mgd.TempInputObj('settings', 'bysim'))

    pyp.sch.transform('simulate_mixture', ('bysim',), {'mem':1},
        sim_pipeline.simulate_mixture,
        None,
        mgd.TempOutputFile('mixture', 'bysim'),
        mgd.TempInputFile('genomes', 'bysim'),
        mgd.TempInputObj('settings', 'bysim'))

    pyp.sch.transform('simulate_experiment', ('bysim',), {'mem':1},
        sim_pipeline.simulate_experiment,
        None,
        mgd.TempOutputFile('experiment', 'bysim'),
        mgd.TempInputFile('mixture', 'bysim'),
        mgd.TempInputObj('settings', 'bysim'))

    pyp.sch.transform('tabulate_experiment', ('bysim',), {'mem':1},
        sim_pipeline.tabulate_experiment,
        None,
        mgd.TempOutputFile('exp_table.tsv', 'bysim'),
        mgd.InputInstance('bysim'),
        mgd.TempInputFile('experiment', 'bysim'))

    pyp.sch.transform('plot_experiment', ('bysim',), {'mem':8},
        sim_pipeline.plot_experiment,
        None,
        mgd.TempOutputFile('experiment_plot.pdf', 'bysim'),
        mgd.TempInputFile('experiment', 'bysim'))

    pyp.sch.transform('create_model', ('bysim',), {'mem':1},
        run_inference_sim.create_model,
        None,
        mgd.TempOutputFile('model', 'bysim'),
        mgd.TempInputFile('experiment', 'bysim'),
        mgd.TempInputObj('settings', 'bysim'))

    pyp.sch.transform('create_candidate_h', ('bysim',), {'mem':1},
        run_inference_sim.create_candidate_h,
        mgd.TempOutputObj('h', 'bysim', 'byh'),
        mgd.TempInputFile('model', 'bysim'),
        mgd.TempInputFile('experiment', 'bysim'),
        mgd.TempOutputFile('candidate_h_plot.pdf', 'bysim'))

    pyp.sch.transform('optimize_h', ('bysim', 'byh'), {'mem':1},
        run_inference_sim.optimize_h,
        mgd.TempOutputObj('opt_h', 'bysim', 'byh'),
        mgd.TempInputFile('model', 'bysim'),
        mgd.TempInputFile('experiment', 'bysim'),
        mgd.TempInputObj('h', 'bysim', 'byh'))

    pyp.sch.transform('tabulate_h', ('bysim',), {'mem':1},
        run_inference_sim.tabulate_h,
        None,
        mgd.TempOutputFile('h_table.tsv', 'bysim'),
        mgd.InputInstance('bysim'),
        mgd.TempInputObj('opt_h', 'bysim', 'byh'))

    pyp.sch.transform('plot_inferred_cn', ('bysim',), {'mem':8},
        run_inference_sim.plot_inferred_cn,
        None,
        mgd.TempOutputFile('inferred_cn_plot.pdf', 'bysim'),
        mgd.TempInputFile('model', 'bysim'),
        mgd.TempInputFile('experiment', 'bysim'),
        mgd.TempInputFile('h_table.tsv', 'bysim'))

    pyp.sch.transform('merge_h_table', (), {'mem':1},
        sim_pipeline.merge_tables,
        None,
        mgd.TempOutputFile('h_table.tsv'),
        mgd.TempInputFile('h_table.tsv', 'bysim'))

    pyp.sch.transform('merge_exp_table', (), {'mem':1},
        sim_pipeline.merge_tables,
        None,
        mgd.TempOutputFile('exp_table.tsv'),
        mgd.TempInputFile('exp_table.tsv', 'bysim'))

    pyp.sch.transform('tabulate_results', (), {'mem':1},
        sim_pipeline.tabulate_results,
        None,
        mgd.OutputFile(args['table']),
        mgd.TempInputObj('settings', 'bysim'),
        mgd.TempInputFile('h_table.tsv'),
        mgd.TempInputFile('exp_table.tsv'))

    pyp.run()

else:

    def create_model(model_filename, experiment_filename, params):

        with open(experiment_filename, 'r') as experiment_file:
            exp = pickle.load(experiment_file)

        assert exp.M == 3

        model = cn_model.CopyNumberModel(3, exp.adjacencies, exp.breakpoints)
        model.emission_model = 'negbin'
        model.e_step_method = 'forwardbackward'
        model.total_cn = True

        model.infer_offline_parameters(exp.x, exp.l)

        for key, _ in model.__dict__.iteritems():
            if key in params:
                model.__dict__[key] = params[key]

        with open(model_filename, 'w') as model_file:
            pickle.dump(model, model_file)


    def create_candidate_h(model_filename, experiment_filename, candidate_h_plot_filename):

        with open(model_filename, 'r') as model_file:
            model = pickle.load(model_file)

        with open(experiment_filename, 'r') as experiment_file:
            exp = pickle.load(experiment_file)

        fig = plt.figure(figsize=(8,8))

        ax = plt.subplot(1, 1, 1)

        candidate_h_init = model.candidate_h(exp.x, exp.l, ax=ax)

        fig.savefig(candidate_h_plot_filename, format='pdf')

        return dict(enumerate([tuple(h) for h in candidate_h_init]))


    def optimize_h(model_filename, experiment_filename, h_init):

        with open(experiment_filename, 'r') as experiment_file:
            exp = pickle.load(experiment_file)

        with open(model_filename, 'r') as model_file:
            model = pickle.load(model_file)

        h, log_posterior, converged = model.optimize_h(exp.x, exp.l, np.array(h_init))

        return list(h), log_posterior, converged


    def tabulate_h(h_table_filename, sim_id, h_opt):

        h_table = list()

        for idx, (h, log_posterior, converged) in h_opt.iteritems():

            h_row = dict()

            h_row['idx'] = idx
            h_row['log_posterior'] = log_posterior
            h_row['converged'] = converged

            for h_idx, h_value in enumerate(h):
                h_row['h_{}'.format(h_idx)] = h_value

            h_table.append(h_row)

        h_table = pd.DataFrame(h_table)

        h_table['sim_id'] = sim_id

        h_table.to_csv(h_table_filename, sep='\t', index=False)


    def plot_inferred_cn(experiment_plot_filename, model_filename, experiment_filename, h_table_filename):

        with open(experiment_filename, 'r') as experiment_file:
            exp = pickle.load(experiment_file)

        with open(model_filename, 'r') as model_file:
            model = pickle.load(model_file)

        h_table = pd.read_csv(h_table_filename, sep='\t')

        h = h_table.sort('log_posterior', ascending=False).iloc[0][['h_{0}'.format(idx) for idx in xrange(3)]].values.astype(float)

        cn, brk_cn = model.decode(exp.x, exp.l, h)

        fig = cn_plot.experiment_plot(exp, cn, h, model.p)

        fig.savefig(experiment_plot_filename, format='pdf', bbox_inches='tight', dpi=300)


