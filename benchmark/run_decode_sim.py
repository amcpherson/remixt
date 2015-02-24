import ConfigParser
import itertools
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pypeliner
import pypeliner.managed as mgd

import experiment_sim
import cn_model
import cn_plot
import sim_pipeline

if __name__ == '__main__':

    import run_decode_sim

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pypeliner.app.add_arguments(argparser)
    argparser.add_argument('simdef', help='Simulation Definition Filename')
    argparser.add_argument('table', help='Output Table Filename')

    args = vars(argparser.parse_args())

    pyp = pypeliner.app.Pypeline([run_decode_sim], args)

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

    pyp.sch.transform('set_e_step', ('bysim',), {'mem':1,'local':True},
        run_decode_sim.set_e_step,
        mgd.OutputChunks('bysim', 'byestep'))

    pyp.sch.transform('decode_cn', ('bysim', 'byestep'), {'mem':8},
        run_decode_sim.decode_cn,
        None,
        mgd.TempOutputFile('stats.tsv', 'bysim', 'byestep'),
        mgd.TempOutputFile('inferred_cn_plot.pdf', 'bysim', 'byestep'),
        mgd.InputInstance('bysim'),
        mgd.InputInstance('byestep'),
        mgd.TempInputFile('experiment', 'bysim'),
        mgd.TempOutputFile('cn_table.tsv', 'bysim', 'byestep'),
        mgd.TempOutputFile('brk_cn_table.tsv', 'bysim', 'byestep'))

    pyp.sch.transform('merge_stats_1', ('bysim',), {'mem':1},
        sim_pipeline.merge_tables,
        None,
        mgd.TempOutputFile('stats.tsv', 'bysim'),
        mgd.TempInputFile('stats.tsv', 'bysim', 'byestep'))

    pyp.sch.transform('merge_stats_2', (), {'mem':1},
        sim_pipeline.merge_tables,
        None,
        mgd.TempOutputFile('stats.tsv'),
        mgd.TempInputFile('stats.tsv', 'bysim'))

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
        mgd.TempInputFile('stats.tsv'),
        mgd.TempInputFile('exp_table.tsv'))

    pyp.run()

else:

    def set_e_step():
        return ('genomegraph', 'viterbi', 'independent')
    
    def decode_cn(stats_filename, cn_plot_filename, sim_id, e_step_method, experiment_filename, cn_filename, brk_cn_filename):

        with open(experiment_filename, 'r') as experiment_file:
            exp = pickle.load(experiment_file)

        model = cn_model.CopyNumberModel(3, exp.adjacencies, exp.breakpoints)
        model.emission_model = 'negbin'
        model.e_step_method = e_step_method
        model.total_cn = False

        model.infer_offline_parameters(exp.x, exp.l)

        cn, brk_cn = model.decode(exp.x, exp.l, exp.h_pred)

        fig = cn_plot.experiment_plot(exp, cn=cn)

        fig.savefig(cn_plot_filename, format='pdf', bbox_inches='tight', dpi=300)

        cn_table = pd.DataFrame({'n':xrange(cn.shape[0])})

        cn_table['l'] = exp.l

        true_cols = list()
        for m in xrange(1, exp.M):
            for ell in xrange(2):
                cn_table['true_cn_{0}_{1}'.format(m, ell)] = exp.cn[:,m,ell]
                true_cols.append('true_cn_{0}_{1}'.format(m, ell))

        pred_cols = list()
        for m in xrange(1, exp.M):
            for ell in xrange(2):
                cn_table['pred_cn_{0}_{1}'.format(m, ell)] = cn[:,m,ell]
                pred_cols.append('pred_cn_{0}_{1}'.format(m, ell))

        h_ratio = exp.h[1:].min() / exp.h[1:].max()
        if h_ratio > 0.75:
            cn_table['cn_correct'] = (np.sort(cn_table[true_cols].values, axis=1) == np.sort(cn_table[pred_cols].values, axis=1)).all(axis=1)
        else:
            cn_table['cn_correct'] = (cn_table[true_cols].values == cn_table[pred_cols].values).all(axis=1)

        cn_table['true_present'] = (cn_table[true_cols] > 0).any(axis=1)
        cn_table['pred_present'] = (cn_table[pred_cols] > 0).any(axis=1)

        cn_table['true_subclonal'] = (cn_table[true_cols] == 0).any(axis=1) & cn_table['true_present']
        cn_table['pred_subclonal'] = (cn_table[pred_cols] == 0).any(axis=1) & cn_table['pred_present']

        cn_table.to_csv(cn_filename, sep='\t', index=False)

        true_brk_cn = exp.genome_mixture.genome_collection.breakpoint_copy_number

        brk_cn_table = list()
        for bp in set(true_brk_cn.keys() + brk_cn.keys()):

            (((n_1, ell_1), side_1), ((n_2, ell_2), side_2)) = sorted(bp)

            brk_cn_row = {'n_1':n_1, 'ell_1':ell_1, 'side_1':side_1,
                          'n_2':n_2, 'ell_2':ell_2, 'side_2':side_2}

            if bp in brk_cn and np.all(brk_cn[bp] == 0) and bp not in true_brk_cn:
                continue

            if bp in brk_cn:
                for m in xrange(1, exp.M):
                    brk_cn_row['pred_cn_{0}'.format(m)] = brk_cn[bp][m]

            if bp in true_brk_cn:
                for m in xrange(1, exp.M):
                    brk_cn_row['true_cn_{0}'.format(m)] = true_brk_cn[bp][m]

            brk_cn_table.append(brk_cn_row)

        brk_cn_table = pd.DataFrame(brk_cn_table)

        brk_cn_table = brk_cn_table.set_index(['n_1', 'ell_1', 'side_1', 'n_2', 'ell_2', 'side_2']).sort_index().reset_index()

        true_cols = list()
        for m in xrange(1, exp.M):
            true_cols.append('true_cn_{0}'.format(m))

        pred_cols = list()
        for m in xrange(1, exp.M):
            pred_cols.append('pred_cn_{0}'.format(m))

        # Sum across alleles
        brk_cn_table = brk_cn_table.groupby(['n_1', 'side_1', 'n_2', 'side_2'])[true_cols+pred_cols].sum().reset_index()

        brk_cn_table['cn_correct'] = (brk_cn_table[true_cols].values == brk_cn_table[pred_cols].values).all(axis=1)

        brk_cn_table['true_present'] = (brk_cn_table[true_cols] > 0).any(axis=1)
        brk_cn_table['pred_present'] = (brk_cn_table[pred_cols] > 0).any(axis=1)

        brk_cn_table['true_subclonal'] = (brk_cn_table[true_cols] == 0).any(axis=1) & brk_cn_table['true_present']
        brk_cn_table['pred_subclonal'] = (brk_cn_table[pred_cols] == 0).any(axis=1) & brk_cn_table['pred_present']

        brk_cn_table.to_csv(brk_cn_filename, sep='\t', index=False)

        stats = dict()

        stats['sim_id'] = sim_id
        stats['e_step_method'] = e_step_method

        stats['cn_correct_proportion'] = float(cn_table['cn_correct'].sum()) / float(len(cn_table.index))

        stats['cn_subclonal_num_true'] = float(cn_table['true_subclonal'].sum())
        stats['cn_subclonal_num_pos'] = float(cn_table['pred_subclonal'].sum())
        stats['cn_subclonal_num_true_pos'] = float((cn_table['pred_subclonal'] & cn_table['true_subclonal']).sum())

        stats['cn_brk_correct_proportion'] = float(brk_cn_table['cn_correct'].sum()) / float(len(brk_cn_table.index))

        stats['cn_brk_present_num_true'] = float(brk_cn_table['true_present'].sum())
        stats['cn_brk_present_num_pos'] = float(brk_cn_table['pred_present'].sum())
        stats['cn_brk_present_num_true_pos'] = float((brk_cn_table['pred_present'] & brk_cn_table['true_present']).sum())

        stats['cn_brk_subclonal_num_true'] = float(brk_cn_table['true_subclonal'].sum())
        stats['cn_brk_subclonal_num_pos'] = float(brk_cn_table['pred_subclonal'].sum())
        stats['cn_brk_subclonal_num_true_pos'] = float((brk_cn_table['pred_subclonal'] & brk_cn_table['true_subclonal']).sum())

        stats = pd.DataFrame([stats])

        for m in ('cn_subclonal', 'cn_brk_present', 'cn_brk_subclonal'):

            stats[m+'_precision'] = stats[m+'_num_true_pos'] / stats[m+'_num_pos']
            stats[m+'_recall'] = stats[m+'_num_true_pos'] / stats[m+'_num_true']
            stats[m+'_fmeasure'] = 2 * stats[m+'_precision'] * stats[m+'_recall'] / (stats[m+'_precision'] + stats[m+'_recall'])

        stats.to_csv(stats_filename, sep='\t', index=False)


