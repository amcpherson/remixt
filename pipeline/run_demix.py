import os
import sys
import ConfigParser
import itertools
import argparse
import pickle
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pypeliner
import pypeliner.managed as mgd


demix_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))

default_config_filename = os.path.join(demix_directory, 'defaultconfig.py')

sys.path.append(demix_directory)

import demix.cn_model
import demix.cn_plot
import demix.analysis.experiment


if __name__ == '__main__':

    import run_demix

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('counts',
                           help='Input segment counts filename')

    argparser.add_argument('breakpoints',
                           help='Input breakpoints filename')

    argparser.add_argument('cn',
                           help='Output segment copy number filename')

    argparser.add_argument('brk_cn',
                           help='Output breakpoint copy number filename')

    argparser.add_argument('cn_plot',
                           help='Output segment copy number plot pdf filename')

    argparser.add_argument('mix',
                           help='Output mixture filename')

    args = vars(argparser.parse_args())

    pyp = pypeliner.app.Pypeline([run_demix], args)

    pyp.sch.transform('create_experiment', (), {'mem':1},
        demix.analysis.experiment.create_experiment,
        None,
        mgd.TempOutputFile('experiment'),
        mgd.InputFile(args['counts']),
        mgd.InputFile(args['breakpoints']))

    pyp.sch.transform('create_model', (), {'mem':1},
        run_demix.create_model,
        None,
        mgd.TempOutputFile('model'),
        mgd.TempInputFile('experiment'))

    pyp.sch.transform('create_candidate_h', (), {'mem':1},
        run_demix.create_candidate_h,
        mgd.TempOutputObj('h', 'byh'),
        mgd.TempInputFile('model'),
        mgd.TempInputFile('experiment'),
        mgd.TempOutputFile('candidate_h_plot.pdf'))

    pyp.sch.transform('optimize_h', ('byh',), {'mem':1},
        run_demix.optimize_h,
        mgd.TempOutputObj('opt_h', 'byh'),
        mgd.TempInputFile('model'),
        mgd.TempInputFile('experiment'),
        mgd.TempInputObj('h', 'byh'))

    pyp.sch.transform('tabulate_h', (), {'mem':1},
        run_demix.tabulate_h,
        None,
        mgd.TempOutputFile('h_table.tsv'),
        mgd.TempInputObj('opt_h', 'byh'))

    pyp.sch.transform('infer_cn', (), {'mem':8},
        run_demix.infer_cn,
        None,
        mgd.OutputFile(args['cn']),
        mgd.OutputFile(args['brk_cn']),
        mgd.OutputFile(args['cn_plot']),
        mgd.OutputFile(args['mix']),
        mgd.TempInputFile('model'),
        mgd.TempInputFile('experiment'),
        mgd.TempInputFile('h_table.tsv'))

    pyp.run()

else:


    def create_model(model_filename, experiment_filename):

        with open(experiment_filename, 'r') as experiment_file:
            exp = pickle.load(experiment_file)

        model = demix.cn_model.CopyNumberModel(3, exp.adjacencies, exp.breakpoints)
        model.emission_model = 'negbin'
        model.e_step_method = 'forwardbackward'
        model.total_cn = True

        model.infer_offline_parameters(exp.x, exp.l)

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


    def tabulate_h(h_table_filename, h_opt):

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

        h_table.to_csv(h_table_filename, sep='\t', index=False)


    def infer_cn(cn_table_filename, brk_cn_table_filename, experiment_plot_filename,
            mix_filename, model_filename, experiment_filename, h_table_filename):

        with open(experiment_filename, 'r') as experiment_file:
            exp = pickle.load(experiment_file)

        with open(model_filename, 'r') as model_file:
            model = pickle.load(model_file)

        h_table = pd.read_csv(h_table_filename, sep='\t')

        h = h_table.sort('log_posterior', ascending=False).iloc[0][['h_{0}'.format(idx) for idx in xrange(3)]].values.astype(float)

        mix = h / h.sum()

        with open(mix_filename, 'w') as mix_file:
            mix_file.write('\t'.join([str(a) for a in mix]))

        model.e_step_method = 'genomegraph'

        cn, brk_cn = model.decode(exp.x, exp.l, h)

        cn_table = demix.analysis.experiment.create_cn_table(exp, cn, h, model.p)

        cn_table.to_csv(cn_table_filename, sep='\t', index=False, header=True)

        fig = demix.cn_plot.cn_mix_plot(cn_table)

        fig.savefig(experiment_plot_filename, format='pdf', bbox_inches='tight')

        brk_cn_table = list()

        for bp, bp_cn in brk_cn.iteritems():
            ((n_1, ell_1), side_1), ((n_2, ell_2), side_2) = bp
            bp_id = exp.breakpoint_ids[frozenset([(n_1, side_1), (n_2, side_2)])]
            brk_cn_table.append([bp_id, ell_1, ell_2] + list(bp_cn[1:]))

        brk_cn_cols = ['prediction_id', 'allele_1', 'allele_2']
        for m in xrange(1, model.M):
            brk_cn_cols.append('cn_{0}'.format(m))
        brk_cn_table = pd.DataFrame(brk_cn_table, columns=brk_cn_cols)

        brk_cn_table.to_csv(brk_cn_table_filename, sep='\t', index=False, header=True)



