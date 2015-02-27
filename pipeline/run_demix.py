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

import demix.cn_model as cn_model
import demix.cn_plot as cn_plot


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

    args = vars(argparser.parse_args())

    pyp = pypeliner.app.Pypeline([run_demix], args)

    pyp.sch.transform('create_experiment', (), {'mem':1},
        run_demix.create_experiment,
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
        mgd.TempInputFile('model'),
        mgd.TempInputFile('experiment'),
        mgd.TempInputFile('h_table.tsv'))

    pyp.run()

else:


    Experiment = collections.namedtuple('Experiment', [
        'segment_chromosome_id',
        'segment_start',
        'segment_end',
         'x',
         'l',
         'adjacencies',
         'breakpoints',
         'breakpoint_ids',
    ])


    def find_closest_segment_end(segment_data, breakpoint_data):

        break_ends = breakpoint_data[['prediction_id',
                                      'chromosome_1', 'strand_1', 'position_1',
                                      'chromosome_2', 'strand_2', 'position_2']]

        break_ends.set_index('prediction_id', inplace=True)
        break_ends.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in break_ends.columns])
        break_ends = break_ends.stack()

        segment_end = segment_data[['segment_id',
                                    'chromosome_1', 'strand_1', 'position_1',
                                    'chromosome_2', 'strand_2', 'position_2']]

        segment_end.set_index('segment_id', inplace=True)
        segment_end.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in segment_end.columns])
        segment_end = segment_end.stack()

        chromosomes = list(segment_end['chromosome'].unique())
        strands = ('+', '-')

        break_segment_table = list()

        for chromosome, strand in itertools.product(chromosomes, strands):

            break_end_pos = break_ends.loc[(break_ends['chromosome'] == chromosome) &
                                           (break_ends['strand'] == strand), 'position']
            break_end_pos.sort()

            segment_end_pos = segment_end.loc[(segment_end['chromosome'] == chromosome) &
                                              (segment_end['strand'] == strand), 'position']
            segment_end_pos.sort()

            right_idx = np.minimum(np.searchsorted(segment_end_pos.values, break_end_pos.values), len(segment_end_pos.values)-1)
            left_idx = np.maximum(right_idx - 1, 0)

            left_pos = segment_end_pos.values[left_idx]
            right_pos = segment_end_pos.values[right_idx]

            left_dist = break_end_pos.values - left_pos
            right_dist = right_pos - break_end_pos.values

            least_dist_idx = np.where(left_dist < right_dist, left_idx, right_idx)
            least_dist = np.minimum(left_dist, right_dist)
            
            least_dist_segments = pd.DataFrame.from_records(list(segment_end_pos.index.values[least_dist_idx]),
                                                            columns=['segment_id', 'segment_side'])
            
            break_end_pos = break_end_pos.reset_index()
            break_end_pos.columns = ['prediction_id', 'prediction_side', 'position']
            
            break_end_pos['dist'] = least_dist
            break_end_pos['segment_id'] = segment_end_pos.index.get_level_values(0)[least_dist_idx]
            break_end_pos['segment_side'] = segment_end_pos.index.get_level_values(1)[least_dist_idx]
            break_end_pos['chromosome'] = chromosome
            break_end_pos['strand'] = strand

            break_segment_table.append(break_end_pos)

        break_segment_table = pd.concat(break_segment_table, ignore_index=True)

        return break_segment_table


    def create_experiment(experiment_filename, count_filename, breakpoint_filename, min_length=100000, min_brk_dist=2000):

        count_data = pd.read_csv(count_filename, sep='\t')

        breakpoint_data = pd.read_csv(breakpoint_filename, sep='\t')

        count_data = count_data[count_data['length'] > min_length]

        count_data = count_data.sort(['chromosome_1', 'strand_1', 'position_1'])

        chromosomes = count_data['chromosome_1'].unique()

        # Filter breakpoints between chromosomes with no count data
        breakpoint_data = breakpoint_data[(
            (breakpoint_data['chromosome_1'].isin(chromosomes)) &
            (breakpoint_data['chromosome_2'].isin(chromosomes))
        )]

        # Ensure the data frame is indexed 0..n-1 and add the index as a column called 'index'
        count_data = count_data.reset_index(drop=True).reset_index()

        # Adjacent segments in the same chromosome
        adjacencies = set()
        for idx in xrange(len(count_data.index) - 1):
            if count_data.iloc[idx]['chromosome_1'] == count_data.iloc[idx+1]['chromosome_1']:
                adjacencies.add((idx, idx+1))

        # Table of segments closest to breakpoints
        break_segment_table = find_closest_segment_end(count_data, breakpoint_data)

        # Data is indexed by position in count_data table, add that index
        break_segment_table = break_segment_table.merge(count_data['segment_id'].reset_index(), how='left')

        # Should have a pair of breakends per breakpoint
        break_segment_table = break_segment_table.set_index(['prediction_id', 'prediction_side']) \
                                                 .unstack() \
                                                 .dropna() \
                                                 .reset_index()

        # Breakpoints as segment index, segment side (0/1)
        breakpoint_ids = dict()
        for idx, row in break_segment_table.iterrows():

            if row['dist'].sum() > 'min_brk_dist':
                continue

            n_1 = row['index']['1']
            n_2 = row['index']['2']

            side_1 = int(row['segment_side']['1']) - 1
            side_2 = int(row['segment_side']['2']) - 1

            # Remove small events that look like wild type adjacencies
            if (n_1, n_2) in adjacencies and side_1 == 1 and side_2 == 0:
                continue
            if (n_2, n_1) in adjacencies and side_2 == 1 and side_1 == 0:
                continue

            # No support for loop back inversions
            if (n_1, side_1) == (n_2, side_2):
                continue

            breakpoint_ids[frozenset([(n_1, side_1), (n_2, side_2)])] = row['prediction_id']

        x = count_data[['major_readcount', 'minor_readcount', 'readcount']].values
        l = count_data['length'].values

        exp = Experiment(
            count_data['chromosome_1'].values,
            count_data['position_1'].values,
            count_data['position_2'].values,
            x,
            l,
            adjacencies,
            breakpoint_ids.keys(),
            breakpoint_ids,
        )

        with open(experiment_filename, 'w') as experiment_file:
            pickle.dump(exp, experiment_file)


    def create_model(model_filename, experiment_filename):

        with open(experiment_filename, 'r') as experiment_file:
            exp = pickle.load(experiment_file)

        model = cn_model.CopyNumberModel(3, exp.adjacencies, exp.breakpoints)
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
                 model_filename, experiment_filename, h_table_filename):

        with open(experiment_filename, 'r') as experiment_file:
            exp = pickle.load(experiment_file)

        with open(model_filename, 'r') as model_file:
            model = pickle.load(model_file)

        h_table = pd.read_csv(h_table_filename, sep='\t')

        h = h_table.sort('log_posterior', ascending=False).iloc[0][['h_{0}'.format(idx) for idx in xrange(3)]].values.astype(float)

        model.e_step_method = 'genomegraph'

        cn, brk_cn = model.decode(exp.x, exp.l, h)

        fig = cn_plot.experiment_plot(exp, cn, h, model.p)

        fig.savefig(experiment_plot_filename, format='pdf', bbox_inches='tight')

        cn_table = pd.DataFrame({
            'chromosome':exp.segment_chromosome_id,
            'start':exp.segment_start,
            'end':exp.segment_end,
        })

        for m in xrange(1, exp.M):
            for l in xrange(2):
                cn_table['cn_{0}_{1}'.format(m, l)] = cn[:,m,l]

        cn_table.to_csv(cn_table_filename, sep='\t', index=False, header=True)

        brk_ids = list(exp.breakpoint_ids.values())
        brk_cn_values = [brk_cn[bp] for bp in exp.breakpoint_ids.iterkeys()]

        brk_cn_table = pd.DataFrame({'prediction_id':brk_ids})

        for m in xrange(1, exp.M):
            for l in xrange(2):
                brk_cn_table['cn_{0}_{1}'.format(m, l)] = brk_cn_values[:,m,l]

        brk_cn_table.to_csv(brk_cn_table_filename, sep='\t', index=False, header=True)








