import collections
import itertools
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import demix.cn_model


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

    segment_end = pd.concat([
        pd.DataFrame({
            'segment_id':segment_data['segment_id'],
            'segment_side':0,
            'chromosome':segment_data['chromosome'],
            'strand':'-',
            'position':segment_data['start']
        }),
        pd.DataFrame({
            'segment_id':segment_data['segment_id'],
            'segment_side':1,
            'chromosome':segment_data['chromosome'],
            'strand':'+',
            'position':segment_data['end']
        }),
    ])
    segment_end.set_index(['segment_id', 'segment_side'], inplace=True)

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
    count_data = count_data.reset_index().rename(columns={'index':'segment_id'})

    breakpoint_data = pd.read_csv(breakpoint_filename, sep='\t')

    if min_length is not None:
        count_data = count_data[count_data['length'] > min_length]

    count_data = count_data.sort(['chromosome', 'start'])

    chromosomes = count_data['chromosome'].unique()

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
        if count_data.iloc[idx]['chromosome'] == count_data.iloc[idx+1]['chromosome']:
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

        if row['dist'].sum() > min_brk_dist:
            continue

        n_1 = row['index']['1']
        n_2 = row['index']['2']

        side_1 = row['segment_side']['1']
        side_2 = row['segment_side']['2']

        # Remove small events that look like wild type adjacencies
        if (n_1, n_2) in adjacencies and side_1 == 1 and side_2 == 0:
            continue
        if (n_2, n_1) in adjacencies and side_2 == 1 and side_1 == 0:
            continue

        # No support for loop back inversions
        if (n_1, side_1) == (n_2, side_2):
            continue

        breakpoint_ids[frozenset([(n_1, side_1), (n_2, side_2)])] = row['prediction_id'].iloc[0]

    x = count_data[['major_readcount', 'minor_readcount', 'readcount']].values
    l = count_data['length'].values

    exp = Experiment(
        count_data['chromosome'].values,
        count_data['start'].values,
        count_data['end'].values,
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


