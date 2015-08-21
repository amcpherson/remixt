import collections
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import remixt.cn_model
import remixt.cn_plot
import remixt.analysis.experiment

def init(
        learn_experiment_filename,
        learn_model_filename,
        infer_experiment_filename,
        infer_model_filename,
        candidate_h_filename_callback,
        candidate_h_plot_filename,
        segment_allele_count_filename,
        breakpoint_filename,
    ):

    # Prepare learning experiment file
    experiment = remixt.analysis.experiment.create_experiment(
        segment_allele_count_filename,
        breakpoint_filename,
    )

    with open(learn_experiment_filename, 'w') as f:
        pickle.dump(experiment, f)

    # Prepare learning model file
    model = remixt.cn_model.CopyNumberModel(experiment.adjacencies, experiment.breakpoints)
    model.emission_model = 'negbin'
    model.e_step_method = 'forwardbackward'
    model.total_cn = True

    model.infer_offline_parameters(experiment.x, experiment.l)

    with open(learn_model_filename, 'w') as f:
        pickle.dump(model, f)

    # Prepare inference experiment file
    experiment = remixt.analysis.experiment.create_experiment(
        segment_allele_count_filename,
        breakpoint_filename,
        min_length=None,
    )

    with open(infer_experiment_filename, 'w') as f:
        pickle.dump(experiment, f)

    # Prepare inference model file
    model = remixt.cn_model.CopyNumberModel(experiment.adjacencies, experiment.breakpoints)
    model.emission_model = 'negbin'
    model.e_step_method = 'genomegraph'
    model.total_cn = False

    model.infer_offline_parameters(experiment.x, experiment.l)

    with open(infer_model_filename, 'w') as f:
        pickle.dump(model, f)

    # Prepare candidate haploid depth initializations
    fig = plt.figure(figsize=(8,8))

    ax = plt.subplot(1, 1, 1)

    candidate_h_init = model.candidate_h(experiment.x, experiment.l, ax=ax)

    fig.savefig(candidate_h_plot_filename, format='pdf')

    for idx, h in enumerate(candidate_h_init):
        with open(candidate_h_filename_callback(idx), 'w') as f:
            pickle.dump(h, f)


def learn_h(
        h_opt_filename,
        experiment_filename,
        model_filename,
        h_init_filename,
    ):

    with open(experiment_filename, 'r') as f:
        experiment = pickle.load(f)

    with open(model_filename, 'r') as f:
        model = pickle.load(f)

    with open(h_init_filename, 'r') as f:
        h_init = pickle.load(f)

    # Optimize initial haploid read depth
    h, log_posterior, converged = model.optimize_h(experiment.x, experiment.l, h_init)

    optimize_result = {
        'h':h,
        'log_posterior':log_posterior,
        'converged':converged,
    }

    with open(h_opt_filename, 'w') as f:
        pickle.dump(optimize_result, f)


def tabulate_h(h_table_filename, h_opt_filenames):

    h_table = list()

    for idx, h_opt_filename in h_opt_filenames.iteritems():

        with open(h_opt_filename, 'r') as f:
            h_opt = pickle.load(f)

        h_row = dict()

        h_row['idx'] = idx
        h_row['log_posterior'] = h_opt['log_posterior']
        h_row['converged'] = h_opt['converged']

        for h_idx, h_value in enumerate(h_opt['h']):
            h_row['h_{}'.format(h_idx)] = h_value

        h_table.append(h_row)

    h_table = pd.DataFrame(h_table)

    h_table.to_csv(h_table_filename, sep='\t', index=False, na_rep='NA')


def infer_cn(
        cn_table_filename,
        brk_cn_table_filename,
        experiment_plot_filename,
        mix_filename,
        experiment_filename,
        model_filename,
        h_table_filename,
        model_debug_filename=None,
    ):

    with open(experiment_filename, 'r') as experiment_file:
        experiment = pickle.load(experiment_file)

    with open(model_filename, 'r') as model_file:
        model = pickle.load(model_file)

    h_table = pd.read_csv(h_table_filename, sep='\t')

    h = h_table.sort('log_posterior', ascending=False).iloc[0][['h_{0}'.format(idx) for idx in xrange(3)]].values.astype(float)
    h = h[~np.isnan(h)]

    mix = h / h.sum()

    M = h.shape[0]

    with open(mix_filename, 'w') as mix_file:
        mix_file.write('\t'.join([str(a) for a in mix]) + '\n')

    cn, brk_cn = model.decode(experiment.x, experiment.l, h)

    if model_debug_filename is not None:
        with open(model_debug_filename, 'w') as model_debug_file:
            pickle.dump(model, model_debug_file)

    cn_table = remixt.analysis.experiment.create_cn_table(experiment, cn, h, model.p)

    cn_table['log_likelihood'] = model.log_likelihood_cn(experiment.x, experiment.l, cn, h)
    cn_table['log_prior'] = model.log_prior_cn(experiment.l, cn)
    cn_table['major_readcount'] = experiment.x[:,0]
    cn_table['minor_readcount'] = experiment.x[:,1]
    cn_table['total_readcount'] = experiment.x[:,2]
    cn_table['major_expected'] = model.expected_read_count(experiment.l, cn, h, model.p)[:,0]
    cn_table['minor_expected'] = model.expected_read_count(experiment.l, cn, h, model.p)[:,1]
    cn_table['total_expected'] = model.expected_read_count(experiment.l, cn, h, model.p)[:,2]
    cn_table['major_residual'] = np.absolute(cn_table['major_readcount'] - cn_table['major_expected'])
    cn_table['minor_residual'] = np.absolute(cn_table['minor_readcount'] - cn_table['minor_expected'])
    cn_table['total_residual'] = np.absolute(cn_table['total_readcount'] - cn_table['total_expected'])

    cn_table.to_csv(cn_table_filename, sep='\t', index=False, header=True)

    fig = remixt.cn_plot.experiment_plot(experiment, cn, h, model.p)

    fig.savefig(experiment_plot_filename, format='pdf', bbox_inches='tight')

    brk_cn_table = list()

    for bp, bp_cn in brk_cn.iteritems():
        ((n_1, ell_1), side_1), ((n_2, ell_2), side_2) = bp
        bp_id = experiment.breakpoint_ids[frozenset([(n_1, side_1), (n_2, side_2)])]
        brk_cn_table.append([bp_id, ell_1, ell_2] + list(bp_cn[1:]))

    brk_cn_cols = ['prediction_id', 'allele_1', 'allele_2']
    for m in xrange(1, M):
        brk_cn_cols.append('cn_{0}'.format(m))
    brk_cn_table = pd.DataFrame(brk_cn_table, columns=brk_cn_cols)

    brk_cn_table.to_csv(brk_cn_table_filename, sep='\t', index=False, header=True)



