import collections
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import remixt.likelihood
import remixt.cn_model
import remixt.em
import remixt.genome_graph
import remixt.analysis.experiment
import remixt.analysis.readdepth


def create_cn_prior_matrix(cn_proportions_filename):
    """ Create a matrix of prior probabilities for copy number states.

    Args:
        cn_proportions_filename (str): tsv table of proportions of each state

    Returns:
        numpy.array: copy number prior matrix

    """

    cn_proportions = pd.read_csv(cn_proportions_filename, sep='\t',
        converters={'major':int, 'minor':int})

    cn_max = cn_proportions['major'].max()

    cn_amp_prior = 1. - cn_proportions['proportion'].sum()

    cn_prior = cn_proportions.set_index(['major', 'minor'])['proportion'].unstack().fillna(0.0)
    cn_prior = cn_prior.reindex(columns=range(cn_max+1), index=range(cn_max+1))

    assert not cn_prior.isnull().any().any()

    cn_prior = cn_prior.values
    cn_prior = cn_prior + cn_prior.T - cn_prior * np.eye(*cn_prior.shape)

    cn_prior_full = np.ones((cn_prior.shape[0] + 1, cn_prior.shape[1] + 1)) * cn_amp_prior

    cn_prior_full[0:cn_prior.shape[0], 0:cn_prior.shape[1]] = cn_prior

    return cn_prior_full


def init(
        experiment_filename,
        candidate_h_filename_callback,
        candidate_h_plot_filename,
        segment_allele_count_filename,
        breakpoint_filename,
        num_clones=None,
    ):

    # Prepare experiment file
    experiment = remixt.analysis.experiment.create_experiment(
        segment_allele_count_filename,
        breakpoint_filename,
    )

    with open(experiment_filename, 'w') as f:
        pickle.dump(experiment, f)

    # Prepare candidate haploid depth initializations
    fig = plt.figure(figsize=(8,8))

    ax = plt.subplot(1, 1, 1)

    emission = remixt.likelihood.ReadCountLikelihood()
    emission.estimate_parameters(experiment.x, experiment.l)

    candidate_h_init = remixt.analysis.readdepth.candidate_h(experiment.x, experiment.l, num_clones=num_clones, ax=ax)

    fig.savefig(candidate_h_plot_filename, format='pdf')

    for idx, h in enumerate(candidate_h_init):
        with open(candidate_h_filename_callback(idx), 'w') as f:
            pickle.dump(h, f)


def fit(
        results_filename,
        experiment_filename,
        h_init_filename,
        cn_proportions_filename,
    ):

    with open(experiment_filename, 'r') as f:
        experiment = pickle.load(f)

    with open(h_init_filename, 'r') as f:
        h_init = pickle.load(f)

    N = experiment.l.shape[0]
    M = h_init.shape[0]

    # Create emission / prior / copy number models
    emission = remixt.likelihood.NegBinLikelihood(total_cn=True)
    emission.estimate_parameters(experiment.x, experiment.l)

    cn_probs = create_cn_prior_matrix(cn_proportions_filename)
    prior = remixt.cn_model.CopyNumberPrior(N, M, cn_probs)
    prior.set_lengths(experiment.l)

    model = remixt.cn_model.HiddenMarkovModel(N, M, emission, prior)
    model.set_observed_data(experiment.x, experiment.l)

    # Estimate haploid depths
    estimator = remixt.em.ExpectationMaximizationEstimator()
    h_init, _, _ = estimator.learn_param(model, 'h', h_init)

    # Re-estimate phi
    estimator = remixt.em.ExpectationMaximizationEstimator()
    _, _, phi_converged = estimator.learn_param(model, 'phi', emission.phi)
    phi_em_iter = estimator.em_iter

    # Re-estimate haploid depths
    estimator = remixt.em.ExpectationMaximizationEstimator()
    h, log_posterior, h_converged = estimator.learn_param(model, 'h', h_init)
    h_em_iter = estimator.em_iter

    # Infer copy number
    _, cn_init = model.optimal_state()
    graph = remixt.genome_graph.GenomeGraph(emission, prior, experiment.adjacencies, experiment.breakpoints)
    graph.set_observed_data(experiment.x, experiment.l)
    graph.init_copy_number(cn_init)
    log_posterior_graph, cn = graph.optimize()
    brk_cn = graph.breakpoint_copy_number
    graph_opt_iter = graph.opt_iter
    decreased_log_posterior = graph.decreased_log_posterior

    # Create copy number table
    cn_table = remixt.analysis.experiment.create_cn_table(experiment, emission, cn, h)
    cn_table['log_likelihood'] = emission.log_likelihood(experiment.x, experiment.l, cn)
    cn_table['log_prior'] = prior.log_prior(cn)
    cn_table['major_expected'] = emission.expected_read_count(experiment.l, cn)[:,0]
    cn_table['minor_expected'] = emission.expected_read_count(experiment.l, cn)[:,1]
    cn_table['total_expected'] = emission.expected_read_count(experiment.l, cn)[:,2]
    cn_table['major_residual'] = np.absolute(cn_table['major_readcount'] - cn_table['major_expected'])
    cn_table['minor_residual'] = np.absolute(cn_table['minor_readcount'] - cn_table['minor_expected'])
    cn_table['total_residual'] = np.absolute(cn_table['readcount'] - cn_table['total_expected'])
    cn_table['phi'] = emission.phi

    # Create copy number table
    # Account for both orderings of the two breakends
    column_swap = {
        'n_1':'n_2',
        'ell_1':'ell_2',
        'side_1':'side_2',
        'n_2':'n_1',
        'ell_2':'ell_1',
        'side_2':'side_1', 
    }
    brk_cn_table_1 = brk_cn.merge(experiment.breakpoint_table)
    brk_cn_table_2 = brk_cn.merge(experiment.breakpoint_table.rename(columns=column_swap))
    brk_cn_table = pd.concat([brk_cn_table_1, brk_cn_table_2], ignore_index=True)

    # Create a table of relevant statistics
    stats_table = {
        'log_posterior':log_posterior,
        'log_posterior_graph':log_posterior_graph,
        'phi_converged':phi_converged,
        'h_converged':h_converged,
        'num_clones':len(h),
        'num_segments':len(experiment.x),
        'phi_em_iter':phi_em_iter,
        'h_em_iter':h_em_iter,
        'graph_opt_iter':graph_opt_iter,
        'decreased_log_posterior':decreased_log_posterior,
    }
    stats_table = pd.DataFrame(stats_table, index=[0])

    # Store in hdf5 format
    with pd.HDFStore(results_filename, 'w') as store:
        store['stats'] = stats_table
        store['h_init'] = pd.Series(h_init, index=xrange(len(h)))
        store['h'] = pd.Series(h, index=xrange(len(h)))
        store['cn'] = cn_table
        store['brk_cn'] = brk_cn_table


def collate(collate_filename, breakpoints_filename, experiment_filename, results_filenames):

    with pd.HDFStore(collate_filename, 'w') as collated:
        
        stats_table = list()

        for idx, results_filename in results_filenames.iteritems():

            with pd.HDFStore(results_filename, 'r') as results:
                for key, value in results.iteritems():
                    collated['solutions/{0}/{1}'.format(idx, key)] = results[key]

                stats = results['stats']
                stats['idx'] = idx

                stats_table.append(stats)

        stats_table = pd.concat(stats_table, ignore_index=True)
        stats_table['bic'] = -2. * stats_table['log_posterior'] + stats_table['num_clones'] * np.log(stats_table['num_segments'])
        stats_table.sort('bic', ascending=True, inplace=True)
        stats_table['bic_optimal'] = False
        stats_table.loc[stats_table.index[0], 'bic_optimal'] = True

        collated['stats'] = stats_table

        breakpoints = pd.read_csv(breakpoints_filename, sep='\t')
        collated['breakpoints'] = breakpoints
        
        with open(experiment_filename, 'r') as f:
            experiment = pickle.load(f)

        collated['reference_adjacencies'] = pd.DataFrame(list(experiment.adjacencies), columns=['n_1', 'n_2'])
        collated['breakpoint_adjacencies'] = experiment.breakpoint_table


