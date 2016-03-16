import pickle
import pkg_resources
import numpy as np
import pandas as pd

import remixt.config
import remixt.likelihood
import remixt.cn_model
import remixt.em
import remixt.genome_graph
import remixt.analysis.experiment
import remixt.analysis.readdepth

default_cn_proportions_filename = pkg_resources.resource_filename('remixt', 'data/cn_proportions.tsv')


def create_cn_prior_matrix(cn_proportions_filename=None):
    """ Create a matrix of prior probabilities for copy number states.

    KwArgs:
        cn_proportions_filename (str): tsv table of proportions of each state

    Returns:
        numpy.array: copy number prior matrix

    """

    if cn_proportions_filename is None:
        cn_proportions_filename = default_cn_proportions_filename

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
    candidate_h_filenames,
    init_results_filename,
    experiment_filename,
    config,
):
    num_clones = remixt.config.get_param(config, 'num_clones')

    with open(experiment_filename, 'r') as f:
        experiment = pickle.load(f)

    read_depth = remixt.analysis.readdepth.calculate_depth(experiment)
    minor_modes = remixt.analysis.readdepth.calculate_minor_modes(read_depth)
    candidate_h = remixt.analysis.readdepth.calculate_candidate_h(minor_modes, num_clones=num_clones)
    candidate_h = remixt.analysis.readdepth.filter_high_ploidy(candidate_h, experiment, max_ploidy=5.0)

    for idx, h in enumerate(candidate_h):
        with open(candidate_h_filenames[idx], 'w') as f:
            pickle.dump(h, f)

    with pd.HDFStore(init_results_filename, 'w') as store:
        store['read_depth'] = read_depth
        store['minor_modes'] = pd.Series(minor_modes, index=xrange(len(minor_modes)))


def fit_hmm_viterbi(experiment, emission, prior, h_init, normal_contamination):
    N = experiment.l.shape[0]
    M = h_init.shape[0]

    results = dict()
    results['stats'] = dict()

    # Initialize haploid depths
    emission.h = h_init

    model = remixt.cn_model.HiddenMarkovModel(N, M, emission, prior, experiment.chains, normal_contamination=normal_contamination)

    # Estimate haploid depths and overdispersion parameters
    estimator = remixt.em.ExpectationMaximizationEstimator()
    log_posterior = estimator.learn_param(
        model,
        emission.h_param,
        emission.r_param,
        emission.M_param,
        emission.z_param,
        emission.hdel_mu_param,
        emission.loh_p_param,
    )
    results['h'] = emission.h
    results['r'] = emission.r
    results['M'] = emission.M
    results['stats']['h_log_posterior'] = log_posterior
    results['stats']['h_converged'] = estimator.converged
    results['stats']['h_em_iter'] = estimator.em_iter
    results['stats']['h_error_message'] = estimator.error_message

    # Infer copy number from viterbi
    log_posterior_viterbi, cn = model.optimal_state()

    # Naive breakpoint copy number
    brk_cn = remixt.cn_model.decode_breakpoints_naive(cn, experiment.adjacencies, experiment.breakpoints)

    # Infer copy number
    results['cn'] = cn
    results['brk_cn'] = brk_cn
    results['stats']['viterbi_log_posterior'] = log_posterior_viterbi
    results['stats']['log_posterior'] = log_posterior_viterbi
    results['stats']['log_prior'] = prior.log_prior(cn).sum()

    return results


def fit_hmm_graph(experiment, emission, prior, h_init, normal_contamination):
    N = experiment.l.shape[0]
    M = h_init.shape[0]

    results = dict()
    results['stats'] = dict()

    # Initialize haploid depths
    emission.h = h_init

    model = remixt.cn_model.HiddenMarkovModel(N, M, emission, prior, experiment.chains, normal_contamination=normal_contamination)

    # Estimate haploid depths
    estimator = remixt.em.ExpectationMaximizationEstimator()
    log_posterior = estimator.learn_param(
        model,
        emission.h_param,
        emission.r_param,
        emission.M_param,
        emission.z_param,
        emission.hdel_mu_param,
        emission.loh_p_param,
    )
    results['h'] = emission.h
    results['r'] = emission.r
    results['M'] = emission.M
    results['stats']['h_log_posterior'] = log_posterior
    results['stats']['h_converged'] = estimator.converged
    results['stats']['h_em_iter'] = estimator.em_iter
    results['stats']['h_error_message'] = estimator.error_message

    # Set to allele independent prior as allele dependence will
    # cause the genome graph algorithm to fail
    prior.allele_specific = False

    # Infer copy number from viterbi
    _, cn_init = model.optimal_state()

    # Create genome graph initializing from viterbi
    graph = remixt.genome_graph.GenomeGraph(emission, prior, experiment.adjacencies, experiment.breakpoints)
    graph.set_observed_data(experiment.x, experiment.l)
    graph.init_copy_number(cn_init)

    # Infer copy number
    log_posterior_graph, cn = graph.optimize()
    results['cn'] = cn
    results['brk_cn'] = graph.breakpoint_copy_number
    results['stats']['graph_opt_iter'] = graph.opt_iter
    results['stats']['graph_log_posterior'] = log_posterior_graph
    results['stats']['graph_decreased_log_posterior'] = graph.decreased_log_posterior
    results['stats']['log_posterior'] = log_posterior_graph
    results['stats']['log_prior'] = prior.log_prior(cn).sum()

    return results


def fit_graph(experiment, emission, prior, h_init, normal_contamination):
    N = experiment.l.shape[0]
    M = h_init.shape[0]

    results = dict()
    results['stats'] = dict()

    # Infer initial copy number from viterbi with 1 tumour clone
    h_init_single = np.zeros((2,))
    h_init_single[0] = h_init[0]
    h_init_single[1] = h_init[1:].sum()
    emission.h = h_init_single
    model = remixt.cn_model.HiddenMarkovModel(N, 2, emission, prior, experiment.chains, normal_contamination=normal_contamination)
    _, cn = model.optimal_state()
    cn_init = np.ones((N, M, 2))
    for m in xrange(1, M):
        cn_init[:,m,:] = cn[:,1,:]

    # Initialize haploid depths
    emission.h = h_init

    # Create genome graph
    graph = remixt.genome_graph.GenomeGraph(emission, prior, experiment.adjacencies, experiment.breakpoints)
    graph.set_observed_data(experiment.x, experiment.l)
    graph.init_copy_number(cn_init)

    # Estimate haploid depths and copy number
    estimator = remixt.em.HardAssignmentEstimator()
    h, log_posterior, h_converged = estimator.learn_param(graph, 'h', h_init)

    results['h'] = h
    results['cn'] = graph.cn
    results['brk_cn'] = graph.breakpoint_copy_number
    results['stats']['h_em_iter'] = estimator.em_iter
    results['stats']['graph_opt_iter'] = graph.opt_iter
    results['stats']['graph_log_posterior'] = log_posterior
    results['stats']['graph_decreased_log_posterior'] = graph.decreased_log_posterior
    results['stats']['log_posterior'] = log_posterior
    results['stats']['log_prior'] = prior.log_prior(graph.cn).sum()

    return results


fit_methods = [
    'hmm_viterbi',
    'hmm_graph',
    'graph',
]


def fit(
    results_filename,
    experiment_filename,
    h_init_filename,
    config,
    ref_data_dir,
):
    fit_method = remixt.config.get_param(config, 'fit_method')
    normal_contamination = remixt.config.get_param(config, 'normal_contamination')
    cn_proportions_filename = remixt.config.get_filename(config, ref_data_dir, 'cn_proportions')

    with open(experiment_filename, 'r') as f:
        experiment = pickle.load(f)

    with open(h_init_filename, 'r') as f:
        h_init = pickle.load(f)

    # Create emission / prior / copy number models
    emission = remixt.likelihood.NegBinBetaBinLikelihood(experiment.x, experiment.l)
    emission.h = h_init

    # Create prior probability model
    cn_probs = create_cn_prior_matrix(cn_proportions_filename)
    prior = remixt.cn_model.CopyNumberPrior(cn_probs)
    prior.set_lengths(experiment.l)

    # Mask amplifications from likelihood
    emission.add_amplification_mask(experiment.x, experiment.l, prior.cn_max)

    if fit_method == 'hmm_viterbi':
        fit_results = fit_hmm_viterbi(experiment, emission, prior, h_init, normal_contamination)
    elif fit_method == 'hmm_graph':
        fit_results = fit_hmm_graph(experiment, emission, prior, h_init, normal_contamination)
    elif fit_method == 'graph':
        fit_results = fit_graph(experiment, emission, prior, h_init, normal_contamination)
    else:
        raise ValueError('unknown fit method {}'.format(fit_method))

    h = fit_results['h']
    cn = fit_results['cn']
    brk_cn = fit_results['brk_cn']

    # Create copy number table
    cn_table = experiment.create_cn_table(emission, cn, h)
    cn_table['log_likelihood'] = emission.log_likelihood(cn)
    cn_table['log_prior'] = prior.log_prior(cn)
    cn_table['major_expected'] = emission.expected_read_count(experiment.l, cn)[:,0]
    cn_table['minor_expected'] = emission.expected_read_count(experiment.l, cn)[:,1]
    cn_table['total_expected'] = emission.expected_read_count(experiment.l, cn)[:,2]
    cn_table['ratio_expected'] = emission.expected_allele_ratio(cn)
    cn_table['major_residual'] = np.absolute(cn_table['major_readcount'] - cn_table['major_expected'])
    cn_table['minor_residual'] = np.absolute(cn_table['minor_readcount'] - cn_table['minor_expected'])
    cn_table['total_residual'] = np.absolute(cn_table['readcount'] - cn_table['total_expected'])

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
    brk_cn_table_1 = brk_cn.merge(experiment.breakpoint_segment_data)
    brk_cn_table_2 = brk_cn.merge(experiment.breakpoint_segment_data.rename(columns=column_swap))
    brk_cn_table = pd.concat([brk_cn_table_1, brk_cn_table_2], ignore_index=True)

    ploidy = (cn[:,1:,:].mean(axis=1).T * experiment.l).sum() / experiment.l.sum()
    divergent = (cn[:,1:,:].max(axis=1) != cn[:,1:,:].min(axis=1)) * 1.
    proportion_divergent = (divergent.T * experiment.l).sum() / (2. * experiment.l.sum())

    # Create a table of relevant statistics
    stats_table = fit_results['stats'].copy()
    stats_table['num_clones'] = len(h),
    stats_table['num_segments'] = len(experiment.x),
    stats_table['ploidy'] = ploidy
    stats_table['proportion_divergent'] = proportion_divergent
    stats_table = pd.DataFrame(stats_table, index=[0])

    # Store in hdf5 format
    with pd.HDFStore(results_filename, 'w') as store:
        store['stats'] = stats_table
        store['h_init'] = pd.Series(h_init, index=xrange(len(h)))
        store['h'] = pd.Series(h, index=xrange(len(h)))
        store['cn'] = cn_table
        store['brk_cn'] = brk_cn_table
        store['negbin_r'] = pd.Series(emission.r, index=xrange(len(emission.r)))
        store['betabin_M'] = pd.Series(emission.M, index=xrange(len(emission.M)))


def collate(collate_filename, experiment_filename, init_results_filename, fit_results_filenames):

    with pd.HDFStore(collate_filename, 'w') as collated:

        with pd.HDFStore(init_results_filename, 'r') as results:
            for key, value in results.iteritems():
                collated[key] = results[key]

        stats_table = list()

        for idx, results_filename in fit_results_filenames.iteritems():

            with pd.HDFStore(results_filename, 'r') as results:
                for key, value in results.iteritems():
                    collated['solutions/solution_{0}/{1}'.format(idx, key)] = results[key]

                stats = results['stats']
                stats['idx'] = idx

                stats_table.append(stats)

        stats_table = pd.concat(stats_table, ignore_index=True)
        stats_table['bic'] = -2. * stats_table['log_posterior'] + stats_table['num_clones'] * np.log(stats_table['num_segments'])
        stats_table.sort('bic', ascending=True, inplace=True)
        stats_table['bic_optimal'] = False
        stats_table.loc[stats_table.index[0], 'bic_optimal'] = True

        collated['stats'] = stats_table

        with open(experiment_filename, 'r') as f:
            experiment = pickle.load(f)

        collated['breakpoints'] = experiment.breakpoint_data
        collated['reference_adjacencies'] = pd.DataFrame(list(experiment.adjacencies), columns=['n_1', 'n_2'])
        collated['breakpoint_adjacencies'] = experiment.breakpoint_segment_data


