import pickle
import itertools
import numpy as np
import pandas as pd

import remixt.config
import remixt.likelihood
import remixt.cn_model
import remixt.em
import remixt.genome_graph
import remixt.analysis.experiment
import remixt.analysis.readdepth


def init(
    init_results_filename,
    experiment_filename,
    config,
):
    min_ploidy = remixt.config.get_param(config, 'min_ploidy')
    max_ploidy = remixt.config.get_param(config, 'max_ploidy')
    tumour_mix_fractions = remixt.config.get_param(config, 'tumour_mix_fractions')
    prior_variances = remixt.config.get_param(config, 'prior_variances')

    with open(experiment_filename, 'r') as f:
        experiment = pickle.load(f)

    # Calculate candidate haploid depths for normal contamination and a single
    # tumour clone based on modes of the minor allele depth
    read_depth = remixt.analysis.readdepth.calculate_depth(experiment)
    minor_modes = remixt.analysis.readdepth.calculate_minor_modes(read_depth)
    init_h_mono = remixt.analysis.readdepth.calculate_candidate_h_monoclonal(minor_modes)

    # Calculate candidate haploid depths for normal contamination and multiple clones
    # Filter candidates with inappropriate ploidy
    init_h_params = []
    ploidy_estimates = []
    for mode_idx, h_mono in enumerate(init_h_mono):
        estimated_ploidy = remixt.analysis.readdepth.estimate_ploidy(h_mono, experiment)
        assert not np.isinf(estimated_ploidy) and not np.isnan(estimated_ploidy)
        ploidy_estimates.append(estimated_ploidy)

        if min_ploidy is not None and estimated_ploidy < min_ploidy:
            continue

        if max_ploidy is not None and estimated_ploidy > max_ploidy:
            continue

        for mix_frac in tumour_mix_fractions:
            params = {
                'mode_idx': mode_idx,
                'h_normal': h_mono[0],
                'h_tumour': h_mono[1],
                'mix_frac': mix_frac,
            }

            init_h_params.append(params)

    # Check if min and max ploidy was too strict
    if len(init_h_params) == 0:
        raise Exception('no valid ploidy estimates in range {}-{}, candidates are {}'.format(
            min_ploidy, max_ploidy, repr(ploidy_estimates)))

    # Attempt several divergence parameters
    init_params = []
    prior_variance_params = [{'prior_variance': w} for w in prior_variances]
    for h_p, w_p in itertools.product(init_h_params, prior_variance_params):
        params = h_p.copy()
        params.update(w_p)
        init_params.append(params)

    with pd.HDFStore(init_results_filename, 'w') as store:
        store['read_depth'] = read_depth
        store['minor_modes'] = pd.Series(minor_modes, index=xrange(len(minor_modes)))

    return dict(enumerate(init_params))


def fit(
    results_filename,
    experiment_filename,
    init_params,
    config,
    ref_data_dir,
):
    normal_contamination = remixt.config.get_param(config, 'normal_contamination')

    max_copy_number = remixt.config.get_param(config, 'max_copy_number')

    with open(experiment_filename, 'r') as f:
        experiment = pickle.load(f)

    h_init = np.array([
        init_params['h_normal'],
        init_params['h_tumour'] * init_params['mix_frac'],
        init_params['h_tumour'] * (1. - init_params['mix_frac']),
    ])

    fit_results = fit_remixt_variational(
        experiment,
        h_init,
        max_copy_number,
        normal_contamination,
        init_params['prior_variance'],
    )

    h = fit_results['h']
    cn = fit_results['cn']

    ploidy = (cn[:,1:,:].mean(axis=1).T * experiment.l).sum() / experiment.l.sum()
    divergent = (cn[:,1:,:].max(axis=1) != cn[:,1:,:].min(axis=1)) * 1.
    proportion_divergent = (divergent.T * experiment.l).sum() / (2. * experiment.l.sum())

    # Create a table of relevant statistics
    fit_results['stats']['num_clones'] = len(h)
    fit_results['stats']['num_segments'] = len(experiment.x)
    fit_results['stats']['ploidy'] = ploidy
    fit_results['stats']['proportion_divergent'] = proportion_divergent
    fit_results['stats']['mode_idx'] = init_params['mode_idx']
    fit_results['stats']['prior_variance'] = init_params['prior_variance']

    # Store in pickle format
    with open(results_filename, 'w') as f:
        pickle.dump(fit_results, f)


def fit_remixt_variational(
    experiment,
    h_init,
    max_copy_number,
    normal_contamination,
    prior_variance,
):
    results = dict()

    model = remixt.cn_model.BreakpointModel(
        experiment.x,
        experiment.l,
        experiment.adjacencies,
        experiment.breakpoints,
        max_copy_number=max_copy_number,
        normal_contamination=normal_contamination,
        prior_variance=prior_variance,
    )

    elbo = model.optimize(h_init)

    h = model.h
    cn = model.optimal_cn()
    brk_cn = model.optimal_brk_cn()

    results['h'] = h
    results['cn'] = cn
    results['brk_cn'] = brk_cn

    # Save estimation statistics
    results['stats'] = dict()
    results['stats']['elbo'] = elbo
    results['stats']['converged'] = model.converged
    results['stats']['num_iter'] = model.num_iter
    results['stats']['error_message'] = ''

    return results


def fit_hmm_viterbi(
    experiment,
    emission,
    prior,
    max_copy_number,
    normal_contamination,
    likelihood_params,
):
    N = experiment.l.shape[0]
    M = emission.h.shape[0]

    results = dict()
    results['stats'] = dict()

    model = remixt.cn_model.HiddenMarkovModel(
        N, M, emission, prior, experiment.chains,
        max_copy_number=max_copy_number,
        normal_contamination=normal_contamination,
    )

    # Estimate haploid depths and overdispersion parameters
    estimator = remixt.em.ExpectationMaximizationEstimator()
    log_likelihood = estimator.learn_param(model, *likelihood_params)

    # Save estimation statistics
    results['stats']['h_log_likelihood'] = log_likelihood
    results['stats']['h_converged'] = estimator.converged
    results['stats']['h_em_iter'] = estimator.em_iter
    results['stats']['h_error_message'] = estimator.error_message

    # Infer copy number from viterbi
    log_likelihood_viterbi, cn = model.optimal_state()

    # Naive breakpoint copy number
    brk_cn = remixt.cn_model.decode_breakpoints_naive(cn, experiment.adjacencies, experiment.breakpoints)

    # Infer copy number
    results['cn'] = cn
    results['brk_cn'] = brk_cn
    results['stats']['viterbi_log_likelihood'] = log_likelihood_viterbi
    results['stats']['log_likelihood'] = log_likelihood_viterbi
    results['stats']['log_prior'] = prior.log_prior(cn).sum()

    return results


def fit_hmm_graph(
    experiment,
    emission,
    prior,
    max_copy_number,
    normal_contamination,
    likelihood_params,
):
    N = experiment.l.shape[0]
    M = emission.h.shape[0]

    results = dict()
    results['stats'] = dict()

    model = remixt.cn_model.HiddenMarkovModel(
        N, M, emission, prior, experiment.chains,
        max_copy_number=max_copy_number,
        normal_contamination=normal_contamination,
    )

    # Estimate haploid depths
    estimator = remixt.em.ExpectationMaximizationEstimator()
    log_likelihood = estimator.learn_param(model, *likelihood_params)

    # Save estimation statistics
    results['stats']['h_log_likelihood'] = log_likelihood
    results['stats']['h_converged'] = estimator.converged
    results['stats']['h_em_iter'] = estimator.em_iter
    results['stats']['h_error_message'] = estimator.error_message

    # Infer copy number from viterbi
    _, cn_init = model.optimal_state()

    # Create genome graph initializing from viterbi
    graph = remixt.genome_graph.GenomeGraphModel(N, M, emission, prior, experiment.adjacencies, experiment.breakpoints)
    graph.init_copy_number(cn_init)

    # Infer copy number
    log_likelihood_graph = graph.optimize()
    cn = graph.segment_cn

    results['cn'] = cn
    results['brk_cn'] = graph.breakpoint_copy_number
    results['stats']['graph_opt_iter'] = graph.opt_iter
    results['stats']['graph_log_likelihood'] = log_likelihood_graph
    results['stats']['graph_decreased_log_prob'] = graph.decreased_log_prob
    results['stats']['log_likelihood'] = log_likelihood_graph
    results['stats']['log_prior'] = prior.log_prior(cn).sum()

    return results


def fit_graph(
    experiment,
    emission,
    prior,
    max_copy_number,
    normal_contamination,
    likelihood_params,
):
    N = experiment.l.shape[0]
    M = emission.h.shape[0]

    results = dict()
    results['stats'] = dict()

    # Save initial h in preparation for single clone viterbi
    h_init = emission.h

    # Infer initial copy number from viterbi with 1 tumour clone
    h_init_single = np.zeros((2,))
    h_init_single[0] = h_init[0]
    h_init_single[1] = h_init[1:].sum()
    emission.h = h_init_single
    model = remixt.cn_model.HiddenMarkovModel(
        N, 2, emission, prior, experiment.chains,
        max_copy_number=max_copy_number,
        normal_contamination=normal_contamination,
    )
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
    log_likelihood = estimator.learn_param(graph, *likelihood_params)

    # Save copy number and estimation statistics
    results['cn'] = graph.cn
    results['brk_cn'] = graph.breakpoint_copy_number
    results['stats']['h_em_iter'] = estimator.em_iter
    results['stats']['h_converged'] = estimator.converged
    results['stats']['graph_opt_iter'] = graph.opt_iter
    results['stats']['graph_log_likelihood'] = log_likelihood
    results['stats']['graph_decreased_log_posterior'] = graph.decreased_log_posterior
    results['stats']['log_likelihood'] = log_likelihood
    results['stats']['log_prior'] = prior.log_prior(graph.cn).sum()

    return results


fit_methods = {
    'hmm_viterbi': fit_hmm_viterbi,
    'hmm_graph': fit_hmm_graph,
    'graph': fit_graph,
}


def store_fit_results(store, experiment, fit_results, key_prefix):

    h = fit_results['h']
    cn = fit_results['cn']
    brk_cn = fit_results['brk_cn']

    # Create copy number table
    cn_table = remixt.analysis.experiment.create_cn_table(experiment, cn, h)
    # cn_table['log_likelihood'] = emission.log_likelihood(cn)
    # cn_table['log_prior'] = prior.log_prior(cn)
    # cn_table['major_expected'] = emission.expected_read_count(experiment.l, cn)[:,0]
    # cn_table['minor_expected'] = emission.expected_read_count(experiment.l, cn)[:,1]
    # cn_table['total_expected'] = emission.expected_read_count(experiment.l, cn)[:,2]
    # cn_table['ratio_expected'] = emission.expected_allele_ratio(cn)
    # cn_table['major_residual'] = np.absolute(cn_table['major_readcount'] - cn_table['major_expected'])
    # cn_table['minor_residual'] = np.absolute(cn_table['minor_readcount'] - cn_table['minor_expected'])
    # cn_table['total_residual'] = np.absolute(cn_table['readcount'] - cn_table['total_expected'])

    brk_cn_table = remixt.analysis.experiment.create_brk_cn_table(experiment, brk_cn)

    store[key_prefix + '/h'] = pd.Series(h, index=xrange(len(h)))
    store[key_prefix + '/cn'] = cn_table
    store[key_prefix + '/mix'] = pd.Series(h / h.sum(), index=xrange(len(h)))
    store[key_prefix + '/brk_cn'] = brk_cn_table


def store_optimal_solution(stats, store, config):
    max_prop_diverge = remixt.config.get_param(config, 'max_prop_diverge')

    stats = stats[stats['proportion_divergent'] < max_prop_diverge].copy()
    stats.sort_values('elbo', ascending=False, inplace=True)
    solution_idx = stats.loc[stats.index[0], 'init_id']

    key_prefix = '/solutions/solution_{}'.format(solution_idx)
    store['/cn'] = store[key_prefix + '/cn']
    store['/mix'] = store[key_prefix + '/mix']
    store['/brk_cn'] = store[key_prefix + '/brk_cn']


def collate(collate_filename, experiment_filename, init_results_filename, fit_results_filenames, config):

    # Extract the statistics for selecting solutions
    stats_table = list()
    for init_id, results_filename in fit_results_filenames.iteritems():
        with open(results_filename, 'r') as f:
            results = pickle.load(f)
            stats = results['stats']
            stats['init_id'] = init_id
            stats_table.append(stats)
    stats_table = pd.DataFrame(stats_table)

    # Write out selected solutions
    with pd.HDFStore(collate_filename, 'w') as collated:
        collated['stats'] = stats_table

        with pd.HDFStore(init_results_filename, 'r') as results:
            for key, value in results.iteritems():
                collated[key] = results[key]

        with open(experiment_filename, 'r') as f:
            experiment = pickle.load(f)

        for init_id, results_filename in fit_results_filenames.iteritems():
            with open(results_filename, 'r') as f:
                results = pickle.load(f)
                store_fit_results(collated, experiment, results, 'solutions/solution_{0}'.format(init_id))

        store_optimal_solution(stats_table, collated, config)

