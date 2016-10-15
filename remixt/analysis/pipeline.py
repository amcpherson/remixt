import pickle
import itertools
import numpy as np
import pandas as pd

import remixt.config
import remixt.cn_model
import remixt.em
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
    divergence_weights = remixt.config.get_param(config, 'divergence_weights')

    with open(experiment_filename, 'r') as f:
        experiment = pickle.load(f)

    # Calculate candidate haploid depths for normal contamination and a single
    # tumour clone based on modes of the minor allele depth
    read_depth = remixt.analysis.readdepth.calculate_depth(experiment)
    minor_modes = remixt.analysis.readdepth.calculate_minor_modes(read_depth)
    init_h_mono = remixt.analysis.readdepth.calculate_candidate_h_monoclonal(minor_modes)

    # Calculate candidate haploid depths for normal contamination and multiple clones
    init_h_params = []
    ploidy_estimates = []
    for mode_idx, h_mono in enumerate(init_h_mono):
        estimated_ploidy = remixt.analysis.readdepth.estimate_ploidy(h_mono, experiment)
        assert not np.isinf(estimated_ploidy) and not np.isnan(estimated_ploidy)

        for mix_frac in tumour_mix_fractions:
            params = {
                'mode_idx': mode_idx,
                'h_normal': h_mono[0],
                'h_tumour': h_mono[1],
                'mix_frac': mix_frac,
            }

            init_h_params.append(params)
            ploidy_estimates.append(estimated_ploidy)

    # Filter candidates with unacceptable ploidy
    def ploidy_filter_dist(ploidy):
        if min_ploidy is not None and ploidy < min_ploidy:
            return min_ploidy - ploidy

        if max_ploidy is not None and ploidy > max_ploidy:
            return ploidy - max_ploidy
            
        return 0.

    def ploidy_filter(ploidy):
        return ploidy_filter_dist(ploidy) == 0.

    is_ploidy_filtered = [ploidy_filter(a) for a in ploidy_estimates]
    
    # Check if min and max ploidy was too strict and select single closest ploidy
    if not any(is_ploidy_filtered):
        ploidy_dists = [ploidy_filter_dist(a) for a in ploidy_estimates]
        is_ploidy_filtered = [a == min(ploidy_dists) for a in ploidy_dists]
        
    # Filter ploidy and h initializations
    init_h_params = [a for i, a in enumerate(init_h_params) if is_ploidy_filtered[i]]
    ploidy_estimates = [a for i, a in enumerate(ploidy_estimates) if is_ploidy_filtered[i]]

    # Attempt several divergence parameters
    init_params = []
    divergence_weight_params = [{'divergence_weight': w} for w in divergence_weights]
    for h_p, w_p in itertools.product(init_h_params, divergence_weight_params):
        params = h_p.copy()
        params.update(w_p)
        init_params.append(params)

    with pd.HDFStore(init_results_filename, 'w') as store:
        store['read_depth'] = read_depth
        store['minor_modes'] = pd.Series(minor_modes, index=xrange(len(minor_modes)))

    return dict(enumerate(init_params))


def fit_task(
    results_filename,
    experiment_filename,
    init_params,
    config,
):
    with open(experiment_filename, 'r') as f:
        experiment = pickle.load(f)
        
    fit_results = fit(experiment, init_params, config)
    
    with open(results_filename, 'w') as f:
        pickle.dump(fit_results, f)


def fit(experiment, init_params, config):
    h_init = np.array([
        init_params['h_normal'],
        init_params['h_tumour'] * init_params['mix_frac'],
        init_params['h_tumour'] * (1. - init_params['mix_frac']),
    ])
    divergence_weight = init_params['divergence_weight']

    normal_contamination = remixt.config.get_param(config, 'normal_contamination')
    max_copy_number = remixt.config.get_param(config, 'max_copy_number')
    min_segment_length = remixt.config.get_param(config, 'likelihood_min_segment_length')
    min_proportion_genotyped = remixt.config.get_param(config, 'likelihood_min_proportion_genotyped')
    num_em_iter = remixt.config.get_param(config, 'num_em_iter')
    num_update_iter = remixt.config.get_param(config, 'num_update_iter')

    model = remixt.cn_model.BreakpointModel(
        h_init,
        experiment.x,
        experiment.l,
        experiment.adjacencies,
        experiment.breakpoints,
        max_copy_number=max_copy_number,
        normal_contamination=normal_contamination,
        divergence_weight=divergence_weight,
        min_segment_length=min_segment_length,
        min_proportion_genotyped=min_proportion_genotyped,
    )
    model.num_update_iter = num_update_iter

    # Estimate haploid depths and other likelihood parameters
    estimator = remixt.em.ExpectationMaximizationEstimator()
    estimator.num_em_iter = num_em_iter
    elbo = estimator.learn_param(model, *model.likelihood_params)

    fit_results = dict()

    fit_results['h'] = model.h
    fit_results['cn'] = model.optimal_cn()
    fit_results['brk_cn'] = model.optimal_brk_cn()

    # Save estimation statistics
    fit_results['stats'] = dict()
    fit_results['stats']['elbo'] = elbo
    fit_results['stats']['elbo_diff'] = model.prev_elbo_diff
    fit_results['stats']['error_message'] = ''

    cn = fit_results['cn']
    ploidy = (cn[:,1:,:].mean(axis=1).T * experiment.l).sum() / experiment.l.sum()
    divergent = (cn[:,1:,:].max(axis=1) != cn[:,1:,:].min(axis=1)) * 1.
    proportion_divergent = (divergent.T * experiment.l).sum() / (2. * experiment.l.sum())

    # Create a table of relevant statistics
    fit_results['stats']['num_clones'] = len(model.h)
    fit_results['stats']['num_segments'] = len(experiment.x)
    fit_results['stats']['ploidy'] = ploidy
    fit_results['stats']['proportion_divergent'] = proportion_divergent
    fit_results['stats']['mode_idx'] = init_params['mode_idx']
    fit_results['stats']['divergence_weight'] = init_params['divergence_weight']

    return fit_results
    

def store_fit_results(store, experiment, fit_results, key_prefix):
    h = fit_results['h']
    cn = fit_results['cn']
    brk_cn = fit_results['brk_cn']

    # Create copy number table
    cn_table = remixt.analysis.experiment.create_cn_table(experiment, cn, h)

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
        results = pickle.load(open(results_filename))
        stats = dict(results['stats'])
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
            results = pickle.load(open(results_filename))
            store_fit_results(collated, experiment, results, 'solutions/solution_{0}'.format(init_id))

        store_optimal_solution(stats_table, collated, config)
