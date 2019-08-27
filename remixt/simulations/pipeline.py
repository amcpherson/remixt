import itertools
import pickle
import yaml
import collections
import pandas as pd
import numpy as np

import remixt.segalg
import remixt.simulations.experiment
import remixt.simulations.haplotype
import remixt.simulations.seqread
import remixt.utils
import remixt.cn_plot


def read_sim_defs(sim_defs_filename):

    sim_defs = dict()
    execfile(sim_defs_filename, {}, sim_defs)

    default_settings = sim_defs['defaults']

    settings_dicts = dict()

    for name, settings in sim_defs.items():

        if not name.endswith('_settings'):
            continue

        name = name[:-len('_settings')]

        permute = [zip(itertools.repeat(key), values) for key, values in settings.items()]
        product = itertools.product(*permute)

        # Any settings with tuples of strings as keys represent settings
        # that are tied, multiple values are set for each single simulation
        # and these lists of settings tuples must be unzipped

        def unzip_tied_setting(key, value):
            if isinstance(key, tuple):
                if len(key) != len(value):
                    raise ValueError('incompatible key/value lengths for tied values')
                for k, v in zip(key, value):
                    yield k, v
            else:
                yield key, value

        def unzip_sim(sim):
            return itertools.chain(*[unzip_tied_setting(key, value) for key, value in sim])

        unzipped = list([dict(unzip_sim(a)) for a in product])

        settings_df = pd.DataFrame(unzipped)

        assert not settings_df.isnull().any().any()

        settings_df['name'] = name

        for key, value in default_settings.items():
            if key not in settings_df:
                settings_df[key] = value

        # Sim ID as hash of settings for consistency
        settings_df['sim_hash'] = settings_df.apply(lambda row: abs(hash(frozenset(row.to_dict().items()))), axis=1).astype(str)
        assert not settings_df['sim_hash'].duplicated().any()
        settings_df['sim_id'] = settings_df['name'] + '_' + settings_df['sim_hash']

        for idx, row in settings_df.iterrows():
            settings_dicts[row['sim_id']] = row.to_dict()

    return settings_dicts


def create_simulations(sim_defs_filename, config, ref_data_dir):
    chromosome_lengths = remixt.config.get_chromosome_lengths(config, ref_data_dir)

    sim_defs = yaml.load(open(sim_defs_filename))

    def get_sim_instance_name(sim_name, sim_idx, rep_idx):
        return '{}_{}_{}'.format(sim_name, sim_idx, rep_idx)

    simulations = dict()
    for sim_name, sim_params in sim_defs['simulations'].items():
        num_simulations = sim_params['num_simulations']
        num_replicates = sim_params['num_replicates']
        random_seed = sim_params['random_seed_start']

        for sim_idx in range(num_simulations):
            for rep_idx in range(num_replicates):
                simulations[get_sim_instance_name(sim_name, sim_idx, rep_idx)] = sim_defs['defaults'].copy()
                simulations[get_sim_instance_name(sim_name, sim_idx, rep_idx)]['random_seed'] = random_seed
                random_seed += 1

        for sim_config_name, sim_config_value in sim_params.items():
            if sim_config_name == 'num_simulations':
                continue

            try:
                len(sim_config_value)
            except TypeError:
                sim_config_value = [sim_config_value]

            if len(sim_config_value) == 1:
                sim_config_value = [sim_config_value[0]] * num_simulations

            if len(sim_config_value) != num_simulations:
                raise TypeError('sim config length mismatch for {}, {}'.format(sim_name, sim_config_name))

            for sim_idx, value in enumerate(sim_config_value):
                for rep_idx in range(num_replicates):
                    simulations[get_sim_instance_name(sim_name, sim_idx, rep_idx)][sim_config_name] = value

    for sim_instance_name, sim_params in simulations.items():
        if 'chromosome_lengths' not in sim_params:
            if 'chromosomes' in sim_params:
                chromosomes = sim_params['chromosomes']
            else:
                chromosomes = [str(a) for a in range(1, 23)]
            sim_params['chromosome_lengths'] = {chrom: chromosome_lengths[chrom] for chrom in chromosomes}

        if 'chromosomes' not in sim_params:
            sim_params['chromosomes'] = sim_params['chromosome_lengths'].keys()

    return simulations


def simulate_genome_mixture(mixture_filename, mixture_plot_filename, params):

    history_sampler = remixt.simulations.experiment.RearrangementHistorySampler(params)
    genomes_sampler = remixt.simulations.experiment.GenomeCollectionSampler(history_sampler, params)
    mixture_sampler = remixt.simulations.experiment.GenomeMixtureSampler(params)

    np.random.seed(params['random_seed'])

    genomes = genomes_sampler.sample_genome_collection()
    genome_mixture = mixture_sampler.sample_genome_mixture(genomes)

    with open(mixture_filename, 'w') as mixture_file:
        pickle.dump(genome_mixture, mixture_file)

    remixt.cn_plot.plot_mixture(mixture_plot_filename, mixture_filename)


def simulate_experiment(experiment_filename, experiment_plot_filename, params):

    history_sampler = remixt.simulations.experiment.RearrangementHistorySampler(params)
    genomes_sampler = remixt.simulations.experiment.GenomeCollectionSampler(history_sampler, params)
    mixture_sampler = remixt.simulations.experiment.GenomeMixtureSampler(params)
    experiment_sampler = remixt.simulations.experiment.ExperimentSampler(params)

    np.random.seed(params['random_seed'])

    genomes = genomes_sampler.sample_genome_collection()
    genome_mixture = mixture_sampler.sample_genome_mixture(genomes)
    experiment = experiment_sampler.sample_experiment(genome_mixture)

    with open(experiment_filename, 'w') as experiment_file:
        pickle.dump(experiment, experiment_file)

    remixt.cn_plot.plot_experiment(experiment_plot_filename, experiment_filename)


def simulate_germline_alleles(germline_alleles_filename, params, config, ref_data_dir):

    np.random.seed(params['random_seed'])
    
    with pd.HDFStore(germline_alleles_filename, 'w', complevel=9, complib='zlib') as germline_alleles_store:
        for chromosome in params['chromosomes']:
            alleles_table = remixt.simulations.haplotype.create_sim_alleles(chromosome, config, ref_data_dir)
            germline_alleles_store.put('/chromosome_{}'.format(chromosome), alleles_table, format='table')


def simulate_normal_data(read_data_filename, mixture_filename, germline_alleles_filename, params):

    with open(mixture_filename, 'r') as mixture_file:
        genome_mixture = pickle.load(mixture_file)

    germline_genome = genome_mixture.genome_collection.genomes[0]
    germline_alleles = pd.HDFStore(germline_alleles_filename, 'r')

    np.random.seed(params['random_seed'])

    remixt.simulations.seqread.simulate_mixture_read_data(
        read_data_filename,
        [germline_genome],
        [params['h_total']],
        germline_alleles,
        params)


def resample_normal_data(read_data_filename, source_filename, mixture_filename, germline_alleles_filename, params):

    with open(mixture_filename, 'r') as mixture_file:
        genome_mixture = pickle.load(mixture_file)

    germline_genome = genome_mixture.genome_collection.genomes[0]
    germline_alleles = pd.HDFStore(germline_alleles_filename, 'r')

    np.random.seed(params['random_seed'])

    remixt.simulations.seqread.resample_mixture_read_data(
        read_data_filename,
        source_filename,
        [germline_genome],
        [params['h_total']],
        germline_alleles,
        params)


def simulate_tumour_data(read_data_filename, mixture_filename, germline_alleles_filename, params):

    with open(mixture_filename, 'r') as mixture_file:
        gm = pickle.load(mixture_file)

    germline_alleles = pd.HDFStore(germline_alleles_filename, 'r')

    np.random.seed(params['random_seed'])

    remixt.simulations.seqread.simulate_mixture_read_data(
        read_data_filename,
        gm.genome_collection.genomes,
        gm.frac * params['h_total'],
        germline_alleles,
        params)


def resample_tumour_data(read_data_filename, source_filename, mixture_filename, germline_alleles_filename, params):

    with open(mixture_filename, 'r') as mixture_file:
        gm = pickle.load(mixture_file)

    germline_alleles = pd.HDFStore(germline_alleles_filename, 'r')

    np.random.seed(params['random_seed'])

    remixt.simulations.seqread.resample_mixture_read_data(
        read_data_filename,
        source_filename,
        gm.genome_collection.genomes,
        gm.frac * params['h_total'],
        germline_alleles,
        params)


def tabulate_experiment(exp_table_filename, sim_id, experiment_filename):

    with open(experiment_filename, 'r') as experiment_file:
        exp = pickle.load(experiment_file)

    exp_data = dict()
    exp_data['sim_id'] = sim_id

    exp_data['proportion_divergent'] = exp.genome_mixture.genome_collection.proportion_divergent()

    for idx, proportion_loh in enumerate(exp.genome_mixture.genome_collection.proportion_loh()):
        exp_data['proportion_loh_{0}'.format(idx)] = proportion_loh

    for idx, proportion_hdel in enumerate(exp.genome_mixture.genome_collection.proportion_hdel()):
        exp_data['proportion_hdel_{0}'.format(idx)] = proportion_hdel

    for idx, proportion_hlamp in enumerate(exp.genome_mixture.genome_collection.proportion_hlamp()):
        exp_data['proportion_hlamp_{0}'.format(idx)] = proportion_hlamp

    exp_table = pd.DataFrame([exp_data])

    exp_table.to_csv(exp_table_filename, sep='\t', index=False)


def merge_tables(output_filename, input_filenames):

    output_table = list()

    for input_filename in input_filenames.values():
        output_table.append(pd.read_csv(input_filename, sep='\t', dtype=str))

    output_table = pd.concat(output_table, ignore_index=True)

    output_table.to_csv(output_filename, sep='\t', index=False)


def tabulate_results(results_table_filename, sim_defs, experiment_filenames, results_filenames):

    stats_table = pd.read_csv(stats_table_filename, sep='\t')
    exp_table = pd.read_csv(exp_table_filename, sep='\t')

    results_table = pd.DataFrame(sim_defs.values())

    results_table = results_table.merge(stats_table, on='sim_id')
    results_table = results_table.merge(exp_table, on='sim_id')

    results_table.to_csv(results_table_filename, sep='\t', index=False)


def write_segments(segment_filename, genomes_filename):

    with open(genomes_filename, 'r') as genomes_file:
        gc = pickle.load(genomes_file)

    segment_data = pd.DataFrame({
        'chromosome':gc.segment_chromosome_id,
        'start':gc.segment_start,
        'end':gc.segment_end,
    })

    segment_data.to_csv(segment_filename, sep='\t', index=False, header=True)


def write_perfect_segments(segment_filename, genomes_filename):

    with open(genomes_filename, 'r') as genomes_file:
        gc = pickle.load(genomes_file)

    is_diff_next = (np.abs(np.diff(gc.cn, axis=0)).sum(axis=(1,2)) > 0) * 1
    is_new_seg = np.concatenate(([1], is_diff_next))
    seg_id = is_new_seg.cumsum() - 1

    segment_data = pd.DataFrame({
        'chromosome':gc.segment_chromosome_id,
        'start':gc.segment_start,
        'end':gc.segment_end,
        'seg_id':seg_id,
    })

    segment_data = (
        segment_data
        .groupby(['chromosome', 'seg_id'])
        .agg({'start':np.min, 'end':np.max})
        .reset_index()
        .drop('seg_id', axis=1)
    )

    segment_data.to_csv(segment_filename, sep='\t', index=False, header=True)


def write_breakpoints(breakpoint_filename, mixture_filename):

    with open(mixture_filename, 'r') as mixture_file:
        mixture = pickle.load(mixture_file)

    mixture.breakpoint_segment_data.to_csv(breakpoint_filename, sep='\t', header=True, index=False)


def evaluate_cn_results(genome_mixture, cn_data_table, order_true, order_pred, allow_swap):
    """ Evaluate predicted copy number results against known mixture.

    Args:
        genome_mixture (remixt.simulations.GenomeMixture): known mixture
        cn_data_table (pandas.DataFrame): predicted copy number data
        order_true (numpy.array): indices in order of increasing true clone prevalence
        order_pred (numpy.array): indices in order of increasing predicted clone prevalence
        allow_swap (bool): allow copy number swapping with no penalty

    Returns:
        dict: dictionary of pandas.DataFrame with evaluation statistics

    Predicted copy number data `cn_data_table` should have columns 'chromosome', 'start', 'end',
    in addition to either 'major_m', 'minor_m' for clone 'm', or 'total_m' for allele non-specific
    copy number predictions.

    """
    sim_segments = pd.DataFrame({
        'chromosome':genome_mixture.segment_chromosome_id,
        'start':genome_mixture.segment_start,
        'end':genome_mixture.segment_end,
    })

    if 'major_1' in cn_data_table:
        cn_true = genome_mixture.cn[:,1:,:]

        cn_pred = np.array(
            [
                [cn_data_table['major_1'], cn_data_table['minor_1']],
                [cn_data_table['major_2'], cn_data_table['minor_2']],
            ]
        ).swapaxes(0, 2).swapaxes(1, 2)

    else:
        cn_true = np.zeros((genome_mixture.cn.shape[0], genome_mixture.cn.shape[1]-1, 1))

        cn_true[:,:,0] = genome_mixture.cn[:,1:,:].sum(axis=2)

        cn_pred = np.array(
            [
                [cn_data_table['total_1']],
                [cn_data_table['total_2']],
            ]
        ).swapaxes(0, 2).swapaxes(1, 2)

    # Ensure true tumour clones are consistent, largest first
    cn_true = cn_true[:,order_true,:]

    # Ensure predicted tumour clones are consistent, largest first
    cn_pred = cn_pred[:,order_pred,:]

    # Ensure major minor ordering is consistent
    cn_true = np.sort(cn_true, axis=2)
    cn_pred = np.sort(cn_pred, axis=2)

    cn_data_index = remixt.segalg.reindex_segments(sim_segments, cn_data_table)

    cn_true = cn_true[cn_data_index['idx_1'].values,:,:]
    cn_pred = cn_pred[cn_data_index['idx_2'].values,:,:]
    segment_lengths = (cn_data_index['end'] - cn_data_index['start']).values

    # Handle different number of clones
    if cn_true.shape[1] != cn_pred.shape[1]:
        proportion_cn_correct = -1.

    else:
        # Allow for clone copy number swapping if the mix fractions are close to equal
        if allow_swap:
            cn_correct = (cn_true == cn_pred).all(axis=(1, 2)) | (cn_true == cn_pred[:,::-1,:]).all(axis=(1, 2))
        else:
            cn_correct = (cn_true == cn_pred).all(axis=(1, 2))
        proportion_cn_correct = float((cn_correct * segment_lengths).sum()) / float(segment_lengths.sum())

    is_dom_cn_correct = np.all(cn_true[:,0,:] == cn_pred[:,0,:], axis=1)

    proportion_dom_cn_correct = float((is_dom_cn_correct * segment_lengths).sum()) / float(segment_lengths.sum())

    is_clonal_true = np.all(cn_true[:,0:1,:].swapaxes(1, 2) == cn_true[:,:,:].swapaxes(1, 2), axis=(1, 2))
    is_clonal_pred = np.all(cn_pred[:,0:1,:].swapaxes(1, 2) == cn_pred[:,:,:].swapaxes(1, 2), axis=(1, 2))
    is_clonal_correct = is_clonal_true == is_clonal_pred
    is_subclonal_correct = ~is_clonal_true == ~is_clonal_pred

    proportion_clonal_correct = float((is_clonal_correct * segment_lengths).sum()) / float(segment_lengths.sum())
    proportion_subclonal_correct = float((is_subclonal_correct * segment_lengths).sum()) / float(segment_lengths.sum())

    pred_ploidy = (cn_pred.mean(axis=1) * segment_lengths[:, np.newaxis]).sum() / float(segment_lengths.sum())
    true_ploidy = (cn_true.mean(axis=1) * segment_lengths[:, np.newaxis]).sum() / float(segment_lengths.sum())

    pred_ploidy_1 = (cn_pred[:, 0, :] * segment_lengths[:, np.newaxis]).sum() / float(segment_lengths.sum())
    true_ploidy_1 = (cn_true[:, 0, :] * segment_lengths[:, np.newaxis]).sum() / float(segment_lengths.sum())

    pred_ploidy_2 = (cn_pred[:, 1, :] * segment_lengths[:, np.newaxis]).sum() / float(segment_lengths.sum())
    true_ploidy_2 = (cn_true[:, 1, :] * segment_lengths[:, np.newaxis]).sum() / float(segment_lengths.sum())

    pred_divergent = (cn_pred.max(axis=1) != cn_pred.min(axis=1)) * 1.
    true_divergent = (cn_true.max(axis=1) != cn_true.min(axis=1)) * 1.

    pred_proportion_divergent = (pred_divergent * segment_lengths[:, np.newaxis]).sum() / (2. * segment_lengths.sum())
    true_proportion_divergent = (true_divergent * segment_lengths[:, np.newaxis]).sum() / (2. * segment_lengths.sum())

    evaluation = dict()
    evaluation['proportion_cn_correct'] = proportion_cn_correct
    evaluation['proportion_dom_cn_correct'] = proportion_dom_cn_correct
    evaluation['proportion_clonal_correct'] = proportion_clonal_correct
    evaluation['proportion_subclonal_correct'] = proportion_subclonal_correct
    evaluation['pred_ploidy'] = pred_ploidy
    evaluation['true_ploidy'] = true_ploidy
    evaluation['pred_ploidy_1'] = pred_ploidy_1
    evaluation['true_ploidy_1'] = true_ploidy_1
    evaluation['pred_ploidy_2'] = pred_ploidy_2
    evaluation['true_ploidy_2'] = true_ploidy_2
    evaluation['pred_proportion_divergent'] = pred_proportion_divergent
    evaluation['true_proportion_divergent'] = true_proportion_divergent
    evaluation = pd.Series(evaluation)
    
    results = {
        'cn_evaluation': evaluation,
    }

    return results


def evaluate_brk_cn_results(genome_mixture, brk_cn_table, order_true, order_pred, allow_swap):
    """ Evaluate breakpoint copy number results.

    Args:
        genome_mixture (remixt.simulations.GenomeMixture): known mixture
        brk_cn_table (pandas.DataFrame): predicted copy number table
        order_true (numpy.array): indices in order of increasing true clone prevalence
        order_pred (numpy.array): indices in order of increasing predicted clone prevalence
        allow_swap (bool): allow copy number swapping with no penalty

    Returns:
        dict: dictionary of pandas.DataFrame with evaluation statistics

    Predicted breakpoint copy number data `brk_cn_table` should have columns
    'n_1', 'side_1', 'n_2', 'side_2', in addition to either 'cn_m', for clone 'm'.

    """

    # List of column names for known true copy number
    true_cols = []
    for m in range(1, genome_mixture.M):
        true_cols.append('true_cn_{}'.format(m))

    # List of column names for minimized known true copy number
    min_true_cols = []
    for m in range(1, genome_mixture.M):
        min_true_cols.append('min_true_cn_{}'.format(m))

    # List of column names for predicted copy number
    pred_cols = []
    for m in itertools.count(1):
        if 'cn_{}'.format(m) not in brk_cn_table:
            break
        pred_cols.append('cn_{}'.format(m))

    data = genome_mixture.breakpoint_segment_data.set_index('prediction_id')

    # Add columns for known true copy number
    for col in itertools.chain(true_cols, min_true_cols):
        data[col] = 0

    # Default False for is_balanced
    data['is_balanced'] = False

    # Annotate each breakpoint with its predicted copy number, using
    # breakpoints collapsed by allele
    true_brk_cn = genome_mixture.genome_collection.collapsed_breakpoint_copy_number()
    min_true_brk_cn = genome_mixture.genome_collection.collapsed_minimal_breakpoint_copy_number()
    true_balanced_breakpoints = genome_mixture.genome_collection.collapsed_balanced_breakpoints()

    # Add true copy number and balanced indicator to table
    for prediction_id, breakpoint in genome_mixture.detected_breakpoints.items():
        if breakpoint not in true_brk_cn:
            continue
        data.loc[prediction_id, true_cols] = true_brk_cn[breakpoint][1:]
        data.loc[prediction_id, min_true_cols] = min_true_brk_cn[breakpoint][1:]
        if breakpoint in true_balanced_breakpoints:
            data.loc[prediction_id, 'is_balanced'] = True

    data.reset_index(inplace=True)

    # Merge predicted breakpoint copies
    data = data.merge(brk_cn_table[['prediction_id'] + pred_cols], on='prediction_id', how='left').fillna(0.0)

    # Remove balanced breakpoints
    data = data[~data['is_balanced']]

    # Ensure true tumour clones are consistent
    cn_true = data[min_true_cols].values[:, order_true]

    # Ensure predicted tumour clones are consistent
    cn_pred = data[pred_cols].values[:, order_pred]

    # Handle different number of clones
    if cn_true.shape[1] != cn_pred.shape[1]:
        cn_correct = -1.

    else:
        # Allow for clone copy number swapping if the mix fractions are close to equal
        if allow_swap:
            cn_correct = (cn_true == cn_pred).all(axis=(1,)) | (cn_true == cn_pred[:, ::-1]).all(axis=(1,))
        else:
            cn_correct = (cn_true == cn_pred).all(axis=(1,))

    # Calculate correctness per prediction
    data['cn_correct'] = cn_correct
    data['true_present'] = (data[min_true_cols] > 0).any(axis=1)
    data['pred_present'] = (data[pred_cols] > 0).any(axis=1)
    data['true_subclonal'] = (data[min_true_cols] == 0).any(axis=1) & data['true_present']
    data['pred_subclonal'] = (data[pred_cols] == 0).any(axis=1) & data['pred_present']

    evaluation = dict()
    evaluation['brk_cn_correct_proportion'] = float(data['cn_correct'].sum()) / float(len(data.index))
    evaluation['brk_cn_present_num_true'] = float(data['true_present'].sum())
    evaluation['brk_cn_present_num_pos'] = float(data['pred_present'].sum())
    evaluation['brk_cn_present_num_true_pos'] = float((data['pred_present'] & data['true_present']).sum())
    evaluation['brk_cn_subclonal_num_true'] = float(data['true_subclonal'].sum())
    evaluation['brk_cn_subclonal_num_pos'] = float(data['pred_subclonal'].sum())
    evaluation['brk_cn_subclonal_num_true_pos'] = float((data['pred_subclonal'] & data['true_subclonal']).sum())
    evaluation = pd.Series(evaluation)
    
    results = {
        'brk_cn_table': data,
        'brk_cn_evaluation': evaluation,
    }

    return results


def evaluate_results(genome_mixture, cn_table, brk_cn_table, mix_pred):
    """ Evaluate predicted results against known mixture.

    Args:
        mixture (remixt.simulations.GenomeMixture): known mixture
        cn_table (pandas.DataFrame): predicted copy number data
        brk_cn_table (pandas.DataFrame): predicted copy number table
        mix_pred (numpy.array): predicted mixture

    Returns:
        dict: dictionary of pandas.DataFrame and pandas.Series with evaluation statistics

    Predicted copy number data `cn_table` should have columns 'chromosome', 'start', 'end',
    in addition to either 'major_m', 'minor_m' for clone 'm', or 'total_m' for allele non-specific
    copy number predictions.

    """

    # Return empty evaluation for empty results
    if len(cn_table.index) == 0 or mix_pred.shape[0] == 0:
        empty_results = {
            'brk_cn_evaluation': pd.Series(),
            'brk_cn_table': pd.DataFrame(),
            'cn_evaluation': pd.Series(),
            'mix_results': pd.Series()}

        return empty_results

    cn_table = cn_table.copy()
    brk_cn_table = brk_cn_table.copy()
    mix_true = genome_mixture.frac.copy()

    # Evaluation code assumes 2 tumour clones
    if 'major_1' in cn_table and 'major_2' not in cn_table:
        cn_table['major_2'] = cn_table['major_1']
        cn_table['minor_2'] = cn_table['minor_1']

    if 'total_1' in cn_table and 'total_2' not in cn_table:
        cn_table['total_2'] = cn_table['total_1']

    if 'cn_2' not in brk_cn_table:
        brk_cn_table['cn_2'] = brk_cn_table['cn_1']

    if len(mix_pred) == 2:
        mix_pred = np.concatenate([mix_pred, [0.]])

    assert isinstance(mix_pred, np.ndarray)
    assert isinstance(mix_true, np.ndarray)

    # Ensure true tumour clones are consistent, largest first
    order_true = np.argsort(mix_true[1:])[::-1]
    mix_true[1:] = mix_true[1:][order_true]

    # Ensure predicted tumour clones are consistent, largest first
    order_pred = np.argsort(mix_pred[1:])[::-1]
    mix_pred[1:] = mix_pred[1:][order_pred]

    # Allow for swapping between almost equally prevalent clones
    allow_swap = mix_true[1:].min() / mix_true[1:].max() > 0.75

    results = evaluate_cn_results(genome_mixture, cn_table, order_true, order_pred, allow_swap)

    brk_cn_results = evaluate_brk_cn_results(genome_mixture, brk_cn_table, order_true, order_pred, allow_swap)
    results.update(brk_cn_results)
    
    mix_results = {}
    for idx, f in enumerate(mix_true):
        mix_results['mix_true_'+str(idx)] = f
    for idx, f in enumerate(mix_pred):
        mix_results['mix_pred_'+str(idx)] = f
    results['mix_results'] = pd.Series(mix_results)

    return results


def evaluate_likelihood_results(
    experiment,
    cn_data_table,
):
    """ Experiment specific performance evaluation.

    Args:
        experiment (Experiment): experiment object
        cn_data_table (str): copy number table

    Returns:
        dict: dictionary of pandas.DataFrame and pandas.Series with evaluation statistics
    """

    sim_segments = pd.DataFrame({
        'chromosome': experiment.genome_mixture.segment_chromosome_id,
        'start': experiment.genome_mixture.segment_start,
        'end': experiment.genome_mixture.segment_end,
    })

    cn_data_index = remixt.segalg.reindex_segments(sim_segments, cn_data_table)

    is_outlier_total_pred = cn_data_table['prob_is_outlier_total'] > 0.5
    is_outlier_allele_pred = cn_data_table['prob_is_outlier_allele'] > 0.5

    is_outlier_total_true = experiment.is_outlier_total[cn_data_index['idx_1'].values]
    is_outlier_allele_true = experiment.is_outlier_allele[cn_data_index['idx_1'].values]

    is_outlier_total_pred = is_outlier_total_pred[cn_data_index['idx_2'].values]
    is_outlier_allele_pred = is_outlier_allele_pred[cn_data_index['idx_2'].values]

    is_outlier_total_correct = is_outlier_total_true == is_outlier_total_pred
    is_outlier_allele_correct = is_outlier_allele_true == is_outlier_allele_pred

    segment_lengths = (cn_data_index['end'] - cn_data_index['start']).values

    evaluation = {}
    evaluation['correct_outlier_total_proportion'] = (is_outlier_total_correct * segment_lengths).sum() / float(segment_lengths.sum())
    evaluation['correct_outlier_allele_proportion'] = (is_outlier_allele_correct * segment_lengths).sum() / float(segment_lengths.sum())
    evaluation = pd.Series(evaluation)

    return {'outlier_evaluation': evaluation}


def evaluate_results_task(
    evaluation_filename,
    results_filename,
    mixture_filename=None,
    experiment_filename=None,
    key_prefix='',
):
    """ Evaluate results task.

    Args:
        evaluation_filename (str): output hdf filename of evaluation
        results_filename (str): input prediction results hdf5 filename

    KwArgs:
        mixture_filename (str): input genome mixture pickle filename
        experiment_filename (str): input experiment pickle filename

    """

    with pd.HDFStore(results_filename, 'r') as store:
        cn_table = store[key_prefix + '/cn']
        brk_cn_table = pd.DataFrame(columns=['prediction_id', 'cn_1', 'cn_2'])
        if key_prefix + '/brk_cn' in store:
            brk_cn_table = store[key_prefix + '/brk_cn']
        mix_pred = store[key_prefix + '/mix'].values

    if mixture_filename is not None:
        with open(mixture_filename, 'r') as mixture_file:
            mixture = pickle.load(mixture_file)
    elif experiment_filename is not None:
        with open(experiment_filename, 'r') as experiment_file:
            experiment = pickle.load(experiment_file)
        mixture = experiment.genome_mixture
    else:
        raise ValueError('either mixture_filename or experiment_filename must be set')

    evaluation = evaluate_results(mixture, cn_table, brk_cn_table, mix_pred)

    if experiment_filename is not None:
        evaluation.update(evaluate_likelihood_results(experiment, cn_table))

    with pd.HDFStore(evaluation_filename, 'w') as store:
        for key, data in evaluation.items():
            store['/' + key] = data


def merge_evaluations(merged_filename, sim_defs, evaluation_filenames, key_names):
    """ Merge multiple evaluations of prediction results.

    Args:
        merged_filename (str): output hdf filename of merged evaluations
        sim_defs (dict): simulation definitions per simulation
        evaluation_filenames (dict of str): hdf filename of evaluation per simulation per tool

    """
    
    merged_store = pd.HDFStore(merged_filename, 'w')

    sim_defs_table = pd.DataFrame(
        sim_defs.values(),
        index=pd.Index(sim_defs.keys(), name='sim_id'),
    ).reset_index()

    merged_store['/simulations'] = sim_defs_table

    tables = collections.defaultdict(list)
    for key, evaluation_filename in evaluation_filenames.items():
        store = pd.HDFStore(evaluation_filename, 'r')

        if not isinstance(key, tuple):
            key = (key,)

        for table_name in ('/cn_evaluation', '/brk_cn_evaluation', '/mix_results', 'outlier_evaluation'):
            if table_name not in store:
                continue
            table = store[table_name]
            for value, name in zip(key, key_names):
                table[name] = value
            tables[table_name].append(table)

        merged_store['/brk_cn_table/' + '/'.join(key)] = store['/brk_cn_table']
    
    for table_name, table in tables.items():
        merged_store[table_name] = pd.DataFrame(table)


def create_tool_workflow(
    tool_info,
    normal_seqdata_filename,
    tumour_seqdata_filename,
    breakpoints_filename,
    results_filenames,
    raw_data_directory,
    **kwargs
):
    """ Create workflow for a copy number tool.

    Args:
        tool_info (dict): tool information
        normal_seqdata_filename (str): normal seq data in hdf5 format
        tumour_seqdata_filename (str): tumour seq data in hdf5 format
        breakpoints_filename (str): breakpoint table
        results_filenames (str): results table filename
        raw_data_directory (str): directory for tool specific raw data

    """

    workflow_module = __import__(tool_info['workflow']['module'], fromlist=[''])
    workflow_function = getattr(workflow_module, tool_info['workflow']['run_function'])

    config = tool_info['config']

    if 'kwargs' in tool_info:
        kwargs.update(tool_info['kwargs'])

    seqdata_filenames = {
        'normal': normal_seqdata_filename,
        'tumour': tumour_seqdata_filename,
    }

    return workflow_function(
        seqdata_filenames,
        config,
        results_filenames,
        raw_data_directory,
        somatic_breakpoint_file=breakpoints_filename,
        normal_id='normal',
        **kwargs
    )


def run_setup_function(tool_info, databases, **kwargs):
    """ Run the setup function for a specific tool.

    Args:
        tool_info (dict): tool information
        databases (dict): genome database configuration

    """

    workflow_module = __import__(tool_info['workflow']['module'], fromlist=[''])
    workflow_function = getattr(workflow_module, tool_info['workflow']['setup_function'])

    if 'kwargs' in tool_info:
        kwargs.update(tool_info['kwargs'])

    return workflow_function(tool_info['config'], databases, **kwargs)
