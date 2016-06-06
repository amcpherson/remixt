import itertools
import pickle
import yaml
import pandas as pd
import numpy as np

import remixt.segalg
import remixt.simulations.experiment
import remixt.simulations.haplotype
import remixt.simulations.seqread
import remixt.utils


def read_sim_defs(sim_defs_filename):

    sim_defs = dict()
    execfile(sim_defs_filename, {}, sim_defs)

    default_settings = sim_defs['defaults']

    settings_dicts = dict()

    for name, settings in sim_defs.iteritems():

        if not name.endswith('_settings'):
            continue

        name = name[:-len('_settings')]

        permute = [zip(itertools.repeat(key), values) for key, values in settings.iteritems()]
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

        for key, value in default_settings.iteritems():
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
    genome_fai = remixt.config.get_filename(config, ref_data_dir, 'genome_fai')
    chromosome_lengths = remixt.utils.read_chromosome_lengths(genome_fai)

    sim_defs = yaml.load(open(sim_defs_filename))

    def get_sim_instance_name(sim_name, sim_idx, rep_idx):
        return '{}_{}_{}'.format(sim_name, sim_idx, rep_idx)

    simulations = dict()
    for sim_name, sim_params in sim_defs['simulations'].iteritems():
        num_simulations = sim_params['num_simulations']
        num_replicates = sim_params['num_replicates']
        random_seed = sim_params['random_seed_start']

        for sim_idx in xrange(num_simulations):
            for rep_idx in xrange(num_replicates):
                simulations[get_sim_instance_name(sim_name, sim_idx, rep_idx)] = sim_defs['defaults']
                simulations[get_sim_instance_name(sim_name, sim_idx, rep_idx)]['random_seed'] = random_seed
                random_seed += 1

        for sim_config_name, sim_config_value in sim_params.iteritems():
            if sim_config_name == 'num_simulations':
                continue

            try:
                len(sim_config_value)
            except TypeError:
                sim_config_value = [sim_config_value] * num_simulations

            if len(sim_config_value) != num_simulations:
                raise TypeError('sim config length mismatch for {}, {}'.format(sim_name, sim_config_name))

            for sim_idx, value in enumerate(sim_config_value):
                for rep_idx in xrange(num_replicates):
                    simulations[get_sim_instance_name(sim_name, sim_idx, rep_idx)][sim_config_name] = value

    for sim_instance_name, sim_params in simulations.iteritems():
        if 'chromosome_lengths' not in sim_params:
            if 'chromosome' in sim_params:
                chromosomes = sim_params['chromosomes']
            else:
                chromosomes = [str(a) for a in range(1, 23)]
            sim_params['chromosome_lengths'] = {chrom: chromosome_lengths[chrom] for chrom in chromosomes}

        if 'chromosomes' not in sim_params:
            sim_params['chromosomes'] = sim_params['chromosome_lengths'].keys()

    return simulations


def simulate_genomes(genomes_filename, params):

    rh_sampler = remixt.simulations.experiment.RearrangementHistorySampler(params)
    gc_sampler = remixt.simulations.experiment.GenomeCollectionSampler(rh_sampler, params)

    np.random.seed(params['random_seed'])

    gc = gc_sampler.sample_genome_collection()

    with open(genomes_filename, 'w') as genomes_file:
        pickle.dump(gc, genomes_file)


def simulate_mixture(mixture_filename, genomes_filename, params):

    gm_sampler = remixt.simulations.experiment.GenomeMixtureSampler(params)

    with open(genomes_filename, 'r') as genomes_file:
        gc = pickle.load(genomes_file)

    np.random.seed(params['random_seed'])

    gm = gm_sampler.sample_genome_mixture(gc)

    with open(mixture_filename, 'w') as mixture_file:
        pickle.dump(gm, mixture_file)


def simulate_experiment(experiment_filename, mixture_filename, params):

    exp_sampler = remixt.simulations.experiment.ExperimentSampler(params)

    with open(mixture_filename, 'r') as mixture_file:
        gm = pickle.load(mixture_file)

    np.random.seed(params['random_seed'])

    exp = exp_sampler.sample_experiment(gm)

    with open(experiment_filename, 'w') as experiment_file:
        pickle.dump(exp, experiment_file)


def simulate_germline_alleles(germline_alleles_filename, params, config, ref_data_dir):

    np.random.seed(params['random_seed'])

    chromosomes = params['chromosomes']

    alleles_table = remixt.simulations.haplotype.create_sim_alleles(chromosomes, config, ref_data_dir)

    alleles_table.to_csv(germline_alleles_filename, sep='\t', index=False, header=True)


def simulate_normal_data(read_data_filename, genome_filename, germline_alleles_filename, params):

    with open(genome_filename, 'r') as genome_file:
        gc = pickle.load(genome_file)

    germline_alleles = pd.read_csv(
        germline_alleles_filename, sep='\t',
        usecols=['chromosome', 'position', 'is_alt_0', 'is_alt_1'],
        dtype={'chromosome': str, 'position': np.uint32, 'is_alt_0': np.uint8, 'is_alt_1': np.uint8})

    np.random.seed(params['random_seed'])

    remixt.simulations.seqread.simulate_mixture_read_data(
        read_data_filename,
        [gc.genomes[0]],
        [params['h_total']],
        germline_alleles,
        params)


def resample_normal_data(read_data_filename, source_filename, genome_filename, germline_alleles_filename, params):

    with open(genome_filename, 'r') as genome_file:
        gc = pickle.load(genome_file)

    germline_alleles = pd.read_csv(
        germline_alleles_filename, sep='\t',
        usecols=['chromosome', 'position', 'is_alt_0', 'is_alt_1'],
        dtype={'chromosome': str, 'position': np.uint32, 'is_alt_0': np.uint8, 'is_alt_1': np.uint8})

    np.random.seed(params['random_seed'])

    remixt.simulations.seqread.resample_mixture_read_data(
        read_data_filename,
        source_filename,
        [gc.genomes[0]],
        [params['h_total']],
        germline_alleles,
        params)


def simulate_tumour_data(read_data_filename, mixture_filename, germline_alleles_filename, params):

    with open(mixture_filename, 'r') as mixture_file:
        gm = pickle.load(mixture_file)

    germline_alleles = pd.read_csv(
        germline_alleles_filename, sep='\t',
        usecols=['chromosome', 'position', 'is_alt_0', 'is_alt_1'],
        dtype={'chromosome': str, 'position': np.uint32, 'is_alt_0': np.uint8, 'is_alt_1': np.uint8})

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

    germline_alleles = pd.read_csv(
        germline_alleles_filename, sep='\t',
        usecols=['chromosome', 'position', 'is_alt_0', 'is_alt_1'],
        dtype={'chromosome': str, 'position': np.uint32, 'is_alt_0': np.uint8, 'is_alt_1': np.uint8})

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

    breakpoint_table = list()

    for breakpoint in mixture.detected_breakpoints:

        breakpoint_row = dict()

        for idx, breakend in enumerate(breakpoint):

            n, side = breakend
            chromosome = mixture.segment_chromosome_id[n]
            if side == 0:
                strand = '-'
                position = mixture.segment_start[n]
            elif side == 1:
                strand = '+'
                position = mixture.segment_end[n]

            breakpoint_row['chromosome_{0}'.format(idx+1)] = chromosome
            breakpoint_row['strand_{0}'.format(idx+1)] = strand
            breakpoint_row['position_{0}'.format(idx+1)] = position

        breakpoint_table.append(breakpoint_row)

    breakpoint_table = pd.DataFrame(breakpoint_table)
    breakpoint_table['prediction_id'] = xrange(len(breakpoint_table.index))

    breakpoint_table.to_csv(breakpoint_filename, sep='\t', header=True, index=False)


def evaluate_cn_results(genome_mixture, cn_data_table, order_true, order_pred, allow_swap):
    """ Evaluate predicted copy number results against known mixture.

    Args:
        genome_mixture (remixt.simulations.GenomeMixture): known mixture
        cn_data_table (pandas.DataFrame): predicted copy number data
        order_true (numpy.array): indices in order of increasing true clone prevalence
        order_pred (numpy.array): indices in order of increasing predicted clone prevalence
        allow_swap (bool): allow copy number swapping with no penalty

    Returns:
        dict: dictionary of evaluation statistics

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

    results = dict()
    results['proportion_cn_correct'] = proportion_cn_correct
    results['proportion_dom_cn_correct'] = proportion_dom_cn_correct
    results['proportion_clonal_correct'] = proportion_clonal_correct
    results['proportion_subclonal_correct'] = proportion_subclonal_correct

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
        dict: dictionary of evaluation statistics

    Predicted breakpoint copy number data `brk_cn_table` should have columns
    'n_1', 'side_1', 'n_2', 'side_2', in addition to either 'cn_m', for clone 'm'.

    """

    M = genome_mixture.M

    # List of column names for known true copy number
    true_cols = []
    for m in xrange(1, M):
        true_cols.append('true_cn_{}'.format(m))

    # List of column names for predicted copy number
    pred_cols = []
    for m in xrange(1, M):
        pred_cols.append('cn_{}'.format(m))

    data = brk_cn_table.copy()

    # Add columns for known true copy number
    for col in true_cols:
        data[col] = 0

    # Default False for is_balanced
    data['is_balanced'] = False

    # Annotate each breakpoint with its predicted copy number
    true_breakpoint_copy_number = genome_mixture.genome_collection.breakpoint_copy_number
    true_balanced_breakpoints = genome_mixture.genome_collection.balanced_breakpoints
    for idx in data.index:
        n_1, side_1, n_2, side_2 = data.loc[idx, ['n_1', 'side_1', 'n_2', 'side_2']].values
        for ell_1 in xrange(2):
            for ell_2 in xrange(2):
                allele_bp = frozenset([((n_1, ell_1), side_1), ((n_2, ell_2), side_2)])
                if allele_bp in true_breakpoint_copy_number:
                    bp_cn = true_breakpoint_copy_number[allele_bp]
                    for m in xrange(1, M):
                        data.loc[idx, 'true_cn_{}'.format(m)] = bp_cn[m]
                if allele_bp in true_balanced_breakpoints:
                    data.loc[idx, 'is_balanced'] = True

    # Remove balanced breakpoints
    data = data[~data['is_balanced']]

    # Ensure true tumour clones are consistent
    cn_true = data[true_cols].values[:, order_true]

    # Ensure predicted tumour clones are consistent
    cn_pred = data[pred_cols].values[:, order_pred]

    # Allow for clone copy number swapping if the mix fractions are close to equal
    if allow_swap:
        cn_correct = (cn_true == cn_pred).all(axis=(1,)) | (cn_true == cn_pred[:, ::-1]).all(axis=(1,))
    else:
        cn_correct = (cn_true == cn_pred).all(axis=(1,))

    # Calculate correctness per prediction
    data['cn_correct'] = cn_correct
    data['true_present'] = (data[true_cols] > 0).any(axis=1)
    data['pred_present'] = (data[pred_cols] > 0).any(axis=1)
    data['true_subclonal'] = (data[true_cols] == 0).any(axis=1) & data['true_present']
    data['pred_subclonal'] = (data[pred_cols] == 0).any(axis=1) & data['pred_present']

    results = dict()
    results['brk_cn_correct_proportion'] = float(data['cn_correct'].sum()) / float(len(data.index))
    results['brk_cn_present_num_true'] = float(data['true_present'].sum())
    results['brk_cn_present_num_pos'] = float(data['pred_present'].sum())
    results['brk_cn_present_num_true_pos'] = float((data['pred_present'] & data['true_present']).sum())
    results['brk_cn_subclonal_num_true'] = float(data['true_subclonal'].sum())
    results['brk_cn_subclonal_num_pos'] = float(data['pred_subclonal'].sum())
    results['brk_cn_subclonal_num_true_pos'] = float((data['pred_subclonal'] & data['true_subclonal']).sum())

    return results


def evaluate_results(genome_mixture, cn_table, brk_cn_table, mix_pred):
    """ Evaluate predicted results against known mixture.

    Args:
        mixture (remixt.simulations.GenomeMixture): known mixture
        cn_table (pandas.DataFrame): predicted copy number data
        brk_cn_table (pandas.DataFrame): predicted copy number table
        mix_pred (numpy.array): predicted mixture

    Predicted copy number data `cn_table` should have columns 'chromosome', 'start', 'end',
    in addition to either 'major_m', 'minor_m' for clone 'm', or 'total_m' for allele non-specific
    copy number predictions.

    """

    mix_true = genome_mixture.frac.copy()

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

    for idx, f in enumerate(mix_true):
        results['mix_true_'+str(idx)] = f

    for idx, f in enumerate(mix_pred):
        results['mix_pred_'+str(idx)] = f

    return results


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
    evaluation = pd.DataFrame([evaluation])

    with pd.HDFStore(evaluation_filename, 'w') as store:
        store['/evaluation'] = evaluation


def merge_evaluations(merged_filename, sim_defs, evaluation_filenames, key_names):
    """ Merge multiple evaluations of prediction results.

    Args:
        merged_filename (str): output hdf filename of merged evaluations
        sim_defs (dict): simulation definitions per simulation
        evaluation_filenames (dict of str): hdf filename of evaluation per simulation per tool

    """

    sim_defs_table = pd.DataFrame(
        sim_defs.values(),
        index=pd.Index(sim_defs.keys(), name='sim_id'),
    ).reset_index()

    evaluations_table = []
    for key, evaluation_filename in evaluation_filenames.iteritems():
        with pd.HDFStore(evaluation_filename, 'r') as store:
            evaluation = store['/evaluation']

        for value, name in zip(key, key_names):
            evaluation[name] = value

        evaluations_table.append(evaluation)

    evaluations_table = pd.concat(evaluations_table, ignore_index=True)

    with pd.HDFStore(merged_filename, 'w') as merged_store:
        merged_store['/simulations'] = sim_defs_table
        merged_store['/evaluations'] = evaluations_table


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

    return workflow_function(
        normal_seqdata_filename,
        {'tumour': tumour_seqdata_filename},
        config,
        results_filenames,
        raw_data_directory,
        somatic_breakpoint_file=breakpoints_filename,
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
