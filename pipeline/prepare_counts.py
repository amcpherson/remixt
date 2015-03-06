import sys
import logging
import os
import itertools
import argparse
import gzip
import collections
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

import pypeliner
import pypeliner.managed as mgd


demix_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))

bin_directory = os.path.join(demix_directory, 'bin')
default_config_filename = os.path.join(demix_directory, 'defaultconfig.py')

sys.path.append(demix_directory)

import demix
import demix.seqdataio
import demix.segalg
import demix.utils


if __name__ == '__main__':

    import prepare_counts
    
    argparser = argparse.ArgumentParser()

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
                           help='Reference dataset directory')

    argparser.add_argument('changepoints',
                           help='Input changepoints file')

    argparser.add_argument('normal_file',
                           help='Input normal sequence data filename')

    argparser.add_argument('--tumour_files', nargs='+', required=True,
                           help='Input tumour sequence data filenames')

    argparser.add_argument('--tumour_ids', nargs='+', required=True,
                           help='Input ids for respective tumour filenames')

    argparser.add_argument('--count_files', nargs='+', required=True,
                           help='Output count TSV filenames')

    argparser.add_argument('--config', required=False,
                           help='Configuration Filename')

    args = vars(argparser.parse_args())

    if len(args['tumour_files']) != len(args['tumour_ids']):
        raise Exception('--tumour_ids must correspond one to one with --tumour_files')

    if len(args['tumour_files']) != len(args['count_files']):
        raise Exception('--count_files must correspond one to one with --tumour_files')

    normal_id = 'normal'

    if normal_id in args['tumour_ids']:
        raise Exception('do not specifiy normal with --tumour_files/--tumour_ids')

    config = {'ref_data_directory':args['ref_data_dir']}
    execfile(default_config_filename, {}, config)

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([prepare_counts], config)

    ctx_general = {'mem':16, 'ncpus':1}

    for tumour_id, tumour_filename in zip(args['tumour_ids'], args['tumour_files']):

        pyp.sch.transform('calc_fragment_stats_{0}'.format(tumour_id), (), ctx_general,
            prepare_counts.calculate_fragment_stats,
            mgd.TempOutputObj('fragstats.{0}'.format(tumour_id)),
            mgd.InputFile(tumour_filename))

    for chromosome in config['chromosomes']:

        pyp.sch.transform('infer_haps_{0}'.format(chromosome), (), ctx_general,
            prepare_counts.infer_haps,
            None,
            config,
            pyp.sch.temps_dir,
            normal_id,
            chromosome,
            config['snp_positions'],
            mgd.InputFile(args['normal_file']),
            mgd.TempOutputFile('haps.{0}'.format(chromosome)))

    for tumour_id, tumour_filename in zip(args['tumour_ids'], args['tumour_files']):

        for chromosome in config['chromosomes']:

            pyp.sch.transform('create_readcounts_{0}_{1}'.format(chromosome, tumour_id), (), ctx_general,
                prepare_counts.create_counts,
                None,
                chromosome, 
                mgd.InputFile(tumour_filename),
                mgd.InputFile(config['changepoints']),
                mgd.TempInputFile('haps.{0}'.format(chromosome)),
                mgd.TempOutputFile('segment.readcounts.{0}.{1}'.format(chromosome, tumour_id)),
                mgd.TempOutputFile('alleles.readcounts.{0}.{1}'.format(chromosome, tumour_id)),
                mgd.InputFile(config['genome_fai']))

    for chromosome in config['chromosomes']:

        pyp.sch.transform('phase_segments_{0}'.format(chromosome), (), ctx_general,
            prepare_counts.phase_segments,
            None,
            *([mgd.TempInputFile('alleles.readcounts.{0}.{1}'.format(chromosome, tumour_id)) for tumour_id in args['tumour_ids']] + 
              [mgd.TempOutputFile('alleles.readcounts.phased.{0}.{1}'.format(chromosome, tumour_id)) for tumour_id in args['tumour_ids']]))

    for tumour_id, tumour_filename in zip(args['tumour_ids'], args['tumour_files']):

        pyp.sch.transform('merge_segment_readcounts_{0}'.format(tumour_id), (), ctx_general,
            demix.utils.merge_files,
            None,
            mgd.TempOutputFile('segment.readcounts.{0}'.format(tumour_id)),
            *[mgd.TempInputFile('segment.readcounts.{0}.{1}'.format(chromosome, tumour_id)) for chromosome in config['chromosomes']])

        pyp.sch.transform('merge_allele_readcounts_{0}'.format(tumour_id), (), ctx_general,
            demix.utils.merge_files,
            None,
            mgd.TempOutputFile('alleles.readcounts.phased.{0}'.format(tumour_id)),
            *[mgd.TempInputFile('alleles.readcounts.phased.{0}.{1}'.format(chromosome, tumour_id)) for chromosome in config['chromosomes']])

        pyp.sch.transform('sample_gc_{0}'.format(tumour_id), (), ctx_general,
            prepare_counts.sample_gc,
            None,
            mgd.TempOutputFile('gcsamples.{0}'.format(tumour_id)),
            mgd.InputFile(tumour_filename),
            mgd.TempInputObj('fragstats.{0}'.format(tumour_id)).prop('fragment_mean'),
            config)

        pyp.sch.transform('gc_lowess_{0}'.format(tumour_id), (), ctx_general,
            prepare_counts.gc_lowess,
            None,
            mgd.TempInputFile('gcsamples.{0}'.format(tumour_id)),
            mgd.TempOutputFile('gcloess.{0}'.format(tumour_id)),
            mgd.TempOutputFile('gcplots.{0}.pdf'.format(tumour_id)))

        pyp.sch.commandline('gc_segment_{0}'.format(tumour_id), (), ctx_general,
            os.path.join(bin_directory, 'estimategc'),
            '-m', config['mappability_filename'],
            '-g', config['genome_fasta'],
            '-c', mgd.TempInputFile('segment.readcounts.{0}'.format(tumour_id)),
            '-i',
            '-o', '4',
            '-u', mgd.TempInputObj('fragstats.{0}'.format(tumour_id)).prop('fragment_mean'),
            '-s', mgd.TempInputObj('fragstats.{0}'.format(tumour_id)).prop('fragment_stddev'),
            '-a', config['mappability_length'],
            '-l', mgd.TempInputFile('gcloess.{0}'.format(tumour_id)),
            '>', mgd.TempOutputFile('segment.readcounts.lengths.{0}'.format(tumour_id)))

    for tumour_id, count_filename in zip(args['tumour_ids'], args['count_files']):

        pyp.sch.transform('prepare_counts_{0}'.format(tumour_id), (), ctx_general,
            prepare_counts.prepare_counts,
            None,
            mgd.TempInputFile('segment.readcounts.lengths.{0}'.format(tumour_id)),
            mgd.TempInputFile('alleles.readcounts.phased.{0}'.format(tumour_id)),
            mgd.OutputFile(count_filename))

    pyp.run()


FragmentStats = collections.namedtuple('FragmentStats', [
    'fragment_mean',
    'fragment_stddev',
])

def calculate_fragment_stats(seqdata_filename):

    segment_counts = list()
    
    sum_x = 0.
    sum_x2 = 0.
    n = 0.

    chromosomes = demix.seqdataio.read_chromosomes(seqdata_filename)

    for chrom in chromosomes:

        chrom_reads = next(demix.seqdataio.read_read_data(seqdata_filename, chromosome=chrom))

        length = chrom_reads['end'].values - chrom_reads['start'].values

        sum_x += length.sum()
        sum_x2 += (length * length).sum()
        n += length.shape[0]

    mean = sum_x / n
    stdev = np.sqrt((sum_x2 / n) - (mean * mean)) 

    return FragmentStats(mean, stdev)


def create_counts(chromosome, seqdata_filename, changepoints_filename, haps_filename,
                  segment_filename, allele_counts_filename, genome_fai_filename):
    
    # Read changepoint data
    changepoints = pd.read_csv(changepoints_filename, sep='\t', header=None,
                               converters={'chromosome':str}, 
                               names=['chromosome', 'position'])
    haps = pd.read_csv(haps_filename, sep='\t')
    reads = next(demix.seqdataio.read_read_data(seqdata_filename, chromosome=chromosome))
    reads.sort('start', inplace=True)
    
    # Create a list of regions between changepoints
    changepoints = changepoints[changepoints['chromosome'] == chromosome]
    changepoints = changepoints.append({'chromosome':chromosome, 'position':1}, ignore_index=True)
    changepoints = changepoints.append({'chromosome':chromosome, 'position':demix.utils.read_chromosome_lengths(genome_fai_filename)[chromosome]}, ignore_index=True)
    changepoints.drop_duplicates(inplace=True)
    changepoints.sort('position', inplace=True)
    regions = pd.DataFrame(data=np.array([changepoints['position'].values[:-1],
                                          changepoints['position'].values[1:]]).T,
                           columns=['start', 'end'])
    regions.drop_duplicates(inplace=True)
    regions.sort('start', inplace=True)
    
    # Create an index that matches the sort order
    regions.index = xrange(len(regions))

     # Count segment reads
    segment_counts = demix.segalg.contained_counts(regions[['start', 'end']].values, reads[['start', 'end']].values)

    del reads

    # Create segment data
    segment_data = pd.DataFrame({'start':regions['start'].values, 'end':regions['end'].values, 'counts':segment_counts})

    # Write segment data to a file
    segment_data['id'] = chromosome + '_'
    segment_data['id'] += segment_data.index.values.astype(str)
    segment_data['counts'] = segment_data['counts'].astype(int)
    segment_data['chromosome_1'] = chromosome
    segment_data['strand_1'] = '-'
    segment_data['chromosome_2'] = chromosome
    segment_data['strand_2'] = '+'
    segment_data = segment_data[['id', 'chromosome_1', 'start', 'strand_1', 'chromosome_2', 'end', 'strand_2', 'counts']]
    segment_data.to_csv(segment_filename, sep='\t', index=False, header=False)

    # Merge haplotype information into read alleles table
    alleles = list()
    for alleles_chunk in demix.seqdataio.read_allele_data(seqdata_filename, chromosome=chromosome, num_rows=10000):
        alleles_chunk = alleles_chunk.merge(haps, left_on=['position', 'is_alt'], right_on=['position', 'allele'], how='inner')
        alleles.append(alleles_chunk)
    alleles = pd.concat(alleles, ignore_index=True)

    # Arbitrarily assign a haplotype/allele label to each read
    alleles.drop_duplicates('fragment_id', inplace=True)

    # Create a mapping between regions and snp positions
    snp_region = pd.DataFrame({'position':haps['position'].unique()})
    snp_region['region_idx'] = demix.segalg.find_contained(regions[['start', 'end']].values, snp_region['position'].values)
    snp_region = snp_region.dropna()
    snp_region['region_idx'] = snp_region['region_idx'].astype(int)

    # Add annotation of which region each snp is contained within
    alleles = alleles.merge(snp_region, left_on='position', right_on='position')

    # Count reads for each allele
    alleles.set_index(['region_idx', 'hap_label', 'allele_id'], inplace=True)
    allele_counts = alleles.groupby(level=[0, 1, 2]).size().reset_index().rename(columns={0:'count'})

    # Create region id as chromosome _ index
    allele_counts['region_id'] = chromosome + '_'
    allele_counts['region_id'] += allele_counts['region_idx'].astype(str)

    # Write out allele counts
    allele_counts.to_csv(allele_counts_filename, sep='\t', cols=['region_id', 'hap_label', 'allele_id', 'count'], index=False, header=False)


def infer_haps(config, temps_directory, library, chromosome, snps_filename, seqdata_filename, haps_filename):
    
    def write_null():
        with open(hets_filename, 'w') as hets_file:
            pass
        with open(haps_filename, 'w') as haps_file:
            haps_file.write('pos\tallele\tchangepoint_confidence\thap_label\tallele_id\tallele_label\n')

    accepted_chromosomes = [str(a) for a in range(1, 23)] + ['X']
    if str(chromosome) not in accepted_chromosomes:
        write_null()
        return
    
    # Temporary directory for impute2 files
    haps_temp_directory = os.path.join(os.path.join(temps_directory, library), chromosome)
    try:
        os.makedirs(haps_temp_directory)
    except OSError:
        pass

    # Impute 2 files for thousand genomes data by chromosome
    phased_chromosome = chromosome
    if chromosome == 'X':
        phased_chromosome = config['phased_chromosome_x']
    genetic_map_filename = config['genetic_map_template'].format(phased_chromosome)
    hap_filename = config['haplotypes_template'].format(phased_chromosome)
    legend_filename = config['legend_template'].format(phased_chromosome)

    # Call snps based on reference and alternate read counts from normal
    snp_counts_df = list()
    for alleles_chunk in demix.seqdataio.read_allele_data(seqdata_filename, chromosome=chromosome, num_rows=10000):
        snp_counts_chunk = alleles_chunk.groupby(['position', 'is_alt']).size().unstack().fillna(0)
        snp_counts_chunk = snp_counts_chunk.rename(columns=lambda a: {0:'ref_count', 1:'alt_count'}[a])
        snp_counts_chunk = snp_counts_chunk.astype(float)
        snp_counts_df.append(snp_counts_chunk)
    snp_counts_df = pd.concat(snp_counts_df)
    snp_counts_df = snp_counts_df.groupby(level=0).sum()
    snp_counts_df.sort_index(inplace=True)

    if len(snp_counts_df) == 0:
        write_null()
        return

    snp_counts_df['total_count'] = snp_counts_df['ref_count'] + snp_counts_df['alt_count']

    snp_counts_df['likelihood_AA'] = scipy.stats.binom.pmf(snp_counts_df['alt_count'], snp_counts_df['total_count'], float(config['sequencing_base_call_error']))
    snp_counts_df['likelihood_AB'] = scipy.stats.binom.pmf(snp_counts_df['alt_count'], snp_counts_df['total_count'], 0.5)
    snp_counts_df['likelihood_BB'] = scipy.stats.binom.pmf(snp_counts_df['ref_count'], snp_counts_df['total_count'], float(config['sequencing_base_call_error']))
    snp_counts_df['evidence'] = snp_counts_df['likelihood_AA'] + snp_counts_df['likelihood_AB'] + snp_counts_df['likelihood_BB']

    snp_counts_df['posterior_AA'] = snp_counts_df['likelihood_AA'] / snp_counts_df['evidence']
    snp_counts_df['posterior_AB'] = snp_counts_df['likelihood_AB'] / snp_counts_df['evidence']
    snp_counts_df['posterior_BB'] = snp_counts_df['likelihood_BB'] / snp_counts_df['evidence']

    snp_counts_df['AA'] = (snp_counts_df['posterior_AA'] >= float(config['het_snp_call_threshold'])) * 1
    snp_counts_df['AB'] = (snp_counts_df['posterior_AB'] >= float(config['het_snp_call_threshold'])) * 1
    snp_counts_df['BB'] = (snp_counts_df['posterior_BB'] >= float(config['het_snp_call_threshold'])) * 1

    snp_counts_df = snp_counts_df[(snp_counts_df['AA'] == 1) | (snp_counts_df['AB'] == 1) | (snp_counts_df['BB'] == 1)]

    snps_df_iter = pd.read_csv(snps_filename, sep='\t', names=['chr', 'position', 'ref', 'alt'], converters={'chr':str}, iterator=True, chunksize=10000)
    snps_df = pd.concat([chunk[chunk['chr'] == chromosome] for chunk in snps_df_iter])
    snps_df.drop('chr', axis=1)
    snps_df.set_index('position', inplace=True)

    snp_counts_df = snp_counts_df.merge(snps_df, left_index=True, right_index=True)

    # Create genotype file required by impute2
    temp_gen_filename = os.path.join(haps_temp_directory, 'snps.gen')
    snp_counts_df.reset_index(inplace=True)
    snp_counts_df['chr'] = chromosome
    snp_counts_df['chr_pos'] = snp_counts_df['chr'].astype(str) + ':' + snp_counts_df['position'].astype(str)
    snp_counts_df.to_csv(temp_gen_filename, sep=' ', cols=['chr', 'chr_pos', 'position', 'ref', 'alt', 'AA', 'AB', 'BB'], index=False, header=False)

    # Create single sample file required by impute2
    temp_sample_filename = os.path.join(haps_temp_directory, 'snps.sample')
    with open(temp_sample_filename, 'w') as temp_sample_file:
        temp_sample_file.write('ID_1 ID_2 missing sex\n0 0 0 0\nUNR1 UNR1 0 2\n')

    # Run shapeit to create phased haplotype graph
    hgraph_filename = os.path.join(haps_temp_directory, 'phased.hgraph')
    hgraph_logs_prefix = hgraph_filename + '.log'
    chr_x_flag = ''
    if chromosome == 'X':
        chr_x_flag = '--chrX'
    pypeliner.commandline.execute('shapeit', '-M', genetic_map_filename, '-R', hap_filename, legend_filename, config['sample_filename'],
                                  '-G', temp_gen_filename, temp_sample_filename, '--output-graph', hgraph_filename, chr_x_flag,
                                  '--no-mcmc', '-L', hgraph_logs_prefix)

    # Run shapeit to sample from phased haplotype graph
    sample_template = os.path.join(haps_temp_directory, 'sampled.{0}')
    averaged_changepoints = None
    for s in range(int(config['shapeit_num_samples'])):
        sample_prefix = sample_template.format(s)
        sample_log_filename = sample_prefix + '.log'
        sample_haps_filename = sample_prefix + '.haps'
        sample_sample_filename = sample_prefix + '.sample'
        pypeliner.commandline.execute('shapeit', '-convert', '--input-graph', hgraph_filename, '--output-sample', 
                                      sample_prefix, '--seed', str(s), '-L', sample_log_filename)
        sample_haps = pd.read_csv(sample_haps_filename, sep=' ', header=None, 
                                  names=['id', 'id2', 'position', 'ref', 'alt', 'allele1', 'allele2'],
                                  usecols=['position', 'allele1', 'allele2'])
        sample_haps = sample_haps[sample_haps['allele1'] != sample_haps['allele2']]
        sample_haps['allele'] = sample_haps['allele1']
        sample_haps = sample_haps.drop(['allele1', 'allele2'], axis=1)
        sample_haps.set_index('position', inplace=True)
        sample_changepoints = sample_haps['allele'].diff().abs().astype(float).fillna(0.0)
        if averaged_changepoints is None:
            averaged_changepoints = sample_changepoints
        else:
            averaged_changepoints += sample_changepoints
        os.remove(sample_log_filename)
        os.remove(sample_haps_filename)
        os.remove(sample_sample_filename)
    averaged_changepoints /= float(config['shapeit_num_samples'])
    last_sample_haps = sample_haps

    # Identify changepoints recurrent across samples
    changepoint_confidence = np.maximum(averaged_changepoints, 1.0 - averaged_changepoints)

    # Create a list of labels for haplotypes between recurrent changepoints
    current_hap_label = 0
    hap_label = list()
    for x in changepoint_confidence:
        if x < float(config['shapeit_confidence_threshold']):
            current_hap_label += 1
        hap_label.append(current_hap_label)

    # Create the list of haplotypes
    haps = last_sample_haps
    haps['changepoint_confidence'] = changepoint_confidence
    haps['hap_label'] = hap_label

    haps.reset_index(inplace=True)

    haps['allele_id'] = 0

    haps_allele2 = haps.copy()
    haps_allele2['allele_id'] = 1
    haps_allele2['allele'] = 1 - haps_allele2['allele']

    haps = pd.concat([haps, haps_allele2], ignore_index=True)
    haps.sort(['position', 'allele_id'], inplace=True)

    haps.set_index(['hap_label', 'allele_id'], inplace=True)
    hap_label_counter = itertools.count()
    haps['allele_label'] = haps.groupby(level=[0, 1]).apply(lambda a: next(hap_label_counter))
    haps.reset_index(inplace=True)

    haps = haps[['position', 'allele', 'changepoint_confidence', 'hap_label', 'allele_id', 'allele_label']]

    haps.to_csv(haps_filename, sep='\t', header=True, index=False)


def phase_segments(*args):

    input_alleles_filenames = args[:len(args)/2]
    output_alleles_filenames = args[len(args)/2:]

    allele_phases = list()
    allele_diffs = list()

    for idx, input_alleles_filename in enumerate(input_alleles_filenames):

        allele_data = pd.read_csv(input_alleles_filename, sep='\t', header=None, names=['segment_id', 'hap_label', 'allele_id', 'readcount'])
        
        # Allele readcount table
        allele_data = allele_data.set_index(['segment_id', 'hap_label', 'allele_id'])['readcount'].unstack().fillna(0.0)
        allele_data = allele_data.astype(float)
        
        # Create major allele call
        allele_phase = allele_data.apply(np.argmax, axis=1)
        allele_phase.name = 'major_allele_id'
        allele_phase = allele_phase.reset_index()
        allele_phase['library_idx'] = idx
        allele_phases.append(allele_phase)

        # Calculate major minor allele read counts, and diff between them
        allele_data['major_readcount'] = allele_data.apply(np.max, axis=1)
        allele_data['minor_readcount'] = allele_data.apply(np.min, axis=1)
        allele_data['diff_readcount'] = allele_data['major_readcount'] - allele_data['minor_readcount']
        allele_data['total_readcount'] = allele_data['major_readcount'] + allele_data['minor_readcount']

        # Calculate normalized major and minor read counts difference per segment
        allele_diff = allele_data.groupby(level=[0])[['diff_readcount', 'total_readcount']].sum()
        allele_diff['norm_diff_readcount'] = allele_diff['diff_readcount'] / allele_diff['total_readcount']
        allele_diff = allele_diff[['norm_diff_readcount']]

        # Add to table for all librarys
        allele_diff.reset_index(inplace=True)
        allele_diff['library_idx'] = idx
        allele_diffs.append(allele_diff)

    allele_phases = pd.concat(allele_phases, ignore_index=True)
    allele_diffs = pd.concat(allele_diffs, ignore_index=True)

    def select_largest_diff(df):
        largest_idx = np.argmax(df['norm_diff_readcount'].values)
        return df['library_idx'].values[largest_idx]

    # For each segment, select the library with the largest difference between major and minor
    segment_library = allele_diffs.set_index('segment_id').groupby(level=0).apply(select_largest_diff)
    segment_library.name = 'library_idx'
    segment_library = segment_library.reset_index()

    # For each haplotype block in each segment, take the major allele call of the library
    # with the largest major minor difference and call it allele 'a'
    allele_phases = allele_phases.merge(segment_library, left_on=['segment_id', 'library_idx'], right_on=['segment_id', 'library_idx'], how='right')
    allele_phases = allele_phases[['segment_id', 'hap_label', 'major_allele_id']].rename(columns={'major_allele_id': 'allele_a_id'})

    for idx, (input_alleles_filename, output_allele_filename) in enumerate(zip(input_alleles_filenames, output_alleles_filenames)):

        allele_data = pd.read_csv(input_alleles_filename, sep='\t', header=None, names=['segment_id', 'hap_label', 'allele_id', 'readcount'])
        
        # Add a boolean column denoting which allele is allele 'a'
        allele_data = allele_data.merge(allele_phases, left_on=['segment_id', 'hap_label'], right_on=['segment_id', 'hap_label'])
        allele_data['is_allele_a'] = (allele_data['allele_id'] == allele_data['allele_a_id']) * 1
        allele_data = allele_data[['segment_id', 'hap_label', 'allele_id', 'readcount', 'is_allele_a']]

        allele_data.to_csv(output_allele_filename, sep='\t', header=False, index=False)


def sample_gc(gc_samples_filename, seqdata_filename, fragment_length, config):

    num_samples = config['sample_gc_num_positions']
    position_offset = config['sample_gc_offset']
    genome_fai = config['genome_fai']
    genome_fasta = config['genome_fasta']
    mappability_filename = config['mappability_filename']

    fragment_length = int(fragment_length)
    gc_window = fragment_length - 2 * position_offset

    chromosomes = pd.DataFrame({'chrom_length':demix.utils.read_chromosome_lengths(genome_fai)})
    chromosomes['chrom_end'] = chromosomes['chrom_length'].cumsum()
    chromosomes['chrom_start'] = chromosomes['chrom_end'] - chromosomes['chrom_length']

    # Sample random genomic positions from concatenated genome
    genome_length = chromosomes['chrom_length'].sum()
    sample_pos = np.sort(np.random.randint(0, genome_length, num_samples))


    # Calculate mappability for each sample
    sample_mappability = np.zeros(sample_pos.shape)

    # Iterate large mappability file
    mappability_iter = pd.read_csv(mappability_filename, sep='\t', header=None, iterator=True, chunksize=10000,
        converters={'chromosome':str}, names=['chromosome', 'start', 'end', 'score'])
    for mappability in mappability_iter:

        # Perfect mappability only
        mappability = mappability[mappability['score'] == 1]

        # Add chromosome start end and calculate start/end in concatenated genome
        mappability = mappability.merge(chromosomes[['chrom_start']], left_on='chromosome', right_index=True)
        mappability['start'] += mappability['chrom_start']
        mappability['end'] += mappability['chrom_start']

        # Add mappability for current iteration
        sample_mappability += demix.segalg.overlapping_counts(sample_pos, mappability[['start', 'end']].values)

    # Filter unmappable positions
    sample_pos = sample_pos[sample_mappability > 0]

    # Calculate GC for each position
    sample_gc_count = np.zeros(sample_pos.shape)
    for chrom_id, sequence in demix.utils.read_sequences(genome_fasta):

        # Start and end of current chromosome in concatenated genome
        chrom_start, chrom_end = chromosomes.loc[chrom_id, ['chrom_start', 'chrom_end']].values

        # Calculate gc count within sliding window
        sequence = np.array(list(sequence.upper()))
        gc = ((sequence == 'G') | (sequence == 'C'))
        gc_count = gc.cumsum()
        gc_count[gc_window:] = gc_count[gc_window:] - gc.cumsum()[:-gc_window]

        # Calculate filter of positions in this chromosome
        chrom_sample_idx = (sample_pos >= chrom_start) & (sample_pos < chrom_end)

        # Calculate last position in window
        chrom_window_end = sample_pos[chrom_sample_idx] - chrom_start + fragment_length - position_offset - 1

        # Add the gc count for filtered positions
        sample_gc_count[chrom_sample_idx] += gc_count[chrom_window_end]

    sample_gc_percent = sample_gc_count / float(gc_window)

    # Count number of reads at each position
    sample_read_count = np.zeros(sample_pos.shape, dtype=int)
    for chrom_id in demix.seqdataio.read_chromosomes(seqdata_filename):

        chrom_reads = next(demix.seqdataio.read_read_data(seqdata_filename, chromosome=chrom_id))

        # Calculate read start in concatenated genome
        chrom_reads['start'] += chromosomes.loc[chrom_id, 'chrom_start']

        # Add reads at each start
        sample_read_count += (
            chrom_reads
            .groupby('start')['end']
            .count()
            .reindex(sample_pos)
            .fillna(0)
            .astype(int)
            .values
        )

    # Calculate position in non-concatenated genome
    sample_chrom_idx = np.searchsorted(chromosomes['chrom_end'].values, sample_pos, side='right')
    sample_chrom = chromosomes.index.values[sample_chrom_idx]
    sample_chrom_pos = sample_pos - chromosomes['chrom_start'].values[sample_chrom_idx]

    # Output chromosome, position, gc percent, read count
    gc_sample_data = pd.DataFrame({
        'chromosome':sample_chrom,
        'position':sample_chrom_pos,
        'gc_percent':sample_gc_percent,
        'read_count':sample_read_count,
    })
    gc_sample_data = gc_sample_data[[
        'chromosome',
        'position',
        'gc_percent',
        'read_count'
    ]]

    gc_sample_data.to_csv(gc_samples_filename, sep='\t', header=False, index=False)


def gc_lowess(gc_samples_filename, gc_dist_filename, plot_filename, gc_resolution=100):

    gc_samples = pd.read_csv(gc_samples_filename, sep='\t', names=['chromosome', 'position', 'gc', 'count'])

    gc_samples['gc_bin'] = np.round(gc_samples['gc'] * gc_resolution)

    gc_binned = gc_samples.groupby('gc_bin')['count'] \
                          .agg({'sum':np.sum, 'len':len, 'mean':np.mean}) \
                          .reindex(xrange(gc_resolution+1)) \
                          .fillna(0) \
                          .reset_index() \
                          .rename(columns={'index':'gc_bin'}) \
                          .astype(float)

    gc_binned['smoothed'] = sm.nonparametric.lowess(gc_binned['mean'].values, gc_binned['gc_bin'].values, frac=0.2).T[1]
    assert not gc_binned['smoothed'].isnull().any()

    rescale = 1. / gc_binned['smoothed'].max()

    gc_binned['mean'] = gc_binned['mean'] * rescale
    gc_binned['smoothed'] = gc_binned['smoothed'] * rescale

    fig = plt.figure(figsize=(4,4))

    plt.scatter(gc_binned['gc_bin'].values, gc_binned['mean'].values, c='k', s=4)
    plt.plot(gc_binned['gc_bin'].values, gc_binned['smoothed'].values, c='r')

    plt.xlabel('gc %')
    plt.ylabel('density')
    plt.xlim((-0.5, 100.5))
    plt.ylim((-0.01, gc_binned['mean'].max() * 1.1))

    plt.tight_layout()

    fig.savefig(plot_filename, format='pdf', bbox_inches='tight')

    gc_binned[['smoothed']].to_csv(gc_dist_filename, sep='\t', index=False, header=False)


def prepare_counts(segments_filename, alleles_filename, count_filename):

    segment_data = pd.read_csv(segments_filename, sep='\t', header=None,
                                converters={'segment_id':str, 'chromosome_1':str, 'chromosome_2':str},
                                names=['segment_id', 'chromosome_1', 'position_1', 'strand_1',
                                       'chromosome_2', 'position_2', 'strand_2',
                                       'readcount', 'length'])

    allele_data = pd.read_csv(alleles_filename, sep='\t', header=None,
                              names=['segment_id', 'hap_label', 'allele_id', 'readcount', 'is_allele_a'])

    # Calculate allele a/b readcounts
    allele_data = allele_data.set_index(['segment_id', 'hap_label', 'is_allele_a'])['readcount'].unstack().fillna(0.0)
    allele_data = allele_data.astype(int)
    allele_data = allele_data.rename(columns={0:'allele_b_readcount', 1:'allele_a_readcount'})

    # Merge haplotype blocks contained within the same segment
    allele_data = allele_data.groupby(level=[0])[['allele_a_readcount', 'allele_b_readcount']].sum()

    # Calculate major and minor readcounts, and relationship to allele a/b
    allele_data['major_readcount'] = allele_data[['allele_a_readcount', 'allele_b_readcount']].apply(max, axis=1)
    allele_data['minor_readcount'] = allele_data[['allele_a_readcount', 'allele_b_readcount']].apply(min, axis=1)
    allele_data['major_is_allele_a'] = (allele_data['major_readcount'] == allele_data['allele_a_readcount']) * 1

    # Merge allele data with segment data
    segment_data = segment_data.merge(allele_data, left_on='segment_id', right_index=True)

    segment_data.to_csv(count_filename, sep='\t', index=False, header=True)

