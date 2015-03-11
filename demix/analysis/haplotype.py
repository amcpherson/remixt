import pandas as pd
import numpy as np

import demix.seqdataio

def infer_haps(haps_filename, seqdata_filename, chromosome, temp_directory, config):
    """ Infer haplotype blocks for a chromosome using shapeit

    Args:
        haps_filename (str): output haplotype data file
        seqdata_filename (str): input sequence data file
        chromosome (str): id of chromosome for which haplotype blocks will be inferred
        temp_directory (str): directory in which shapeit temp files will be stored
        config (dict): relavent shapeit parameters including thousand genomes paths

    The output haps file will contain haplotype blocks for each heterozygous SNP position. The
    file will be TSV format with the following columns:

        'chromosome': het snp chromosome
        'position': het snp position
        'allele': binary indicator for reference (0) vs alternate (1) allele
        'hap_label': label of the haplotype block
        'allele_id': binary indicator of the haplotype allele

    """
    
    def write_null():
        with open(haps_filename, 'w') as haps_file:
            haps_file.write('chromosome\tposition\tallele\thap_label\tallele_id\n')

    accepted_chromosomes = [str(a) for a in range(1, 23)] + ['X']
    if str(chromosome) not in accepted_chromosomes:
        write_null()
        return
    
    # Temporary directory for shapeit files
    try:
        os.makedirs(temp_directory)
    except OSError:
        pass

    # Thousand genomes snps
    snps_filename = config['snps_filename']

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

    # Create genotype file required by shapeit
    temp_gen_filename = os.path.join(temp_directory, 'snps.gen')
    snp_counts_df.reset_index(inplace=True)
    snp_counts_df['chr'] = chromosome
    snp_counts_df['chr_pos'] = snp_counts_df['chr'].astype(str) + ':' + snp_counts_df['position'].astype(str)
    snp_counts_df.to_csv(temp_gen_filename, sep=' ', cols=['chr', 'chr_pos', 'position', 'ref', 'alt', 'AA', 'AB', 'BB'], index=False, header=False)

    # Create single sample file required by shapeit
    temp_sample_filename = os.path.join(temp_directory, 'snps.sample')
    with open(temp_sample_filename, 'w') as temp_sample_file:
        temp_sample_file.write('ID_1 ID_2 missing sex\n0 0 0 0\nUNR1 UNR1 0 2\n')

    # Run shapeit to create phased haplotype graph
    hgraph_filename = os.path.join(temp_directory, 'phased.hgraph')
    hgraph_logs_prefix = hgraph_filename + '.log'
    chr_x_flag = ''
    if chromosome == 'X':
        chr_x_flag = '--chrX'
    pypeliner.commandline.execute('shapeit', '-M', genetic_map_filename, '-R', hap_filename, legend_filename, config['sample_filename'],
                                  '-G', temp_gen_filename, temp_sample_filename, '--output-graph', hgraph_filename, chr_x_flag,
                                  '--no-mcmc', '-L', hgraph_logs_prefix)

    # Run shapeit to sample from phased haplotype graph
    sample_template = os.path.join(temp_directory, 'sampled.{0}')
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

    haps['chromosome'] = chromosome

    haps = haps[['chromosome', 'position', 'allele', 'hap_label', 'allele_id']]

    haps.to_csv(haps_filename, sep='\t', index=False)


def create_allele_counts(allele_counts_filename, seqdata_filename, segments_filename, haps_filename, chromosome):
    """ Calculate read counts for haplotype alleles within segments

    Args:
        allele_counts_filename (str): output allele counts file
        seqdata_filename (str): input sequence data file
        segments_filename (str): input genomic segments
        haps_filename (str): input haplotype data file
        chromosome (str): id of chromosome for which counts will be calculated

    The output allele counts file will contain read counts for haplotype blocks within each segment.
    The file will be TSV format with the following columns:

        'chromosome': chromosome of the segment
        'start': start of the segment
        'end': end of the segment
        'hap_label': label of the haplotype block
        'allele_id': binary indicator of the haplotype allele
        'readcount': number of reads specific to haplotype block allele

    """
    
    # Read segment data for selected chromosome
    segments = pd.read_csv(segments_filename, sep='\t', converters={'chromosome':str})
    segments = segments[segments['chromosome'] == chromosome]

    # Read haplotype block data for selected chromosome
    haps = pd.read_csv(haps_filename, sep='\t', converters={'chromosome':str})
    haps = haps[haps['chromosome'] == chromosome]

    # Merge haplotype information into read alleles table
    alleles = list()
    for alleles_chunk in demix.seqdataio.read_allele_data(seqdata_filename, chromosome=chromosome, num_rows=10000):
        alleles_chunk = alleles_chunk.merge(haps, left_on=['position', 'is_alt'], right_on=['position', 'allele'], how='inner')
        alleles.append(alleles_chunk)
    alleles = pd.concat(alleles, ignore_index=True)

    # Arbitrarily assign a haplotype/allele label to each read
    alleles.drop_duplicates('fragment_id', inplace=True)

    # Sort in preparation for search
    segments.sort('start', inplace=True)
    alleles.sort('position', inplace=True)

    # Annotate segment for each allele
    alleles['segment_idx'] = demix.segalg.find_contained(
        segments[['start', 'end']].values,
        alleles['position'].values
    )

    # Merge segment start end, key for each segment (for given chromosome)
    alleles = alleles.merge(segments[['start', 'end']], left_on='segment_idx', right_index=True)

    # Count reads for each allele
    allele_counts = (alleles
        .set_index(['start', 'end', 'hap_label', 'allele_id'])
        .groupby(level=[0, 1, 2, 3])
        .size()
        .reset_index()
        .rename(columns={0:'readcount'})
    )

    # Add chromosome to output
    allele_counts['chromosome'] = chromosome

    # Write out allele counts
    allele_counts.to_csv(allele_counts_filename, sep='\t', 
        columns=['chromosome', 'start', 'end', 'hap_label', 'allele_id', 'readcount'],
        index=False)


def phase_segments(*args):
    """ Phase haplotype blocks within segments

    Takes 2n arguments for n tumours.  The first n arguments are input allele count files,
    the last n arguments are output phased allele count files.

    The output phased allele count file will contain read counts for haplotype blocks within
    each segment, and phasing information inferred for each block.  The file will be TSV
    format with the following columns:

        'segment_id': id of the segment
        'hap_label': label of the haplotype block
        'allele_id': binary indicator of the haplotype allele
        'readcount': number of reads specific to haplotype block allele
        'is_allele_a': indicator, is allele 'a' (1), is allele 'b' (0)

    """

    input_alleles_filenames = args[:len(args)/2]
    output_alleles_filenames = args[len(args)/2:]

    allele_phases = list()
    allele_diffs = list()

    for idx, input_alleles_filename in enumerate(input_alleles_filenames):

        allele_data = pd.read_csv(input_alleles_filename, sep='\t')
        
        # Allele readcount table
        allele_data = allele_data.set_index(['chromosome', 'start', 'end', 'hap_label', 'allele_id'])['readcount'].unstack().fillna(0.0)
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
        allele_diff = allele_data.groupby(level=[0, 1, 2])[['diff_readcount', 'total_readcount']].sum()
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
    segment_library = allele_diffs.set_index(['chromosome', 'start', 'end']).groupby(level=[0, 1, 2]).apply(select_largest_diff)
    segment_library.name = 'library_idx'
    segment_library = segment_library.reset_index()

    # For each haplotype block in each segment, take the major allele call of the library
    # with the largest major minor difference and call it allele 'a'
    allele_phases = allele_phases.merge(segment_library, left_on=['chromosome', 'start', 'end', 'library_idx'], right_on=['chromosome', 'start', 'end', 'library_idx'], how='right')
    allele_phases = allele_phases[['chromosome', 'start', 'end', 'hap_label', 'major_allele_id']].rename(columns={'major_allele_id': 'allele_a_id'})

    for idx, (input_alleles_filename, output_allele_filename) in enumerate(zip(input_alleles_filenames, output_alleles_filenames)):

        allele_data = pd.read_csv(input_alleles_filename, sep='\t')

        # Add a boolean column denoting which allele is allele 'a'
        allele_data = allele_data.merge(allele_phases, left_on=['chromosome', 'start', 'end', 'hap_label'], right_on=['chromosome', 'start', 'end', 'hap_label'])
        allele_data['is_allele_a'] = (allele_data['allele_id'] == allele_data['allele_a_id']) * 1
        allele_data = allele_data[['chromosome', 'start', 'end', 'hap_label', 'allele_id', 'readcount', 'is_allele_a']]

        allele_data.to_csv(output_allele_filename, sep='\t', index=False)

