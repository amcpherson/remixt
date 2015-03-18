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

import demix
import demix.seqdataio
import demix.segalg
import demix.utils
import demix.analysis.haplotype
import demix.analysis.segment


demix_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
bin_directory = os.path.join(demix_directory, 'bin')
default_config_filename = os.path.join(demix_directory, 'defaultconfig.py')


if __name__ == '__main__':

    import prepare_counts
    
    argparser = argparse.ArgumentParser()

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('segment_file',
        help='Input segments file')

    argparser.add_argument('normal_file',
        help='Input normal sequence data filename')

    argparser.add_argument('--tumour_files', nargs='+', required=True,
        help='Input tumour sequence data filenames')

    argparser.add_argument('--count_files', nargs='+', required=True,
        help='Output count TSV filenames')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    if len(args['tumour_files']) != len(args['count_files']):
        raise Exception('--count_files must correspond one to one with --tumour_files')

    config = {'ref_data_directory':args['ref_data_dir']}
    execfile(default_config_filename, {}, config)

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([demix, prepare_counts], config)

    tumour_fnames = dict(enumerate(args['tumour_files']))
    count_fnames = dict(enumerate(args['count_files']))

    pyp.sch.setobj(mgd.OutputChunks('bytumour'), tumour_fnames.keys())

    pyp.sch.transform('calc_fragment_stats', ('bytumour',), {'mem':16},
        prepare_counts.calculate_fragment_stats,
        mgd.TempOutputObj('fragstats', 'bytumour'),
        mgd.InputFile('tumour_file', 'bytumour', fnames=tumour_fnames),
    )

    pyp.sch.setobj(mgd.OutputChunks('bychromosome'), config['chromosomes'])

    pyp.sch.transform('infer_haps', ('bychromosome',), {'mem':16},
        demix.analysis.haplotype.infer_haps,
        None,
        mgd.TempOutputFile('haps.tsv', 'bychromosome'),
        mgd.InputFile(args['normal_file']),
        mgd.InputInstance('bychromosome'),
        mgd.TempFile('haplotyping'),
        config,
    )

    pyp.sch.transform('merge_haps', (), {'mem':16},
        demix.utils.merge_tables,
        None,
        mgd.TempOutputFile('haps.tsv'),
        mgd.TempInputFile('haps.tsv', 'bychromosome'),
    )

    pyp.sch.transform('create_readcounts', ('bytumour',), {'mem':16},
        prepare_counts.create_counts,
        None,
        mgd.TempOutputFile('segment_counts.tsv', 'bytumour'),
        mgd.TempOutputFile('allele_counts.tsv', 'bytumour'),
        mgd.InputFile(args['segment_file']),
        mgd.InputFile('tumour_file', 'bytumour', fnames=tumour_fnames),
        mgd.TempInputFile('haps.tsv'),
    )

    pyp.sch.transform('phase_segments', (), {'mem':16},
        prepare_counts.phase_segments,
        None,
        mgd.TempInputFile('allele_counts.tsv', 'bytumour'),
        mgd.TempOutputFile('phased_allele_counts.tsv', 'bytumour2'),
    )

    pyp.sch.changeaxis('phased_axis', (), 'phased_allele_counts.tsv', 'bytumour2', 'bytumour')

    pyp.sch.transform('sample_gc', ('bytumour',), {'mem':16},
        prepare_counts.sample_gc,
        None,
        mgd.TempOutputFile('gcsamples.tsv', 'bytumour'),
        mgd.InputFile('tumour_file', 'bytumour', fnames=tumour_fnames),
        mgd.TempInputObj('fragstats', 'bytumour').prop('fragment_mean'),
        config,
    )

    pyp.sch.transform('gc_lowess', ('bytumour',), {'mem':16},
        prepare_counts.gc_lowess,
        None,
        mgd.TempInputFile('gcsamples.tsv', 'bytumour'),
        mgd.TempOutputFile('gcloess.tsv', 'bytumour'),
        mgd.TempOutputFile('gcplots.pdf', 'bytumour'),
    )

    pyp.sch.commandline('gc_segment', ('bytumour',), {'mem':16},
        os.path.join(bin_directory, 'estimategc'),
        '-m', config['mappability_filename'],
        '-g', config['genome_fasta'],
        '-c', mgd.TempInputFile('segment_counts.tsv', 'bytumour'),
        '-i',
        '-o', '4',
        '-u', mgd.TempInputObj('fragstats', 'bytumour').prop('fragment_mean'),
        '-s', mgd.TempInputObj('fragstats', 'bytumour').prop('fragment_stddev'),
        '-a', config['mappability_length'],
        '-l', mgd.TempInputFile('gcloess.tsv', 'bytumour'),
        '>', mgd.TempOutputFile('segment_counts_lengths.tsv', 'bytumour'),
    )

    pyp.sch.transform('prepare_counts', ('bytumour',), {'mem':16},
        prepare_counts.prepare_counts,
        None,
        mgd.TempInputFile('segment_counts_lengths.tsv', 'bytumour'),
        mgd.TempInputFile('phased_allele_counts.tsv', 'bytumour2'),
        mgd.OutputFile('count_file', 'bytumour', fnames=count_fnames),
    )

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


def create_counts(segment_counts_filename, allele_counts_filename, segment_filename, seqdata_filename, haps_filename):

    segments = pd.read_csv(segment_filename, sep='\t', converters={'chromosome':str})

    segment_counts = demix.analysis.segment.create_segment_counts(
        segments,
        tumour_filename,
    )

    segment_counts.to_csv(segment_counts_filename, sep='\t', index=False)

    allele_counts = demix.analysis.haplotype.create_allele_counts(
        segments,
        tumour_filename,
        haplotype_filename,
    )

    allele_counts.to_csv(allele_counts_filename, sep='\t', index=False)


def phase_segments(allele_counts_filenames, phased_allele_counts_filename_callback):

    tumour_ids = allele_counts_filenames.keys()

    allele_count_tables = list()
    for allele_counts_filename in allele_counts_filenames.itervalues():
        allele_count_tables.append(pd.read_csv(allele_counts_filename, sep='\t', converters={'chromosome':str}))

    phased_allele_counts_tables = demix.analysis.haplotype.phase_segments(allele_count_tables)

    for tumour_id, phased_allele_counts in zip(tumour_ids, phased_allele_counts_tables):
        phased_allele_counts_filename = phased_allele_counts_filename_callback(tumour_id)
        phased_allele_counts.to_csv(phased_allele_counts_filename, sep='\t', index=False)


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

    segment_data = pd.read_csv(segments_filename, sep='\t', converters={'chromosome':str})
    allele_data = pd.read_csv(alleles_filename, sep='\t', converters={'chromosome':str})

    segment_allele_counts = demix.analysis.segment.create_segment_allele_counts(segment_data, allele_data)

    segment_allele_counts.to_csv(count_filename, sep='\t', index=False)



