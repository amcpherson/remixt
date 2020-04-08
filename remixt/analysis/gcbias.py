import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats

import remixt.config
import remixt.segalg
import remixt.seqdataio
import remixt.utils


def sample_gc(gc_samples_filename, seqdata_filename, fragment_length, config, ref_data_dir):

    chromosomes = remixt.config.get_chromosomes(config, ref_data_dir)
    chromosome_lengths = remixt.config.get_chromosome_lengths(config, ref_data_dir)
    num_samples = remixt.config.get_param(config, 'sample_gc_num_positions')
    position_offset = remixt.config.get_param(config, 'gc_position_offset')
    genome_fasta = remixt.config.get_filename(config, ref_data_dir, 'genome_fasta')
    mappability_filename = remixt.config.get_filename(config, ref_data_dir, 'mappability')
    filter_duplicates = remixt.config.get_param(config, 'filter_duplicates')
    map_qual_threshold = remixt.config.get_param(config, 'map_qual_threshold')

    fragment_length = int(fragment_length)
    gc_window = fragment_length - 2 * position_offset

    chrom_info = pd.DataFrame({'chrom_length':chromosome_lengths})
    chrom_info['chrom_end'] = chrom_info['chrom_length'].cumsum()
    chrom_info['chrom_start'] = chrom_info['chrom_end'] - chrom_info['chrom_length']

    # Sample random genomic positions from concatenated genome
    genome_length = chrom_info['chrom_length'].sum()
    sample_pos = np.sort(np.random.randint(0, genome_length, num_samples))

    # Calculate GC/mappability for each position
    sample_gc_count = np.zeros(sample_pos.shape)
    sample_mappability = np.ones(sample_pos.shape)
    for chrom_id, sequence in remixt.utils.read_sequences(genome_fasta):

        # Ignore extraneous chromosomes
        if chrom_id not in chromosomes:
            continue

        # Read indicator of mappability based on threshold
        mappability = read_mappability_indicator(mappability_filename, chrom_id, len(sequence), map_qual_threshold)

        # Start and end of current chromosome in concatenated genome
        chrom_start, chrom_end = chrom_info.loc[chrom_id, ['chrom_start', 'chrom_end']].values

        # Calculate gc count within sliding window
        sequence = np.array(list(sequence.upper()))
        gc = ((sequence == 'G') | (sequence == 'C'))
        gc_count = gc.cumsum()
        gc_count[gc_window:] = gc_count[gc_window:] - gc.cumsum()[:-gc_window]

        # Append nan for fragments too close to the end of the chromosome
        gc_count = np.concatenate([gc_count, np.ones(fragment_length) * np.nan])

        # Calculate filter of positions in this chromosome
        chrom_sample_idx = (sample_pos >= chrom_start) & (sample_pos < chrom_end)

        # Calculate positions within this chromosome
        sample_chrom_pos = sample_pos[chrom_sample_idx] - chrom_start

        # Set the mappability indicator of the start positions of each read
        sample_mappability[chrom_sample_idx] *= mappability[sample_chrom_pos]

        # Calculate last position in window
        chrom_window_end = sample_chrom_pos + fragment_length - position_offset - 1

        # Add the gc count for filtered positions
        sample_gc_count[chrom_sample_idx] += gc_count[chrom_window_end]

    # Filter unmappable positions and nan gc count values
    sample_filter = ((sample_mappability > 0) & (~np.isnan(sample_gc_count)))
    sample_pos = sample_pos[sample_filter]
    sample_gc_count = sample_gc_count[sample_filter]

    sample_gc_percent = sample_gc_count / float(gc_window)

    # Count number of reads at each position
    sample_read_count = np.zeros(sample_pos.shape, dtype=int)
    for chrom_id in remixt.seqdataio.read_chromosomes(seqdata_filename):

        # Ignore extraneous chromosomes
        if chrom_id not in chromosomes:
            continue

        reads_iter = remixt.seqdataio.read_fragment_data(
            seqdata_filename, chrom_id,
            filter_duplicates=filter_duplicates,
            map_qual_threshold=map_qual_threshold,
            chunksize=1000000)

        for chrom_reads in reads_iter:

            # Calculate read start in concatenated genome
            chrom_reads['start'] += chrom_info.loc[chrom_id, 'chrom_start']

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
    sample_chrom_idx = np.searchsorted(chrom_info['chrom_end'].values, sample_pos, side='right')
    sample_chrom = chrom_info.index.values[sample_chrom_idx]
    sample_chrom_pos = sample_pos - chrom_info['chrom_start'].values[sample_chrom_idx]

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


def gc_lowess(gc_samples_filename, gc_dist_filename, gc_table_filename, gc_resolution=100):

    gc_samples = pd.read_csv(
        gc_samples_filename, sep='\t',
        names=['chromosome', 'position', 'gc', 'count'],
        converters={'chromosome': str},
    )

    gc_samples['gc_bin'] = (gc_samples['gc'] * gc_resolution).round()

    gc_binned = gc_samples.groupby('gc_bin')['count'] \
                          .agg([sum, len, np.mean]) \
                          .reindex(range(gc_resolution+1)) \
                          .fillna(0) \
                          .reset_index() \
                          .rename(columns={'index':'gc_bin'}) \
                          .astype(float)

    gc_binned['smoothed'] = sm.nonparametric.lowess(gc_binned['mean'].values, gc_binned['gc_bin'].values, frac=0.2).T[1]
    assert not gc_binned['smoothed'].isnull().any()

    rescale = 1. / gc_binned['smoothed'].max()

    gc_binned['mean'] = gc_binned['mean'] * rescale
    gc_binned['smoothed'] = gc_binned['smoothed'] * rescale

    gc_binned.to_csv(gc_table_filename, sep='\t', index=False)

    gc_binned[['smoothed']].to_csv(gc_dist_filename, sep='\t', index=False, header=False)


def read_mappability_indicator(mappability_filename, chromosome, max_chromosome_length, map_qual_threshold):
    """ Read a mappability wig file into a mappability vector
    """
    with pd.HDFStore(mappability_filename, 'r') as store:
        mappability_table = store.select('chromosome_'+chromosome, 'quality >= map_qual_threshold')

    mappability = np.zeros(max_chromosome_length, dtype=np.uint8)

    for start, end in mappability_table[['start', 'end']].values:
        end = min(end, max_chromosome_length)
        mappability[start:end] = 1

    return mappability


def read_gc_cumsum(genome_fasta, chromosome):
    """ Read a chromosome sequence and create GC cumulative sum

    TODO: optimize using genome fasta index
    """
    for c, s in remixt.utils.read_sequences(genome_fasta):
        if c == chromosome:
            s = np.array(list(s.upper()), dtype=np.character)
            gc_indicator = ((s == b'G') | (s == b'C')) * 1

    gc_cumsum = gc_indicator.cumsum()

    return gc_cumsum


class GCCurve(object):
    """ Piecewise linear GC probability curve
    """
    def read(self, gc_dist_filename):
        """ Read from a text file
        """
        with open(gc_dist_filename, 'r') as f:
            self.gc_lowess = np.array(f.readlines(), dtype=float)
        self.gc_lowess /= self.gc_lowess.sum()
        self.cache = {}

    def predict(self, x):
        """ Calculate GC probability from percent
        """
        idx = np.clip(int(x * float(len(self.gc_lowess) - 1)), 0, len(self.gc_lowess) - 1)
        return max(self.gc_lowess[idx], 0.0)

    def table(self, l):
        """ Tabulate GC probabilities for a specific fragment length
        """
        if l not in self.cache:
            self.cache[l] = np.array([self.predict(float(x)/float(l)) for x in range(0, l + 1)])
        return self.cache[l]


def gc_map_bias(segment_filename, fragment_mean, fragment_stddev, gc_dist_filename, bias_filename, config, ref_data_dir):
    """ Calculate per segment GC and mappability biases
    """
    segments = pd.read_csv(segment_filename, sep='\t', converters={'chromosome':str})

    biases = calculate_gc_map_bias(segments, fragment_mean, fragment_stddev, gc_dist_filename, config, ref_data_dir)

    biases.to_csv(bias_filename, sep='\t', index=False)


def calculate_gc_map_bias(segments, fragment_mean, fragment_stddev, gc_dist_filename, config, ref_data_dir):
    """ Calculate per segment GC and mappability biases
    """
    do_gc = remixt.config.get_param(config, 'do_gc_correction')
    do_map = remixt.config.get_param(config, 'do_mappability_correction')
    
    position_offset = remixt.config.get_param(config, 'gc_position_offset')
    genome_fasta = remixt.config.get_filename(config, ref_data_dir, 'genome_fasta')
    mappability_filename = remixt.config.get_filename(config, ref_data_dir, 'mappability')
    map_qual_threshold = remixt.config.get_param(config, 'map_qual_threshold')
    read_length = remixt.config.get_param(config, 'mappability_length')

    gc_dist = GCCurve()
    gc_dist.read(gc_dist_filename)

    fragment_dist = scipy.stats.norm(fragment_mean, fragment_stddev)

    fragment_min = int(fragment_dist.ppf(0.01) - 1.)
    fragment_max = int(fragment_dist.ppf(0.99) + 1.)
    fragment_step = 10

    for chromosome, chrom_seg in segments.groupby('chromosome', sort=False):
        gc_cumsum = read_gc_cumsum(genome_fasta, chromosome)
        chromosome_length = gc_cumsum.shape[0]
        mappability = read_mappability_indicator(mappability_filename, chromosome, chromosome_length, map_qual_threshold)

        for idx, (start, end) in chrom_seg[['start', 'end']].iterrows():
            segments.loc[idx, 'bias'] = calculate_segment_gc_map_bias(gc_cumsum[start:end], mappability[start:end],
                gc_dist, fragment_dist, fragment_min, fragment_max, fragment_step, position_offset, read_length,
                do_gc=do_gc, do_map=do_map)

    return segments


def calculate_segment_gc_map_bias(gc_cumsum, mappability, gc_dist, fragment_dist, fragment_min, fragment_max, fragment_step, position_offset, read_length,
        do_gc=True, do_map=True):
    """ Calculate GC/mappability bias
    """
    bias = 0.

    for fragment_length in range(fragment_min, fragment_max+1, fragment_step):
        if fragment_length < read_length:
            continue

        # Calculate gc sum per valid position
        gc_sum = gc_cumsum[fragment_length-position_offset:-position_offset] - gc_cumsum[position_offset:-fragment_length+position_offset]
        gc_length = fragment_length - 2*position_offset
        
        # Create a table mapping total GC to probability
        gc_table = gc_dist.table(gc_length)

        # Calculate per position GC probability
        gc_prob = gc_table[gc_sum]

        # Calculate mappability for read and mate at each valid position
        mate_position = fragment_length - read_length
        map_prob = mappability[:-fragment_length] * mappability[mate_position:-read_length]

        # Calculate fragment length prob
        len_prob = fragment_dist.pdf(fragment_length)
        
        # Default gc prob to 1
        if not do_gc:
            gc_prob = np.ones(gc_prob.shape)

        # Default mappability prob to 1
        if not do_map:
            map_prob = np.ones(map_prob.shape)

        # Calculate per position probability
        prob = len_prob * gc_prob * map_prob
        
        bias += prob.sum()

    return bias


def calculate_biased_length(segments):
    """ Calculate biased segment length.
    """
    # Normalize biases
    segments['bias'] /= segments['bias'].sum()

    # Calculate length as bias scaled by genome length
    segments['length'] = segments['bias'] * float((segments['end'] - segments['start']).sum())

    return segments


def biased_length(length_filename, bias_filename):
    """ Calculate biased segment length task.
    """
    segments = pd.read_csv(bias_filename, sep='\t', converters={'chromosome':str})
    segments = calculate_biased_length(segments)
    segments.to_csv(length_filename, sep='\t', index=False)



