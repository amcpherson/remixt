import collections
import numpy as np
import pandas as pd

import remixt.segalg
import remixt.seqdataio


def simulate_fragment_intervals(genome_length, num_fragments, read_length, fragment_mean, fragment_stddev):
    """ Simulate sequenced fragments as intervals of a genome with given length

    Args:
        genome_length (int): genome length
        num_fragments (float): approximate number of fragments to create
        read_length (int): length of paired reads
        fragment_mean (float): mean of fragment length distribution
        fragment_stddev (float): standard deviation of fragment length distribution

    Returns:
        numpy.array: fragment start
        numpy.array: fragment length

    Returned dataframe has columns: 'start', 'end'

    Sample starting points uniformly across the genome, and sample lengths from a normal
    distribution with specified mean and standard deviation.  Filter fragments that are
    shorter than the read length.

    """

    # Uniformly random start and normally distributed length
    start = np.sort(np.random.randint(0, high=genome_length, size=num_fragments))
    length = (fragment_stddev * np.random.randn(num_fragments) + fragment_mean).astype(int)

    # Filter fragments shorter than the read length
    is_filtered = (length < read_length) | (start + length >= genome_length)
    start = start[~is_filtered]
    length = length[~is_filtered]

    return start, length


def segment_remap(segments, positions):
    """ Remap positions in a set of ordered segments

    Args:
        segments (numpy.array): start and end of segments with shape (N,2) for N segments
        positions (numpy.array): positions mapped relative to concatenated segments, length M

    Returns:
        numpy.array: segment index for each position with length M
        numpy.array: remapped position within segment with length M

    Given a set of positions mapped to a region of length L, remap those positions to
    a segmentation of [0,L), where the segmentation defines a mapping from non-overlapping
    segments of [0,L) to any other equal sized segment.  The segmentation is given as a segment
    of start and end positions with lengths that sum to L.

    """

    # Calculate mapping
    seg_length = segments[:,1] - segments[:,0]
    remap_end = seg_length.cumsum()
    remap_start = remap_end - seg_length

    # Check for positions outside [0,L)
    if np.any(positions > seg_length.sum()):
        raise ValueError('positions should be less than total segment length')

    # Calculate index of segment containing position
    pos_seg_idx = np.searchsorted(
        remap_end,
        positions,
        side='right',
    )

    # Calculate the remapped position based on the segment start
    remap_pos = segments[pos_seg_idx,0] + positions - remap_start[pos_seg_idx]

    return pos_seg_idx, remap_pos


def _create_segment_table(genome):
    # Create a table of segment info
    segment_data = list()

    for tmr_chrom_idx, tmr_chrom in enumerate(genome.chromosomes):

        for (segment_idx, allele_id), orientation in tmr_chrom:

            chrom_id = genome.segment_chromosome_id[segment_idx]
            start = genome.segment_start[segment_idx]
            end = genome.segment_end[segment_idx]
            length = int(genome.l[segment_idx])

            segment_data.append((
                tmr_chrom_idx,
                chrom_id,
                start,
                end,
                allele_id,
                orientation,
                length,
            ))

    segment_data_cols = [
        'tmr_chrom',
        'chromosome',
        'start',
        'end',
        'allele',
        'orientation',
        'length',
    ]

    segment_data = pd.DataFrame(segment_data, columns=segment_data_cols)

    return segment_data


def simulate_mixture_read_data(read_data_filename, genomes, read_depths, snps, params):
    """ Simulate read data from a mixture of genomes.

    Args:
        read_data_filename (str): file to which read data is written
        genomes (list of RearrangedGenome): rearranged genomes in mixture
        read_depths (list of float): read depths of rearranged genomes
        snps (pandas.HDFStore): snp position data
        params (dict): dictionary of simulation parameters

    """

    writer = remixt.seqdataio.Writer(read_data_filename)

    # Start of unique index for fragments, per chromosome
    chromosome_fragment_id_start = collections.Counter()

    for genome, read_depth in zip(genomes, read_depths):
        segment_data = _create_segment_table(genome)

        # Negate and flip remapped start and end for reverse orientation segments
        rev_mask = segment_data['orientation'] != 1
        rev_cols = ['start', 'end']
        segment_data.loc[rev_mask,rev_cols] = -segment_data.loc[rev_mask,rev_cols[::-1]].values

        # Calculate number of reads from this genome
        tumour_genome_length = segment_data['length'].sum()
        num_fragments = int(tumour_genome_length * read_depth)

        # Create chunks of fragments to reduce memory usage
        num_fragments_created = 0
        fragments_per_chunk = 40000000
        while num_fragments_created < num_fragments:

            # Sample fragment intervals from concatenated tumour genome, sorted by start
            fragment_start, fragment_length = simulate_fragment_intervals(
                tumour_genome_length,
                min(fragments_per_chunk, num_fragments - num_fragments_created),
                params['read_length'],
                params['fragment_mean'],
                params['fragment_stddev'],
            )
            fragment_data = pd.DataFrame({'start':fragment_start, 'length':fragment_length})

            # Remap end to reference genome
            fragment_data['segment_idx'], fragment_data['end'] = segment_remap(
                segment_data[['start', 'end']].values,
                fragment_data['start'] + fragment_data['length'],
            )

            # Remap start to reference genome, overwrite start
            fragment_data['segment_idx'], fragment_data['start'] = segment_remap(
                segment_data[['start', 'end']].values,
                fragment_data['start'],
            )

            # Filter discordant
            fragment_data = fragment_data[(fragment_data['end'] - fragment_data['start']) == fragment_data['length']]

            # Negate and flip start and end for reversed fragments
            fragment_data['start'] = np.where(
                fragment_data['start'] < 0,
                -fragment_data['start'] - fragment_data['length'],
                fragment_data['start'],
            )
            fragment_data['end'] = fragment_data['start'] + fragment_data['length']
            fragment_data.drop('length', axis=1, inplace=True)

            # Segment merge/groupby operations require segment_idx index
            fragment_data.set_index('segment_idx', inplace=True)

            # Add allele to fragment data table
            fragment_data['allele'] = segment_data['allele'].reindex(fragment_data.index)

            # Group by chromosome
            fragment_data = dict(list(fragment_data.groupby(segment_data['chromosome'])))

            # Overlap with SNPs and output per chromosome
            for chromosome, chrom_fragments in fragment_data.iteritems():

                # Reindex for subsequent index based merge
                chrom_fragments.reset_index(drop=True, inplace=True)
                chrom_fragments['fragment_id'] = xrange(len(chrom_fragments.index))
                chrom_fragments['fragment_id'] += chromosome_fragment_id_start[chromosome]
                chromosome_fragment_id_start[chromosome] += len(chrom_fragments.index)

                # Position data
                chrom_snps = snps['/chromosome_{}'.format(chromosome)]

                # Overlap snp positions and fragment intervals
                fragment_idx, snp_idx = remixt.segalg.interval_position_overlap(
                    chrom_fragments[['start', 'end']].values,
                    chrom_snps['position'].values,
                )

                # Create fragment snp table
                fragment_snps = pd.DataFrame({'snp_idx':snp_idx, 'fragment_idx':fragment_idx})
                fragment_snps = fragment_snps.merge(chrom_fragments, left_on='fragment_idx', right_index=True)
                fragment_snps = fragment_snps.merge(chrom_snps, left_on='snp_idx', right_index=True)

                # Keep only snps falling within reads
                fragment_snps = fragment_snps[
                    (fragment_snps['position'] < fragment_snps['start'] + params['read_length']) |
                    (fragment_snps['position'] >= fragment_snps['end'] - params['read_length'])
                ]

                # Calculate whether the snp is the alternate based on the allele of the fragment and the genotype
                fragment_snps['is_alt'] = np.where(
                    fragment_snps['allele'] == 0,
                    fragment_snps['is_alt_0'],
                    fragment_snps['is_alt_1'],
                )

                # Random base calling errors at snp positions
                base_call_error = np.random.choice(
                    [True, False],
                    size=len(fragment_snps.index),
                    p=[params['base_call_error'], 1. - params['base_call_error']]
                )
                fragment_snps['is_alt'] = np.where(base_call_error, 1-fragment_snps['is_alt'], fragment_snps['is_alt'])

                # Write out a chunk of data
                writer.write(chromosome, chrom_fragments, fragment_snps)

                num_fragments_created += len(chrom_fragments.index)

    writer.close()


def resample_mixture_read_data(read_data_filename, source_filename, genomes, read_depths, snps, params):
    """ Simulate read data from a mixture of genomes.

    Args:
        read_data_filename (str): file to which read data is written
        source_filename (str): file from which read data is resampled
        genomes (list of RearrangedGenome): rearranged genomes in mixture
        read_depths (list of float): read depths of rearranged genomes
        snps (pandas.HDFStore): snp position data
        params (dict): dictionary of simulation parameters

    """

    segment_data = []

    for genome, read_depth in zip(genomes, read_depths):
        genome_segment_data = _create_segment_table(genome)

        # Calculate number of reads for each segment
        genome_segment_data['num_reads'] = genome_segment_data['length'] * read_depth

        segment_data.append(genome_segment_data)

    segment_data = pd.concat(segment_data, ignore_index=True)

    # Sum number of reads for each segment allele
    read_info = segment_data.groupby(['chromosome', 'start', 'end', 'allele'])['num_reads'].sum().reset_index()

    writer = remixt.seqdataio.Writer(read_data_filename)

    # Start of unique index for fragments, per chromosome
    chromosome_fragment_id_start = collections.Counter()

    for chromosome, chrom_read_info in read_info.groupby('chromosome'):
        source_fragments = remixt.seqdataio.read_fragment_data(source_filename, chromosome)
        source_alleles = remixt.seqdataio.read_allele_data(source_filename, chromosome)

        # Ensure segments are sorted by start, and the index is 0..N-1
        chrom_segment_info = chrom_read_info[['start', 'end']].drop_duplicates().sort_values('start').reset_index(drop=True)
        chrom_segment_info['segment_idx'] = range(len(chrom_segment_info.index))

        # Annotate each fragment as belonging to a single segment
        source_fragments['segment_idx'] = remixt.segalg.find_contained_segments(
            chrom_segment_info[['start', 'end']].values, source_fragments[['start', 'end']].values)

        # Remove fragments not fully contained in segments
        source_fragments = source_fragments[source_fragments['segment_idx'] >= 0]

        # Annotate chrom read info table with segment index
        chrom_read_info = chrom_read_info.merge(chrom_segment_info)

        # Select fragments for each segment allele
        sampled_fragments = []
        for idx in chrom_read_info.index:
            allele = chrom_read_info.loc[idx, 'allele']
            num_reads = chrom_read_info.loc[idx, 'num_reads']

            # Sample required number of reads for this segment with replacement
            source_segment_fragments = source_fragments[source_fragments['segment_idx'] == segment_idx]
            sampled_segment_fragments = source_segment_fragments.loc[np.random.choice(source_segment_fragments.index, size=num_reads, replace=True)]
            sampled_segment_fragments['allele'] = allele

            sampled_fragments.append(sampled_segment_fragments[['start', 'end', 'allele']])

        sampled_fragments = pd.concat(sampled_fragments, ignore_index=True)

        # Reindex for subsequent index based merge
        sampled_fragments.reset_index(drop=True, inplace=True)
        sampled_fragments['fragment_id'] = xrange(len(sampled_fragments.index))
        sampled_fragments['fragment_id'] += chromosome_fragment_id_start[chromosome]
        chromosome_fragment_id_start[chromosome] += len(sampled_fragments.index)

        # Position data
        chrom_snps = snps['/chromosome_{}'.format(chromosome)]

        # Overlap snp positions and fragment intervals
        fragment_idx, snp_idx = remixt.segalg.interval_position_overlap(
            sampled_fragments[['start', 'end']].values,
            chrom_snps['position'].values,
        )

        # Create fragment snp table
        fragment_snps = pd.DataFrame({'snp_idx':snp_idx, 'fragment_idx':fragment_idx})
        fragment_snps = fragment_snps.merge(sampled_fragments, left_on='fragment_idx', right_index=True)
        fragment_snps = fragment_snps.merge(chrom_snps, left_on='snp_idx', right_index=True)

        # Keep only snps falling within reads
        fragment_snps = fragment_snps[
            (fragment_snps['position'] < fragment_snps['start'] + params['read_length']) |
            (fragment_snps['position'] >= fragment_snps['end'] - params['read_length'])
        ]

        # Calculate whether the snp is the alternate based on the allele of the fragment and the genotype
        fragment_snps['is_alt'] = np.where(
            fragment_snps['allele'] == 0,
            fragment_snps['is_alt_0'],
            fragment_snps['is_alt_1'],
        )

        # Random base calling errors at snp positions
        base_call_error = np.random.choice(
            [True, False],
            size=len(fragment_snps.index),
            p=[params['base_call_error'], 1. - params['base_call_error']]
        )
        fragment_snps['is_alt'] = np.where(base_call_error, 1-fragment_snps['is_alt'], fragment_snps['is_alt'])

        # Write out a chunk of data
        writer.write(chromosome, sampled_fragments, fragment_snps)

        num_fragments_created += len(chrom_fragments.index)

    writer.close()

