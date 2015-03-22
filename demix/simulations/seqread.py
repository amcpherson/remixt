import numpy as np
import pandas as pd

import demix.seqdataio


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


def vrange(starts, lengths):
    """ Create concatenated ranges of integers for multiple start/length

    Args:
        starts (numpy.array): starts for each range
        lengths (numpy.array): lengths for each range (same length as starts)

    Returns:
        numpy.array: concatenated ranges

    See the following illustrative example:

        starts = np.array([1, 3, 4, 6])
        lengths = np.array([0, 2, 3, 0])

        print vrange(starts, lengths)
        >>> [3 4 4 5 6]

    """
    
    # Repeat start position index length times and concatenate
    cat_start = np.repeat(starts, lengths)

    # Create group counter that resets for each start/length
    cat_counter = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)

    # Add group counter to group specific starts
    cat_range = cat_start + cat_counter

    return cat_range


def interval_position_overlap(intervals, positions):
    """ Map intervals to contained positions

    Args:
        intervals (numpy.array): start and end of intervals with shape (N,2) for N intervals
        positions (numpy.array): positions, length M, must be sorted

    Returns:
        numpy.array: interval index, length L (arbitrary)
        numpy.array: position index, length L (same as interval index)

    Given a set of possibly overlapping intervals, create a mapping of positions that are contained
    within those intervals.

    """

    # Search for start and end of each interval in list of positions
    start_pos_idx = np.searchsorted(positions, intervals[:,0])
    end_pos_idx = np.searchsorted(positions, intervals[:,1])

    # Calculate number of positions for each segment
    lengths = end_pos_idx - start_pos_idx

    # Interval index for mapping
    interval_idx = np.repeat(np.arange(len(lengths)), lengths)

    # Position index for mapping 
    position_idx = vrange(start_pos_idx, lengths)

    return interval_idx, position_idx


def read_snp_overlap(data, snps, read_length):
    """ Calculate read snp overlap table

    Args:
        data (pandas.DataFrame): fragments
        snps (dict of pandas.DataFrame): snp positions by chromosome
        read_length (int): read length

    Returns:
        pandas.DataFrame

    Input 'fragments' dataframe has columns 'chromosome', 'fragment_id', 'allele', 'start', 'end'.
    The 'fragment_id' should be unique per chromosome.

    Input 'snps' dict is keyed by 'chromosome', values are dataframes with columns 'position',
    'is_alt_0', 'is_alt_1', sorted by 'position', 0..N-1 indexed

    Output dataframe has columns 'chromosome', 'position', 'fragment_id', 'is_alt'.
    The 'fragment_id' column is a foreign key indexing entries in the input data dataframe.

    No sorting assumptions are made.

    """

    allele_data = list()

    for chromosome, chrom_fragments in data.groupby('chromosome'):

        chrom_fragments = chrom_fragments.reset_index(drop=True)

        # Postion data
        chrom_snps = snps[chromosome]

        # Overlap snp positions and fragment intervals
        fragment_idx, snp_idx = interval_position_overlap(
            chrom_fragments[['start', 'end']].values,
            chrom_snps['position'].values,
        )

        # Create fragment snp table
        fragment_snps = pd.DataFrame({'snp_idx':snp_idx}, index=fragment_idx)
        fragment_snps = fragment_snps.merge(chrom_fragments, left_index=True, right_index=True)
        fragment_snps = fragment_snps.merge(chrom_snps, left_on='snp_idx', right_index=True)

        # Keep only snps falling within reads
        fragment_snps = fragment_snps[
            (fragment_snps['position'] < fragment_snps['start'] + read_length) |
            (fragment_snps['position'] >= fragment_snps['end'] - read_length)
        ]

        # Calculate whether the snp is the alternate based on the allele of the fragment and the genotype
        fragment_snps['is_alt'] = np.where(
            fragment_snps['allele'] == 0,
            fragment_snps['is_alt_0'],
            fragment_snps['is_alt_1'],
        )

        allele_data.append(fragment_snps[['chromosome', 'position', 'fragment_id', 'is_alt']])

    allele_data = pd.concat(allele_data, ignore_index=True)

    return allele_data


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


def simulate_mixture_read_data(read_data_filename, genomes, read_depths, snps, temp_dir, params):
    """ Simulate read data from a mixture of genomes.

    Args:
        read_data_filename (str): file to which read data is written
        genomes (list of RearrangedGenome): rearranged genomes in mixture
        read_depths (list of float): read depths of rearranged genomes
        snps (pandas.DataFrame): snp position data
        temp_dir (str): location to write temporary files
        params (dict): dictionary of simulation parameters

    Input 'snps' dataframe has columns 'chromosome', 'position', 'is_alt_0', 'is_alt_1'.

    """

    snps = dict(list(snps.groupby('chromosome')))
    for chromosome, chrom_snps in snps.iteritems():
        chrom_snps.reset_index(drop=True, inplace=True)
        chrom_snps.drop('chromosome', axis=1, inplace=True)

    with demix.seqdataio.Writer(read_data_filename, temp_dir) as w:

        for genome, read_depth in zip(genomes, read_depths):

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

                # Remapped fragments
                fragments_remapped = np.zeros((fragment_start.shape[0], 2))

                # Remap start to reference genome
                segment_idx, fragments_remapped[:,0] = segment_remap(
                    segment_data[['start', 'end']].values,
                    fragment_start,
                )

                # Remap end to reference genome
                end_segment_idx, fragments_remapped[:,1] = segment_remap(
                    segment_data[['start', 'end']].values,
                    fragment_start + fragment_length,
                )

                # Filter discordant
                is_concordant = (fragments_remapped[:,1] - fragments_remapped[:,0]) == fragment_length
                fragments_remapped = fragments_remapped[is_concordant,:]
                segment_idx = segment_idx[is_concordant]

                # Negate and flip start and end for reversed fragments
                fragments_remapped = np.absolute(fragments_remapped)
                fragments_remapped.sort(axis=1)

                # Fragment data mapped to concatenated reference genome
                remapped_data = pd.DataFrame({'start':fragments_remapped[:,0], 'end':fragments_remapped[:,1]}, index=segment_idx)

                # Merge chromosome, allele
                remapped_data = remapped_data.merge(
                    segment_data[['chromosome', 'allele']],
                    left_index=True,
                    right_index=True,
                )

                # Sort by chromosome
                remapped_data.sort('chromosome', inplace=True)
                remapped_data.reset_index(drop=True, inplace=True)

                # Add fragment id, unique per chromosome
                counts = remapped_data.groupby('chromosome', sort=False).size().values
                remapped_data['fragment_id'] = np.arange(counts.sum()) - np.repeat(counts.cumsum() - counts, counts)

                # Create table of read pairs overlapping snp positions
                allele_data = read_snp_overlap(remapped_data, snps, params['read_length'])

                # Random base calling errors at snp positions
                base_call_error = np.random.choice(
                    [True, False],
                    size=len(allele_data.index),
                    p=[params['base_call_error'], 1. - params['base_call_error']]
                )
                allele_data['is_alt'] = np.where(base_call_error, 1-allele_data['is_alt'], allele_data['is_alt'])

                # Write out a chunk of data
                w.write(remapped_data, allele_data)

                num_fragments_created += len(remapped_data.index)


