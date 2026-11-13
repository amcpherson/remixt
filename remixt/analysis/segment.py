import pandas as pd
import numpy as np

import remixt.defaults
import remixt.seqdataio
import remixt.segalg
import remixt.utils
import remixt.analysis.binbias


def create_segments(
    segment_filename,
    genome_fai_filename,
    gap_table_filename,
    chromosomes=remixt.defaults.chromosomes,
    chr_name_prefix=remixt.defaults.chr_name_prefix,
    segment_length=remixt.defaults.segment_length,
    breakpoint_filename=None,
):
    """ Create segments file based on breakpoints and regular segmentation.

    Args:
        segment_filename (str): file to which segments will be written
        genome_fai_filename (str): reference genome .fai file
        gap_table_filename (str): ucsc gap table

    KwArgs:
        chromosomes (list of str): chromosomes to segment
        chr_name_prefix (str): '' or 'chr', checked against chromosome names
        segment_length (int): length of regular segments
        breakpoint_filename (str): file containing breakpoints

    """

    chromosome_lengths = remixt.utils.read_selected_chromosome_lengths(
        genome_fai_filename, chromosomes, chr_name_prefix=chr_name_prefix)
    chromosomes = list(chromosome_lengths.keys())

    gap_table_columns = [
        'bin',
        'chromosome',
        'start',
        'end',
        'ix',
        'n',
        'size',
        'type',
        'bridge',
    ]

    gap_table = pd.read_csv(
        gap_table_filename, sep='\t', compression='gzip', header=None,
        names=gap_table_columns, converters={'chromosome': str})

    if chr_name_prefix == 'chr':
        assert gap_table['chromosome'].str.startswith('chr').all()
    elif chr_name_prefix == '':
        assert not gap_table['chromosome'].str.startswith('chr').any()
    else:
        raise ValueError(f'unrecognized chr_name_prefix {chr_name_prefix}')

    changepoints = list()

    # Add regular segments
    for chromosome in chromosomes:
        length = chromosome_lengths[chromosome]
        for position in np.arange(0, length, segment_length, dtype=int):
            changepoints.append((chromosome, position))
        changepoints.append((chromosome, length))

    # Add gap boundaries to changepoints
    for idx in gap_table.index:
        changepoints.append((gap_table.loc[idx, 'chromosome'], gap_table.loc[idx, 'start']))
        changepoints.append((gap_table.loc[idx, 'chromosome'], gap_table.loc[idx, 'end']))

    # Add breakends to segmentation if provided
    if breakpoint_filename is not None:
        breakpoints = pd.read_csv(
            breakpoint_filename, sep='\t',
            converters={'chromosome_1': str, 'chromosome_2': str, 'position_1': int, 'position_2': int}
        )

        if chr_name_prefix == 'chr':
            assert breakpoints['chromosome_1'].str.startswith('chr').all()
            assert breakpoints['chromosome_2'].str.startswith('chr').all()
        elif chr_name_prefix == '':
            assert not breakpoints['chromosome_1'].str.startswith('chr').any()
            assert not breakpoints['chromosome_2'].str.startswith('chr').any()
        else:
            raise ValueError(f'unrecognized chr_name_prefix {chr_name_prefix}')

        for idx, row in breakpoints.iterrows():
            changepoints.append((row['chromosome_1'], row['position_1']))
            changepoints.append((row['chromosome_2'], row['position_2']))

    changepoints = pd.DataFrame(changepoints, columns=['chromosome', 'position'])
    changepoints.sort_values(['chromosome', 'position'], inplace=True)

    # Create segments from changepoints
    segments = list()
    for chromosome, chrom_changepoints in changepoints.groupby('chromosome'):
        chrom_segments = pd.DataFrame({
            'start': chrom_changepoints['position'].values[:-1],
            'end': chrom_changepoints['position'].values[1:],
        })
        chrom_segments['chromosome'] = chromosome
        segments.append(chrom_segments)
    segments = pd.concat(segments, ignore_index=True)

    # Remove gap segments
    segments['gap'] = False
    for idx in gap_table.index:
        gap_chromosome = gap_table.loc[idx, 'chromosome']
        gap_start = gap_table.loc[idx, 'start']
        gap_end = gap_table.loc[idx, 'end']
        segments.loc[
            (segments['chromosome'] == gap_chromosome) &
            (segments['start'] >= gap_start) &
            (segments['start'] < gap_end),
            'gap'
        ] = True
    segments = segments[~segments['gap']]

    # Remove 0 lengthed segments
    segments = segments[segments['start'] < segments['end']]

    # Sort segments by placement in chromosome list, and position
    segments = segments.merge(pd.DataFrame(list(enumerate(chromosomes)), columns=['chromosome_idx', 'chromosome']))
    segments.sort_values(['chromosome_idx', 'start'], inplace=True)

    segments.to_csv(segment_filename, sep='\t', index=False, columns=['chromosome', 'start', 'end'])


def count_segment_reads(seqdata_filename, chromosome, segments, filter_duplicates=False, map_qual_threshold=1, mask=None):
    """ Count reads falling entirely within segments on a specific chromosome

    Args:
        seqdata_filename (str): input sequence data file
        chromosome (str): chromosome for which to count reads
        segments (str): segments for which to count reads

    KwArgs:
        filter_duplicates (bool): filter reads marked as duplicate
        map_qual_threshold (int): filter reads with less than this mapping quality
        mask (pandas.DataFrame): masked regions, fragments with a masked midpoint are not counted

    Returns:
        pandas.DataFrame: output segment data

    Input segments should have columns 'start', 'end'.

    The output segment counts will be in TSV format with an additional 'readcount' column
    for the number of counts per segment.

    """

    # Read fragment data with filtering, sorted by start, without masked fragments
    reads = remixt.analysis.binbias.read_fragment_midpoints(
        seqdata_filename, chromosome,
        filter_duplicates=filter_duplicates,
        map_qual_threshold=map_qual_threshold,
        mask=mask,
    )

    # Sort in preparation for search
    segments.sort_values('start', inplace=True)
    
    # Count segment reads
    segments['readcount'] = remixt.segalg.contained_counts(
        segments[['start', 'end']].values,
        reads[['start', 'end']].values
    )

    # Sort on index to return dataframe in original order
    segments.sort_index(inplace=True)

    return segments


def create_segment_counts(segments, seqdata_filename, filter_duplicates=False, map_qual_threshold=1, mask=None):
    """ Create a table of read counts for segments

    Args:
        segments (pandas.DataFrame): input segment data
        seqdata_filename (str): input sequence data file

    KwArgs:
        filter_duplicates (bool): filter reads marked as duplicate
        map_qual_threshold (int): filter reads with less than this mapping quality
        mask (pandas.DataFrame): masked regions, fragments with a masked midpoint are not counted

    Returns:
        pandas.DataFrame: output segment data

    Input segments should have columns 'chromosome', 'start', 'end'.

    The output segment counts will be in TSV format with an additional 'readcount' column
    for the number of counts per segment.

    """

    # Count separately for each chromosome, ensuring order is preserved for groups
    gp = segments.groupby('chromosome')

    # Table of read counts, calculated for each group
    counts = list()
    for chrom, segs in gp:
        counts.append(count_segment_reads(
            seqdata_filename, chrom, segs.copy(),
            filter_duplicates=filter_duplicates,
            map_qual_threshold=map_qual_threshold,
            mask=mask))
    counts = pd.concat(counts)

    # Sort on index to return dataframe in original order
    counts.sort_index(inplace=True)

    return counts


def create_segment_allele_counts(segment_data, allele_data):
    """ Create a table of total and allele specific segment counts

    Args:
        segment_data (pandas.DataFrame): counts of reads in segments
        allele_data (pandas.DataFrame): counts of reads in segment haplotype blocks with phasing

    Returns:
        pandas.DataFrame: output segment data

    Input segment_counts table is expected to have columns 'chromosome', 'start', 'end', 'readcount'.

    Input phased_allele_counts table is expected to have columns 'chromosome', 'start', 'end', 
    'hap_label', 'is_allele_a', 'readcount'.

    Output table will have columns 'chromosome', 'start', 'end', 'readcount', 'major_readcount',
    'minor_readcount', 'major_is_allele_a'
    
    """

    # Calculate allele a/b readcounts
    allele_data = (
        allele_data
        .set_index(['chromosome', 'start', 'end', 'hap_label', 'is_allele_a'])['readcount']
        .unstack(fill_value=0)
        .reindex(columns=[0, 1])
        .fillna(0.0)
        .astype(int)
        .rename(columns={0: 'allele_b_readcount', 1: 'allele_a_readcount'})
    )

    # Merge haplotype blocks contained within the same segment
    allele_data = allele_data.groupby(level=[0, 1, 2])[['allele_a_readcount', 'allele_b_readcount']].sum()

    # Reindex and fill with 0
    allele_data = allele_data.reindex(segment_data.set_index(['chromosome', 'start', 'end']).index, fill_value=0)

    # Calculate major and minor readcounts, and relationship to allele a/b
    allele_data['major_readcount'] = allele_data[['allele_a_readcount', 'allele_b_readcount']].apply(max, axis=1)
    allele_data['minor_readcount'] = allele_data[['allele_a_readcount', 'allele_b_readcount']].apply(min, axis=1)
    allele_data['major_is_allele_a'] = (allele_data['major_readcount'] == allele_data['allele_a_readcount']) * 1

    # Merge allele data with segment data
    segment_data = segment_data.merge(allele_data, left_on=['chromosome', 'start', 'end'], right_index=True)

    return segment_data


