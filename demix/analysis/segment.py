import pandas as pd
import numpy as np

import demix.seqdataio


def count_segment_reads(seqdata_filename, chromosome, segments):
    """ Count reads falling entirely within segments on a specific chromosome

    Args:
        seqdata_filename (str): input sequence data file
        chromosome (str): chromosome for which to count reads
        segments (str): segments for which to count reads

    Returns:
        pandas.DataFrame: output segment data

    Input segments should have columns 'start', 'end'.  The table should be sorted by 'start'.

    The output segment counts will be in TSV format with an additional 'readcount' column
    for the number of counts per segment.

    """

    # Read read data for selected chromosome
    reads = next(demix.seqdataio.read_read_data(seqdata_filename, chromosome=chromosome))
        
    # Sort in preparation for search
    reads.sort('start', inplace=True)

     # Count segment reads
    segments['readcount'] = demix.segalg.contained_counts(
        segments[['start', 'end']].values,
        reads[['start', 'end']].values
    )

    return segments


def create_segment_counts(segments, seqdata_filename):
    """ Create a table of read counts for segments

    Args:
        segments (pandas.DataFrame): input segment data
        seqdata_filename (str): input sequence data file

    Returns:
        pandas.DataFrame: output segment data

    Input segments should have columns 'chromosome', 'start', 'end'.

    The output segment counts will be in TSV format with an additional 'readcount' column
    for the number of counts per segment.

    """

    # Sort in preparation for search
    segments.sort(['chromosome', 'start'], inplace=True)

    # Count separately for each chromosome, ensuring order is preserved for groups
    gp = segments.groupby('chromosome', sort=False)

    # Table of read counts, calculated for each group
    counts = [count_segment_reads(seqdata_filename, *a) for a in gp]
    counts = pd.concat(counts, ignore_index=True)

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
    allele_data = allele_data.set_index(['chromosome', 'start', 'end', 'hap_label', 'is_allele_a'])['readcount'].unstack().fillna(0.0)
    allele_data = allele_data.astype(int)
    allele_data = allele_data.rename(columns={0:'allele_b_readcount', 1:'allele_a_readcount'})

    # Merge haplotype blocks contained within the same segment
    allele_data = allele_data.groupby(level=[0, 1, 2])[['allele_a_readcount', 'allele_b_readcount']].sum()

    # Calculate major and minor readcounts, and relationship to allele a/b
    allele_data['major_readcount'] = allele_data[['allele_a_readcount', 'allele_b_readcount']].apply(max, axis=1)
    allele_data['minor_readcount'] = allele_data[['allele_a_readcount', 'allele_b_readcount']].apply(min, axis=1)
    allele_data['major_is_allele_a'] = (allele_data['major_readcount'] == allele_data['allele_a_readcount']) * 1

    # Merge allele data with segment data
    segment_data = segment_data.merge(allele_data, left_on=['chromosome', 'start', 'end'], right_index=True)

    return segment_data


