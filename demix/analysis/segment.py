import pandas as pd
import numpy as np

import demix.seqdataio

def create_segment_counts(segment_count_filename, seqdata_filename, segments_filename, chromosome):
    """ Count reads falling entirely within segments

    Args:
        segment_count_file (str): output segment file with counts per segment
        seqdata_filename (str): input sequence data file
        segments_filename (str): input genomic segments
        chromosome (str): id of chromosome for which counts will be calculated

    The output segment counts will be in TSV format with an additional 'readcount' column
    for the number of counts per segment.

    """
    
    # Read segment data for selected chromosome
    segments = pd.read_csv(segments_filename, sep='\t', converters={'chromosome':str})
    segments = segments[segments['chromosome'] == chromosome]

    # Read read data for selected chromosome
    reads = next(demix.seqdataio.read_read_data(seqdata_filename, chromosome=chromosome))
        
    # Sort in preparation for search
    reads.sort('start', inplace=True)
    segments.sort('start', inplace=True)

     # Count segment reads
    segments['readcount'] = demix.segalg.contained_counts(
        segments[['start', 'end']].values,
        reads[['start', 'end']].values
    )

    segments.to_csv(segment_count_filename, sep='\t', index=False)


def create_segment_allele_counts(segment_allele_count_filename, segment_count_filename, phased_allele_count_filename):
    """
    
    """

    segment_data = pd.read_csv(segment_count_filename, sep='\t', converters={'chromosome':str})

    allele_data = pd.read_csv(phased_allele_count_filename, sep='\t', converters={'chromosome':str})

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

    segment_data.to_csv(segment_allele_count_filename, sep='\t', index=False)

