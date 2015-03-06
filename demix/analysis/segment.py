

def create_segment_counts(segment_count_filename, seqdata_filename, segments_filename, chromosome):
    """ Count reads falling entirely within segments

    Args:
        segment_count_file (str): output segment file with counts per segment
        seqdata_filename (str): input sequence data file
        segments_filename (str): input genomic segments
        chromosome (str): id of chromosome for which counts will be calculated

    The output segment counts will be in TSV format with an additional 'count' column
    for the number of counts per segment.
    
    """
    
    # Read segment data for selected chromosome
    segments = pd.read_csv(segments_filename, sep='\t', converters={'chromosome':str})
    segments = segments[segments['chromosome'] == chromosome]

    # Read read data for selected chromosome
    reads = next(demix.seqdataio.read_read_data(seqdata_filename, chromosome=chromosome))
    reads.sort('start', inplace=True)
        
    # Create an index that matches the sort order
    segments.sort('start', inplace=True)
    segments.index = xrange(len(segments))

     # Count segment reads
    segments['count'] = demix.segalg.contained_counts(
        segments[['start', 'end']].values,
        reads[['start', 'end']].values
    )

    segments.to_csv(segment_filename, sep='\t', index=False, header=False)
