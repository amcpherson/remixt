import numpy as np
import pandas as pd


def is_contained(a, b):
    """ Check if segment b is fully contained within segment a """
    return b[0] >= a[0] and b[1] <= a[1]


def contained_counts_unopt(X, Y):
    """ Find counts of overlapping segments fully contained in non-overlapping segments (unopt)

    Args:
        X (numpy.array): start and end of non-overlapping segments with shape (N,2) for N segments
        Y (numpy.array): start and end of overlapping segments with shape (M,2) for M segments

    Returns:
        numpy.array: N length array of counts of Y countained in X

    Both X and Y are assumed to be ordered by start position.

    """

    C = np.zeros(X.shape[0])
    y_idx = 0
    for x_idx, x in enumerate(X):
        while y_idx < Y.shape[0] and Y[y_idx][0] < x[0]:
            y_idx += 1
        while y_idx < Y.shape[0] and Y[y_idx][0] <= x[1]:
            if is_contained(x, Y[y_idx]):
                C[x_idx] += 1
            y_idx += 1
    return C


def contained_counts(X, Y):
    """ Find counts of overlapping segments fully contained in non-overlapping segments

    Args:
        X (numpy.array): start and end of non-overlapping segments with shape (N,2) for N segments
        Y (numpy.array): start and end of overlapping segments with shape (M,2) for M segments

    Returns:
        numpy.array: N length array of counts of Y countained in X

    X is assumed to be ordered by start position.

    """

    idx = np.searchsorted(X[:,1], Y[:,0])
    end_idx = np.searchsorted(X[:,1], Y[:,1])

    # Mask Y segments outside last X segment
    outside = end_idx >= X.shape[0]
    idx[outside] = 0

    # Filter for containment, same X segment, not outside
    idx = idx[
        (Y[:,0] >= X[idx,0]) &
        (Y[:,1] <= X[idx,1]) &
        (idx == end_idx) &
        (~outside)
    ]

    # Count number of Y in each X
    count = np.bincount(idx, minlength=X.shape[0])

    return count


def overlapping_counts(X, Y):
    """ Find counts of segments in Y overlapping positions in X
    X and Y are assume sorted, Y by starting position, X by position
    """
    C = np.zeros(X.shape[0])
    x_idx = 0
    for y in Y:
        while x_idx < X.shape[0] and X[x_idx] <= y[0]:
            x_idx += 1
        x_idx_1 = x_idx
        while x_idx_1 < X.shape[0] and X[x_idx_1] < y[1]:
            C[x_idx_1] += 1
            x_idx_1 += 1
    return C


def find_contained_positions_unopt(X, Y):
    """ Find mapping of positions contained within non-overlapping segments (unopt)

    Args:
        X (numpy.array): start and end of non-overlapping segments with shape (N,2) for N segments
        Y (numpy.array): positions with shape (M,) for M positions

    Returns:
        numpy.array: M length array of indices into X containing elements in Y

    X is assumed to be ordered by start position.

    For positions not contained within any segment, value in returned array will be -1

    """

    M = [-1]*Y.shape[0]
    for x_idx, x in enumerate(X):
        for y_idx, y in enumerate(Y):
            if Y[y_idx] >= x[0] and Y[y_idx] < x[1]:
                assert M[y_idx] == -1
                M[y_idx] = x_idx
    return M


def find_contained_positions(X, Y):
    """ Find mapping of positions contained within non-overlapping segments

    Args:
        X (numpy.array): start and end of non-overlapping segments with shape (N,2) for N segments
        Y (numpy.array): positions with shape (M,) for M positions

    Returns:
        numpy.array: M length array of indices into X containing elements in Y

    X is assumed to be ordered by start position.

    For positions not contained within any segment, value in returned array will be -1

    """

    # Positions less than segment end point
    idx = np.searchsorted(X[:, 1], Y, side='right')

    # Mask positions outside greatest endpoint
    mask = idx < X.shape[0]
    idx[~mask] = -1

    # Mask positions that are not fully contained within a segment
    mask = mask & (Y >= X[idx, 0]) & (Y < X[idx, 1])
    idx[~mask] = -1

    return idx


def find_contained_segments_unopt(X, Y):
    """ Find mapping of segments contained within non-overlapping segments (unopt)

    Args:
        X (numpy.array): start and end of non-overlapping segments with shape (N,2) for N segments
        Y (numpy.array): start and end of overlapping segments with shape (M,2) for M segments

    Returns:
        numpy.array: M length array of indices into X containing elements in Y

    X is assumed to be ordered by start position.

    For positions not contained within any segment, value in returned array will be -1

    """

    M = [-1]*Y.shape[0]
    for x_idx, x in enumerate(X):
        for y_idx, y in enumerate(Y):
            if Y[y_idx, 0] >= x[0] and Y[y_idx, 1] <= x[1]:
                assert M[y_idx] == -1
                M[y_idx] = x_idx
    return M


def find_contained_segments(X, Y):
    """ Find mapping of segments contained within non-overlapping segments

    Args:
        X (numpy.array): start and end of non-overlapping segments with shape (N,2) for N segments
        Y (numpy.array): start and end of overlapping segments with shape (M,2) for M segments

    Returns:
        numpy.array: M length array of indices into X containing elements in Y

    X is assumed to be ordered by start position.

    For positions not contained within any segment, value in returned array will be -1

    """

    # Y segment start greater than or equal to X segment start
    idx = np.searchsorted(X[:, 0], Y[:, 0], side='right') - 1

    # Y segment end less than or equal to X segment end
    idx_end = np.searchsorted(X[:, 1], Y[:, 1], side='left')

    # Mask positions outside greatest endpoint
    mask = idx == idx_end
    idx[~mask] = -1

    return idx


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


def reindex_segments(cn_1, cn_2):
    """ Reindex segment data to a common set of intervals

    Args:
        cn_1 (pandas.DataFrame): table of copy number
        cn_2 (pandas.DataFrame): another table of copy number

    Returns:
        pandas.DataFrame: reindex table

    Expected columns of input dataframe: 'chromosome', 'start', 'end'

    Output dataframe has columns 'chromosome', 'start', 'end', 'idx_1', 'idx_2'
    where 'idx_1', and 'idx_2' are the indexes into cn_1 and cn_2 of sub segments
    with the given chromosome, start, and end.

    """

    if len(cn_1.index) == 0 or len(cn_2.index) == 0:
        empty = pd.DataFrame(columns=['chromosome', 'start', 'end', 'idx_1', 'idx_2'], dtype=int)
        empty['chromosome'] = empty['chromosome'].astype(str)
        return empty

    reseg = list()

    for chromosome, chrom_cn_1 in cn_1.groupby('chromosome'):
        
        chrom_cn_2 = cn_2[cn_2['chromosome'] == chromosome]
        if len(chrom_cn_2.index) == 0:
            continue

        segment_boundaries = np.concatenate([
            chrom_cn_1['start'].values,
            chrom_cn_1['end'].values,
            chrom_cn_2['start'].values,
            chrom_cn_2['end'].values,
        ])

        segment_boundaries = np.sort(np.unique(segment_boundaries))

        chrom_reseg = pd.DataFrame({
            'start':segment_boundaries[:-1],
            'end':segment_boundaries[1:],
        })

        for suffix, chrom_cn in zip(('_1', '_2'), (chrom_cn_1, chrom_cn_2)):

            chrom_reseg['start_idx'+suffix] = np.searchsorted(
                chrom_cn['start'].values,
                chrom_reseg['start'].values,
                side='right',
            ) - 1

            chrom_reseg['end_idx'+suffix] = np.searchsorted(
                chrom_cn['end'].values,
                chrom_reseg['end'].values,
                side='left',
            )

            chrom_reseg['filter'+suffix] = (
                (chrom_reseg['start_idx'+suffix] != chrom_reseg['end_idx'+suffix]) | 
                (chrom_reseg['start_idx'+suffix] < 0) | 
                (chrom_reseg['start_idx'+suffix] >= len(chrom_reseg['end'].values))
            )

        chrom_reseg = chrom_reseg[~chrom_reseg['filter_1'] & ~chrom_reseg['filter_2']]

        for suffix, chrom_cn in zip(('_1', '_2'), (chrom_cn_1, chrom_cn_2)):

            chrom_reseg['idx'+suffix] = chrom_cn.index.values[chrom_reseg['start_idx'+suffix].values]
            chrom_reseg.drop(['start_idx'+suffix, 'end_idx'+suffix, 'filter'+suffix], axis=1, inplace=True)

        chrom_reseg['chromosome'] = chromosome

        reseg.append(chrom_reseg)

    return pd.concat(reseg, ignore_index=True)


def aggregate_adjacent(cnv, value_cols=None, stable_cols=None, length_normalized_cols=None):
    """ Aggregate adjacent segments with similar copy number state.

    Args:
        cnv (pandas.DataFrame): copy number table

    KwArgs:
        value_cols (list): list of columns to compare for equivalent copy number state
        stable_cols (list): columns for which values are the same between equivalent states
        length_normalized_cols (list): columns that are length normalized for equivalent states

    Returns:
        pandas.DataFrame: copy number with adjacent segments aggregated

    Copy number table should have columns:
        'chromosome', 'start', 'end', 'length',
        'major_1', 'minor_1', 'major_2', 'minor_2',
        'major_raw_e', 'minor_raw_e',
        'major_raw', 'minor_raw',
    """

    if value_cols is None:
        value_cols = ['major_1', 'major_2', 'minor_1', 'minor_2']

    if stable_cols is None:
        stable_cols = [
            'chromosome',
            'major_1',
            'major_2',
            'minor_1',
            'minor_2',
            'major_raw_e',
            'minor_raw_e',
        ]

    if length_normalized_cols is None:
        length_normalized_cols = [
            'major_raw',
            'minor_raw',
        ]

    # Group segments with same state
    cnv['chromosome_index'] = np.searchsorted(np.unique(cnv['chromosome']), cnv['chromosome'])
    cnv['diff'] = cnv[['chromosome_index'] + value_cols].diff().abs().sum(axis=1)
    cnv['is_diff'] = (cnv['diff'] != 0)
    cnv['cn_group'] = cnv['is_diff'].cumsum()

    def agg_segments(df):
        a = df[stable_cols].iloc[0]

        a['start'] = df['start'].min()
        a['end'] = df['end'].max()
        a['length'] = df['length'].sum()

        for col in length_normalized_cols:
            a[col] = (df[col] * df['length']).sum() / (df['length'].sum() + 1e-16)

        return a

    cnv = cnv.groupby('cn_group').apply(agg_segments)
    
    return cnv



