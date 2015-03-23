import numpy as np


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


def find_contained_unopt(X, Y):
    """ Find mapping of positions contained within non-overlapping segments (unopt)

    Args:
        X (numpy.array): start and end of non-overlapping segments with shape (N,2) for N segments
        Y (numpy.array): positions with shape (M,) for M positions

    Returns:
        numpy.array: M length array of indices into X containing elements in Y

    X is assumed to be ordered by start position.
    Y is assumed sorted.

    """

    M = [None]*Y.shape[0]
    y_idx = 0
    for x_idx, x in enumerate(X):
        while y_idx < Y.shape[0] and Y[y_idx] <= x[1]:
            if Y[y_idx] >= x[0]:
                M[y_idx] = x_idx
            y_idx += 1
    return M


def find_contained(X, Y):
    """ Find mapping of positions contained within non-overlapping segments

    Args:
        X (numpy.array): start and end of non-overlapping segments with shape (N,2) for N segments
        Y (numpy.array): positions with shape (M,) for M positions

    Returns:
        numpy.array: M length array of indices into X containing elements in Y

    X is assumed to be ordered by start position.
    Y is assumed sorted.

    """

    # Positions less than segment end point
    idx = np.searchsorted(X[:,1], Y)

    # Mask positions outside greatest endpoint
    mask = idx < X.shape[0]
    idx[~mask] = 0

    # Mask positions that are not fully contained within a segment
    mask = mask & (Y >= X[idx,0]) & (Y <= X[idx,1])

    return np.ma.array(idx, mask=~mask, fill_value=None)


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


