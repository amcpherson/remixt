import numpy as np


def is_contained(a, b):
    """ Check if segment b is fully contained within segment a """
    return b[0] >= a[0] and b[1] <= a[1]


def contained_counts_unopt(X, Y):
    """ Find counts of overlapping segments fully contained in non-overlapping segments (unopt)

    Args:
        X (numpy.array): start and end of overlapping segments with shape (N,2) for N segments
        Y (numpy.array): start and end of non-overlapping segments with shape (M,2) for M segments

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
        X (numpy.array): start and end of overlapping segments with shape (N,2) for N segments
        Y (numpy.array): start and end of non-overlapping segments with shape (M,2) for M segments

    Returns:
        numpy.array: N length array of counts of Y countained in X

    X is assumed to be ordered by start position.

    """

    idx = np.searchsorted(X[:,1], Y[:,0], )
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


def find_contained(X, Y):
    """ Find mapping of positions in Y contained within segments in X
    X and Y are assume sorted, X by starting position, Y by position
    X is a set of non-overlapping segments
    """
    M = [None]*Y.shape[0]
    y_idx = 0
    for x_idx, x in enumerate(X):
        while y_idx < Y.shape[0] and Y[y_idx] <= x[1]:
            if Y[y_idx] >= x[0]:
                M[y_idx] = x_idx
            y_idx += 1
    return M


