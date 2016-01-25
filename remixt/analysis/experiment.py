import collections
import itertools
import pickle
import numpy as np
import pandas as pd


def find_closest(a, v):
    """ Find closest value in a to values in v

    Args:
        a (numpy.array): array in which to search, size N
        v (numpy.array): targets to search for, size M

    Returns:
        numpy.array: index into 'a' of closest element, size M
        numpy.array: distance to closest element, size M

    Values in 'a' must be sorted

    """

    right_idx = np.minimum(np.searchsorted(a, v), len(a)-1)
    left_idx = np.maximum(right_idx - 1, 0)

    left_pos = a[left_idx]
    right_pos = a[right_idx]

    left_dist = v - left_pos
    right_dist = right_pos - v

    least_dist_idx = np.where(left_dist < right_dist, left_idx, right_idx)
    least_dist = np.minimum(left_dist, right_dist)

    return least_dist_idx, least_dist


def find_closest_segment_end(segment_data, breakpoint_data):
    """ Create a mapping between breakpoints and the segments they connect

    Args:
        segment_data (pandas.DataFrame): segmentation of the genome
        breakpoint_data (pandas.DataFrame): genomic breakpoints

    Returns:
        pandas.DataFrame: mapping between side of breakpoint and side of segment

    Input segmentation dataframe has columns: 'chromosome', 'start', 'end'

    Input genomic breakpoints dataframe as columns: 'prediction_id',
    'chromosome_1', 'strand_1', 'position_1', 'chromosome_2', 'strand_2', 'position_2'

    Returned dataframe has columns:

        'prediction_id': id into breakpoint table
        'prediction_side': side of breakpoint, 0 or 1
        'segment_idx': index of segment
        'segment_side': side of segment, 0 or 1
        'dist': distance between breakend and segment end

    """

    break_ends = breakpoint_data[[
        'prediction_id',
        'chromosome_1', 'strand_1', 'position_1',
        'chromosome_2', 'strand_2', 'position_2'
    ]]

    break_ends.set_index('prediction_id', inplace=True)
    break_ends.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in break_ends.columns])
    break_ends = break_ends.stack()
    break_ends.index.names = ('prediction_id', 'prediction_side')
    break_ends = break_ends.reset_index()
    break_ends['prediction_side'] = np.where(break_ends['prediction_side'] == '1', 0, 1)

    segment_end = segment_data[['start', 'end']].rename(columns={'start':0, 'end':1}).stack()
    segment_end.name = 'position'
    segment_end.index.names = ('segment_idx', 'segment_side')
    segment_end = segment_end.reset_index()
    segment_end = segment_end.merge(segment_data[['chromosome']], left_on='segment_idx', right_index=True)
    segment_end['strand'] = np.where(segment_end['segment_side'] == 0, '-', '+')

    chromosomes = list(segment_end['chromosome'].unique())
    strands = ('+', '-')

    break_segment_table = list()

    for chromosome, strand in itertools.product(chromosomes, strands):

        chrom_break_end = break_ends.loc[
            (break_ends['chromosome'] == chromosome) &
            (break_ends['strand'] == strand),
            ['prediction_id', 'prediction_side', 'position']
        ].copy()

        chrom_segment_end = segment_end.loc[
            (segment_end['chromosome'] == chromosome) &
            (segment_end['strand'] == strand),
            ['segment_idx', 'segment_side', 'position']
        ].copy()

        # Must be sorted for find_closest, and reset index to allow for subsequent merge
        chrom_segment_end = chrom_segment_end.sort('position').reset_index()

        idx, dist = find_closest(chrom_segment_end['position'].values, chrom_break_end['position'].values)

        chrom_break_end['idx'] = idx
        chrom_break_end['dist'] = dist

        chrom_break_end = chrom_break_end.merge(
            chrom_segment_end[['segment_idx', 'segment_side']],
            left_on='idx', right_index=True
        )

        chrom_break_end.drop(['idx', 'position'], axis=1, inplace=True)

        break_segment_table.append(chrom_break_end)

    break_segment_table = pd.concat(break_segment_table, ignore_index=True)

    return break_segment_table


def get_wild_type_adjacencies(segment_data):
    """ Calculate adjacencies in segment data.

    Args:
        segment_data (pandas.DataFrame): segmentation of the genome

    Returns:
        set of tuple: pairs of segment indices adjacent in the reference genome

    """

    # Adjacent segments in the same chromosome
    adjacencies = set()
    for idx in xrange(len(segment_data.index) - 1):
        if segment_data.iloc[idx]['chromosome'] == segment_data.iloc[idx+1]['chromosome']:
            adjacencies.add((idx, idx+1))
    return adjacencies


def create_breakpoint_segment_table(segment_data, breakpoint_data, min_brk_dist=2000):
    """ Create a table mapping breakpoints to pairs of segment extremeties.

    Args:
        segment_data (pandas.DataFrame): segmentation of the genome
        breakpoint_data (pandas.DataFrame): genomic breakpoints

    KwArgs:
        min_brk_dist (int): minimum distance to segment extremety

    Returns:
        pandas.DataFrame: mapping between side of breakpoint and side of segment

    Input segmentation dataframe has columns: 'chromosome', 'start', 'end'

    Input genomic breakpoints dataframe as columns: 'prediction_id',
    'chromosome_1', 'strand_1', 'position_1', 'chromosome_2', 'strand_2', 'position_2'

    Returned dataframe has columns:

        'prediction_id': id into breakpoint table
        'n_1': segment index for breakend 1
        'side_1': side of segment, 0 or 1, for breakend 1
        'n_2': segment index for breakend 2
        'side_2': side of segment, 0 or 1, for breakend 2

    """

    # Table of segments closest to breakpoints
    closest_segments = find_closest_segment_end(segment_data, breakpoint_data)

    # Should have a pair of breakends per breakpoint
    closest_segments = (
        closest_segments.set_index(['prediction_id', 'prediction_side'])
        .unstack()
        .dropna()
        .reset_index()
    )

    # Get adjacencies, for filtering
    adjacencies = get_wild_type_adjacencies(segment_data)

    # Breakpoints as segment index, segment side (0/1)
    breakpoint_segment = list()
    for idx, row in closest_segments.iterrows():

        if row['dist'].sum() > min_brk_dist:
            continue

        prediction_id = row['prediction_id'].iloc[0]

        n_1 = row['segment_idx'][0]
        n_2 = row['segment_idx'][1]

        side_1 = row['segment_side'][0]
        side_2 = row['segment_side'][1]

        # Remove small events that look like wild type adjacencies
        if (n_1, n_2) in adjacencies and side_1 == 1 and side_2 == 0:
            continue
        if (n_2, n_1) in adjacencies and side_2 == 1 and side_1 == 0:
            continue

        # No support for loop back inversions
        if (n_1, side_1) == (n_2, side_2):
            continue

        breakpoint_segment.append((prediction_id, n_1, side_1, n_2, side_2))

    breakpoint_segment = pd.DataFrame(breakpoint_segment,
        columns=['prediction_id', 'n_1', 'side_1', 'n_2', 'side_2'])

    return breakpoint_segment


def create_experiment(count_filename, breakpoint_filename, experiment_filename, min_brk_dist=2000, min_length=None):
    count_data = pd.read_csv(count_filename, sep='\t',
        converters={'chromosome': str})

    if min_length is not None:
        count_data = count_data[count_data['length'] > min_length]

    breakpoint_data = pd.read_csv(breakpoint_filename, sep='\t',
        converters={'chromosome_1': str, 'chromosome_2': str})

    experiment = Experiment(count_data, breakpoint_data, min_brk_dist=min_brk_dist)

    with open(experiment_filename, 'w') as f:
        pickle.dump(experiment, f)


class Experiment(object):

    def __init__(self, count_data, breakpoint_data, min_brk_dist=2000):

        self.count_data = count_data
        self.breakpoint_data = breakpoint_data

        chromosomes = self.count_data['chromosome'].unique()

        # Filter breakpoints between chromosomes with no count data
        self.breakpoint_data = self.breakpoint_data[(
            (self.breakpoint_data['chromosome_1'].isin(chromosomes)) &
            (self.breakpoint_data['chromosome_2'].isin(chromosomes))
        )]

        # Ensure the data frame is indexed 0..n-1 and add the index as a column called 'index'
        self.count_data = self.count_data.reset_index(drop=True).reset_index()

        # Create mapping between breakpoints and segment extremeties
        self.breakpoint_segment_data = create_breakpoint_segment_table(self.count_data, self.breakpoint_data, min_brk_dist=min_brk_dist)

    @property
    def segment_chromosome_id(self):
        return self.count_data['chromosome'].values

    @property
    def segment_start(self):
        return self.count_data['start'].values

    @property
    def segment_end(self):
        return self.count_data['end'].values

    @property
    def segment_major_is_allele_a(self):
        return self.count_data['major_is_allele_a'].values
    
    @property
    def x(self):
        return self.count_data[['major_readcount', 'minor_readcount', 'readcount']].values

    @property
    def l(self):
        return self.count_data['length'].values
    
    @property
    def adjacencies(self):
        return get_wild_type_adjacencies(self.count_data)

    @property
    def breakpoints(self):
        breakpoints = set()
        for n_1, side_1, n_2, side_2 in self.breakpoint_segment_data[['n_1', 'side_1', 'n_2', 'side_2']].values:
            breakpoints.add(frozenset([(n_1, side_1), (n_2, side_2)]))
        return breakpoints

    def create_cn_table(self, likelihood, cn, h):
        """ Create a table of relevant copy number data

        Args:
            experiment (Experiment): experiment object containing simulation information
            likelihood (ReadCountLikelihood): likelihood model
            cn (numpy.array): segment copy number
            h (numpy.array): haploid depths

        Returns:
            pandas.DataFrame: table of copy number information

        """

        data = pd.DataFrame({
                'chromosome': self.segment_chromosome_id,
                'start': self.segment_start,
                'end': self.segment_end,
                'major_is_allele_a': self.segment_major_is_allele_a,
                'length': self.l,
                'major_readcount': self.x[:, 0],
                'minor_readcount': self.x[:, 1],
                'readcount': self.x[:, 2],
            })

        data['major_cov'] = data['major_readcount'] / (likelihood.phi * data['length'])
        data['minor_cov'] = data['minor_readcount'] / (likelihood.phi * data['length'])

        data['major_raw'] = (data['major_cov'] - h[0]) / h[1:].sum()
        data['minor_raw'] = (data['minor_cov'] - h[0]) / h[1:].sum()
        
        data['ratio_raw'] = self.x[:, 1].astype(float) / self.x[:, :2].sum(axis=1).astype(float)

        x_e = likelihood.expected_read_count(self.l, cn)

        major_cov_e = x_e[:, 0] / (likelihood.phi * self.l)
        minor_cov_e = x_e[:, 1] / (likelihood.phi * self.l)

        major_raw_e = (major_cov_e - h[0]) / h[1:].sum()
        minor_raw_e = (minor_cov_e - h[0]) / h[1:].sum()

        data['major_raw_e'] = major_raw_e
        data['minor_raw_e'] = minor_raw_e

        for m in xrange(1, cn.shape[1]):
            data['major_{0}'.format(m)] = cn[:, m, 0]
            data['minor_{0}'.format(m)] = cn[:, m, 1]

        if 'major_2' in data:
            data['major_diff'] = np.absolute(data['major_1'] - data['major_2'])
            data['minor_diff'] = np.absolute(data['minor_1'] - data['minor_2'])

        return data

