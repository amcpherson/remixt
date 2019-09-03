import numpy as np
import pandas as pd

import remixt.segalg


def create_breakends(bp):
    be = bp[['prediction_id', 'chromosome_1', 'strand_1', 'position_1', 'chromosome_2', 'strand_2', 'position_2']].copy()
    be.set_index('prediction_id', inplace=True)
    be.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in be.columns])
    be = be.stack()
    be.index.names = ('prediction_id', 'prediction_side')
    be = be.reset_index()
    be['prediction_side'] = np.where(be['prediction_side'] == '1', 0, 1)
    return be


def match_breakpoints(bp1, bp2, search_range=400):
    """ Calculate mapping of approximately similar breakpoints in two datasets

    Args:
        bp1 (pandas.DataFrame): breakpoint set 1
        bp2 (pandas.DataFrame): breakpoint set 2

    KwArgs:
        search_range (int): max distance between breakends

    Returns:
        pandas.DataFrame: matched breakpoint prediction ids

    For breakpoint sets, columns 'prediction_id', 'chromosome_1', 'strand_1', 
    'position_1', 'chromosome_2', 'strand_2', 'position_2' are expected.

    Returned dataframe has columns 'prediction_id_1', 'prediction_id_2' for
    matching predictions.

    """

    be1_gb = dict(list(create_breakends(bp1).groupby(['chromosome', 'strand'])))

    be2_gb = dict(list(create_breakends(bp2).groupby(['chromosome', 'strand'])))

    be_matched = list()

    for (chromosome, strand), be1 in be1_gb.items():
        
        if (chromosome, strand) not in be2_gb:
            continue
            
        be2 = be2_gb[(chromosome, strand)]

        be1.reset_index(drop=True, inplace=True)
        
        be1['search_start'] = be1['position'] - search_range
        be1['search_end'] = be1['position'] + search_range

        be2.sort_values('position', inplace=True)
        be2.reset_index(drop=True, inplace=True)

        idx1, idx2 = remixt.segalg.interval_position_overlap(
            be1[['search_start', 'search_end']].values,
            be2['position'].values,
        )
        
        matched = pd.DataFrame({'idx1':idx1, 'idx2':idx2})
        matched = matched.merge(be1[['prediction_id', 'prediction_side']], left_on='idx1', right_index=True)
        matched = matched.merge(be2[['prediction_id', 'prediction_side']], left_on='idx2', right_index=True, suffixes=('_1', '_2'))
        matched.drop(['idx1', 'idx2'], axis=1, inplace=True)
        
        be_matched.append(matched)
        
    be_matched = pd.concat(be_matched, ignore_index=True)

    matched = list()

    for (id_1, id_2), sides in be_matched.groupby(['prediction_id_1', 'prediction_id_2']):

        if len(sides.index) != 2:
            continue
            
        if len(sides['prediction_side_1'].unique()) != 2:
            continue

        if len(sides['prediction_side_2'].unique()) != 2:
            continue
            
        matched.append((id_1, id_2))
        
    matched = pd.DataFrame(matched, columns=['prediction_id_1', 'prediction_id_2'])

    return matched



