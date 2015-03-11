import numpy as np
import pandas as pd

import demix.cn_model


def create_cn_table(experiment, cn, h, p):
    """ Create a table of relevant copy number data

    Args:
        experiment (Experiment): experiment object containing simulation information
        cn (numpy.array): segment copy number
        h (numpy.array): haploid depths
        p (numpy.array): measurable read proportion

    Returns:
        pandas.DataFrame: table of copy number information

    """

    data = pd.DataFrame({
            'chrom':experiment.segment_chromosome_id,
            'start':experiment.segment_start,
            'end':experiment.segment_end,
            'length':experiment.l,
            'major_readcount':experiment.x[:,0],
            'minor_readcount':experiment.x[:,1],
            'readcount':experiment.x[:,2],
        })    

    data['major_cov'] = data['readcount'] * data['major_readcount'] / ((data['major_readcount'] + data['minor_readcount']) * data['length'])
    data['minor_cov'] = data['readcount'] * data['minor_readcount'] / ((data['major_readcount'] + data['minor_readcount']) * data['length'])

    data['major_raw'] = (data['major_cov'] - h[0]) / h[1:].sum()
    data['minor_raw'] = (data['minor_cov'] - h[0]) / h[1:].sum()

    x_e = demix.cn_model.CopyNumberModel.expected_read_count(experiment.l, cn, h, p)

    major_cov_e = x_e[:,2] * x_e[:,0] / ((x_e[:,0] + x_e[:,1]) * experiment.l)
    minor_cov_e = x_e[:,2] * x_e[:,1] / ((x_e[:,0] + x_e[:,1]) * experiment.l)

    major_raw_e = (major_cov_e - h[0]) / h[1:].sum()
    minor_raw_e = (minor_cov_e - h[0]) / h[1:].sum()

    data['major_raw_e'] = major_raw_e
    data['minor_raw_e'] = minor_raw_e

    for m in xrange(1, cn.shape[1]):
        data['major_{0}'.format(m)] = cn[:,m,0]
        data['minor_{0}'.format(m)] = cn[:,m,1]

    data['major_diff'] = np.absolute(data['major_1'] - data['major_2'])
    data['minor_diff'] = np.absolute(data['minor_1'] - data['minor_2'])

    return data

