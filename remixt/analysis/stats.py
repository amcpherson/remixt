import collections
import numpy as np

import remixt.seqdataio

FragmentStats = collections.namedtuple('FragmentStats', [
    'fragment_mean',
    'fragment_stddev',
])


def calculate_fragment_stats(seqdata_filename):

    sum_x = 0.
    sum_x2 = 0.
    n = 0.

    chromosomes = remixt.seqdataio.read_chromosomes(seqdata_filename)

    for chrom in chromosomes:
        for chrom_reads in remixt.seqdataio.read_fragment_data(seqdata_filename, chrom, chunksize=1000000):
            length = chrom_reads['end'].values - chrom_reads['start'].values

            sum_x += length.sum()
            sum_x2 += (length * length).sum()
            n += length.shape[0]

    mean = sum_x / n
    stdev = np.sqrt((sum_x2 / n) - (mean * mean))

    return FragmentStats(mean, stdev)


