import collections
import numpy as np

import remixt.seqdataio

FragmentStats = collections.namedtuple('FragmentStats', [
    'fragment_mean',
    'fragment_stddev',
])


def calculate_fragment_stats(seqdata_filename):

    segment_counts = list()
    
    sum_x = 0.
    sum_x2 = 0.
    n = 0.

    chromosomes = remixt.seqdataio.read_chromosomes(seqdata_filename)

    for chrom in chromosomes:

        chrom_reads = next(remixt.seqdataio.read_read_data(seqdata_filename, chromosome=chrom))

        length = chrom_reads['end'].values - chrom_reads['start'].values

        sum_x += length.sum()
        sum_x2 += (length * length).sum()
        n += length.shape[0]

    mean = sum_x / n
    stdev = np.sqrt((sum_x2 / n) - (mean * mean)) 

    return FragmentStats(mean, stdev)


