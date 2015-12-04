import sys
import os
import unittest
import copy
import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.optimize

import remixt.seqdataio

np.random.seed(2014)


class seqdata_unittest(unittest.TestCase):

    def test_seqdataio(self):

        writer = remixt.seqdataio.Writer('./test.seqdata', './')

        chromosome = '1'

        num_reads = 10000000
        num_alleles = num_reads * 4
        
        fragments = pd.DataFrame({'start':np.random.randint(0, int(1e8), size=num_reads)})
        fragments['end'] = fragments['start'] + np.random.randint(0, 100, size=num_reads)

        alleles = pd.DataFrame({
            'fragment_id':np.sort(np.random.randint(0, num_reads, size=num_alleles)),
            'position':np.random.randint(0, int(1e8), size=num_alleles),
            'is_alt':np.random.randint(0, 2, size=num_alleles),
        })
        alleles = alleles[['fragment_id', 'position', 'is_alt']]

        chunk_size = 1000000
        num_reads_written = 0

        while num_reads_written < num_reads:

            fragment_ids = pd.DataFrame({'fragment_id':np.arange(chunk_size, dtype=int) + num_reads_written})

            fragments_chunk = fragments.reindex(fragment_ids.set_index('fragment_id').index)
            alleles_chunk = alleles[alleles['fragment_id'].isin(fragment_ids['fragment_id'])].copy()

            fragments_chunk.reset_index(inplace=True, drop=True)
            alleles_chunk['fragment_id'] -= num_reads_written

            writer.write(chromosome, fragments_chunk, alleles_chunk)

            num_reads_written += chunk_size

        writer.close()

        fragments_test = next(remixt.seqdataio.read_read_data('./test.seqdata', chromosome='1', num_rows=None))
        alleles_test = next(remixt.seqdataio.read_allele_data('./test.seqdata', chromosome='1', num_rows=None))

        self.assertEqual(fragments.values.shape, fragments_test.values.shape)
        self.assertEqual(alleles.values.shape, alleles_test.values.shape)

        self.assertTrue(np.all(fragments.values == fragments_test.values))
        self.assertTrue(np.all(alleles.values == alleles_test.values))


if __name__ == '__main__':
    unittest.main()


