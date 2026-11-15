import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import remixt.seqdataio


class seqdata_unittest(unittest.TestCase):

    def test_seqdataio(self):
        """ Fragments and alleles written in chunks are read back whole.
        """
        np.random.seed(2014)

        chromosome = '1'

        num_reads = 1000000
        num_alleles = num_reads * 4

        fragments = pd.DataFrame({'start': np.random.randint(0, int(1e8), size=num_reads)})
        fragments['end'] = fragments['start'] + np.random.randint(0, 100, size=num_reads)

        alleles = pd.DataFrame({
            'fragment_id': np.sort(np.random.randint(0, num_reads, size=num_alleles)),
            'position': np.random.randint(0, int(1e8), size=num_alleles),
            'is_alt': np.random.randint(0, 2, size=num_alleles),
        })

        with tempfile.TemporaryDirectory() as tmp:
            seqdata_filename = os.path.join(tmp, 'test.seqdata')
            writer = remixt.seqdataio.Writer(seqdata_filename)

            chunk_size = 100000
            for chunk_start in range(0, num_reads, chunk_size):
                fragment_ids = np.arange(chunk_start, chunk_start + chunk_size)

                fragments_chunk = fragments.loc[fragment_ids].reset_index(drop=True)
                fragments_chunk['fragment_id'] = fragment_ids

                alleles_chunk = alleles[alleles['fragment_id'].isin(fragment_ids)].copy()

                writer.write(chromosome, fragments_chunk, alleles_chunk)

            writer.close()

            fragments_test = remixt.seqdataio.read_fragment_data(seqdata_filename, chromosome=chromosome)
            alleles_test = remixt.seqdataio.read_allele_data(seqdata_filename, chromosome=chromosome)

        self.assertEqual(len(fragments_test.index), num_reads)
        self.assertEqual(len(alleles_test.index), num_alleles)

        fragments_test = fragments_test.sort_values('fragment_id')
        self.assertTrue(np.array_equal(fragments_test['fragment_id'].values, np.arange(num_reads)))
        self.assertTrue(np.array_equal(fragments_test['start'].values, fragments['start'].values))
        self.assertTrue(np.array_equal(fragments_test['end'].values, fragments['end'].values))

        for col in ('fragment_id', 'position', 'is_alt'):
            self.assertTrue(np.array_equal(alleles_test[col].values, alleles[col].values))


if __name__ == '__main__':
    unittest.main()
