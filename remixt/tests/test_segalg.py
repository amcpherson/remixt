import sys
import os
import unittest
import numpy as np
import pandas as pd

remixt_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(remixt_directory)

import remixt.segalg as segalg
import remixt.simulations.pipeline

np.random.seed(2014)


class segalg_unittest(unittest.TestCase):


    def random_non_overlapping(self, n=20, high=1000):

        start = np.sort(np.random.randint(high, size=n))

        max_end = np.concatenate([start[1:], [1000]])
        max_length = max_end - start

        end = start + (np.random.random(size=n) * max_length).astype(int)

        segments = np.array([start, end]).T

        return segments


    def random_overlapping(self, n=10000, l=10, high=1000):

        start = np.sort(np.random.randint(high, size=n))
        end = start + l

        segments = np.array([start, end]).T

        return segments


    def random_positions(self, n=10000, high=1000):

        positions = np.sort(np.random.randint(high, size=n))

        return positions


    def test_contained_counts_opt(self):

        X = self.random_non_overlapping()
        Y = self.random_overlapping()

        unopt_result = segalg.contained_counts_unopt(X, Y)
        opt_result = segalg.contained_counts(X, Y)

        self.assertTrue(np.all(unopt_result == opt_result))


    def test_find_contained_positions_opt(self):

        X = self.random_non_overlapping()
        Y = self.random_positions()

        unopt_result = segalg.find_contained_positions_unopt(X, Y)
        opt_result = segalg.find_contained_positions(X, Y)

        self.assertTrue(np.all(unopt_result == opt_result))


    def test_find_contained_segments_opt(self):

        X = self.random_non_overlapping()
        Y = self.random_overlapping()

        unopt_result = segalg.find_contained_segments_unopt(X, Y)
        opt_result = segalg.find_contained_segments(X, Y)

        self.assertTrue(np.all(unopt_result == opt_result))


    def test_reindex_segments(self):

        df_1 = pd.DataFrame({
            'chromosome':'1',
            'start':[10, 20, 30],
            'end':[20, 30, 40],
        })

        df_2 = pd.DataFrame({
            'chromosome':'1',
            'start':[0, 5, 25, 27, 28, 45],
            'end':[5, 25, 27, 28, 45, 50],
        })

        df_result = pd.DataFrame({
            'chromosome':'1',
            'start':[10, 20, 25, 27, 28, 30],
            'end':[20, 25, 27, 28, 30, 40],
            'idx_1':[0, 1, 1, 1, 1, 2],
            'idx_2':[1, 1, 2, 3, 4, 4],
        })

        df_reindex = remixt.segalg.reindex_segments(df_1, df_2)

        df_reindex = df_reindex.reindex(columns=df_result.columns)

        self.assertTrue(np.all(df_result.index.values == df_reindex.index.values))
        self.assertTrue(np.all(df_result == df_reindex))


if __name__ == '__main__':
    unittest.main()


