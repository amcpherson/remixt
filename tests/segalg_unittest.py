import sys
import os
import unittest
import numpy as np

demix_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(demix_directory)

import demix.segalg as segalg

np.random.seed(2014)


if __name__ == '__main__':

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


        def test_contained_counts_opt(self):

            X = self.random_non_overlapping()
            Y = self.random_overlapping()

            unopt_result = segalg.contained_counts_unopt(X, Y)
            opt_result = segalg.contained_counts(X, Y)

            self.assertTrue(np.all(unopt_result == opt_result))


    unittest.main()


