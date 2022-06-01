import unittest
import numpy as np
import pandas as pd

import remixt.analysis.haplotype


class haplotype_unittest(unittest.TestCase):

    def test_calculate_haplotypes(self):
        test_data = pd.DataFrame([
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [1, 1, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
        ]).T

        def create_test_alleles(allele1):
            return pd.DataFrame({
                'chromosome': '1',
                'position': range(len(allele1)),
                'allele1': allele1,
                'allele2': 1-allele1}).set_index(['chromosome', 'position'])

        phasing_samples = [create_test_alleles(a[1]) for a in test_data.items()]

        haplotypes = remixt.analysis.haplotype.calculate_haplotypes(phasing_samples, changepoint_threshold=0.95)
        assert (haplotypes['hap_label'] == pd.Series([0, 0, 1, 2, 2, 2])).all()
        assert (haplotypes['allele1'] == pd.Series([0, 0, 0, 1, 1, 0])).all()

        haplotypes = remixt.analysis.haplotype.calculate_haplotypes(phasing_samples, changepoint_threshold=0.55)
        assert (haplotypes['hap_label'] == pd.Series([0, 0, 0, 0, 0, 0])).all()
        assert (haplotypes['allele1'] == pd.Series([0, 0, 0, 1, 1, 0])).all()


if __name__ == '__main__':
    unittest.main()


