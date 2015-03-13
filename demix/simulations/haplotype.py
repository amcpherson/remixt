from collections import *
import os
import pandas as pd
import numpy as np
import gzip


def create_sim_alleles(haplotypes_template, legend_template, chromosomes, recomb_rate=20.0/1.e8):
    """ Create simulated alleles from 1000 genomes SNPs

    Args:
        haplotypes_template (str): template for 1000 genomes haplotypes filename
        legend_template (str): template for 1000 genomes legend filename
        chromosomes (str): chromosomes to simulate

    KwArgs:
        recomb_rate (float): recombination rate per nt

    Returns:
        pandas.DataFrame

    """
    sim_alleles = list()

    for chromosome in chromosomes:

        hap_filename = haplotypes_template.format(chromosome)
        legend_filename = legend_template.format(chromosome)

        data = pd.read_csv(gzip.open(legend_filename, 'r'), sep=' ', usecols=['position', 'a0', 'a1'])

        with gzip.open(hap_filename, 'r') as hap_file:
            num_1kg_individuals = len(hap_file.readline().split()) / 2

        chromosome_length = data['position'].max() + 1000

        num_recombinations = np.ceil(recomb_rate * chromosome_length)

        # Randomly simulate recombinations
        recomb_positions = np.random.random_integers(1, chromosome_length - 1, num_recombinations)
        recomb_positions.sort()

        # Randomly select individuals for each recombinated region
        recomb_individuals = np.random.random_integers(0, num_1kg_individuals, num_recombinations + 1)

        # Recombination regions
        recomb_start = np.array([0] + list(recomb_positions))
        recomb_end = np.array(list(recomb_positions) + [chromosome_length])

        # Add selected individual to legend table
        data['individual'] = None
        for start, end, individual in zip(recomb_start, recomb_end, recomb_individuals):
            data.loc[(data['position'] >= start) & (data['position'] < end), 'individual'] = individual
        data['individual'] = data['individual'].astype(int)
        assert not data['individual'].isnull().any()

        # Select nucleotide codes based on individual
        recomb_nt_code = list()
        with gzip.open(hap_filename, 'r') as hap_file:
            for (idx, row), hap_line in zip(data.iterrows(), hap_file):
                hap_data = hap_line.split()
                individual_nt_code = hap_data[row['individual']*2:row['individual']*2+2]
                individual_nt_code = np.array(individual_nt_code).astype(int)
                recomb_nt_code.append(individual_nt_code)
        recomb_nt_code = pd.DataFrame(recomb_nt_code, columns=['is_alt_0', 'is_alt_1'])

        # Add nucleotide code columns
        data = pd.concat([data, recomb_nt_code], axis=1)

        # Select nucleotides based on codes
        data['nt_0'] = np.where(data['is_alt_0'] == 0, data['a0'], data['a1'])
        data['nt_1'] = np.where(data['is_alt_1'] == 0, data['a0'], data['a1'])

        # Remove indels
        data = data[(data['a0'].str.len() == 1) & (data['a1'].str.len() == 1)]

        # Ensure sorted by position
        data.sort('position', inplace=True)

        # Add chromosome for full table
        data['chromosome'] = chromosome

        # Reformat output
        data = data.rename(columns={'a0':'ref', 'a1':'alt'})
        data = data[['chromosome', 'position', 'ref', 'alt', 'is_alt_0', 'is_alt_1', 'nt_0', 'nt_1']]

        sim_alleles.append(data)

    sim_alleles = pd.concat(sim_alleles, ignore_index=True)
    
    return sim_alleles

    