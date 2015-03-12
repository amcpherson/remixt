import glob
import shutil
import os
import sys
import subprocess
import tarfile
import argparse
import vcf
import itertools
import numpy as np
import pandas as pd

import utils
import cmdline

import demix.seqdataio
import demix.segalg


def calculate_segment_counts(read_data_filename, segments):

    segment_counts = list()
    
    with tarfile.open(read_data_filename, 'r:gz') as tar:

        chromosomes = demix.seqdataio.read_chromosomes(tar)

        for chrom in chromosomes:

            chrom_segs = segments[segments['chromosome'] == chrom]
            chrom_reads = next(demix.seqdataio.read_read_data(tar, chromosome=chrom))

            chrom_reads.sort('start', inplace=True)

            chrom_segs['count'] = demix.segalg.contained_counts(
                chrom_segs[['start', 'end']].values,
                chrom_reads[['start', 'end']].values,
            )

            chrom_segs['count'] = chrom_segs['count'].astype(int)

            segment_counts.append(chrom_segs)

    segment_counts = pd.concat(segment_counts, ignore_index=True)

    return segment_counts


def calculate_allele_counts(read_data_filename):

    allele_counts = list()
    
    with tarfile.open(read_data_filename, 'r:gz') as tar:

        chromosomes = demix.seqdataio.read_chromosomes(tar)

        for chrom in chromosomes:

            chrom_alleles = next(demix.seqdataio.read_allele_data(tar, chromosome=chrom))

            chrom_allele_counts = (
                chrom_alleles
                .groupby(['position', 'is_alt'])['fragment_id']
                .size()
                .unstack()
                .fillna(0)
                .astype(int)
                .rename(columns={0:'ref_count', 1:'alt_count'})
                .reset_index()
            )

            chrom_allele_counts['chromosome'] = chrom

            allele_counts.append(chrom_allele_counts)

    allele_counts = pd.concat(allele_counts, ignore_index=True)

    return allele_counts


def write_theta_format_alleles(allele_filename, allele_count):

    allele_count['total'] = allele_count['ref_count'] + allele_count['alt_count']

    # Nucleotide count columns, dont appear to be used
    for nt in 'ACTG':
        allele_count[nt] = 0

    allele_count = allele_count[[
        'chrom_idx',
        'position',
        'A', 'C', 'T', 'G',
        'total',
        'ref_count',
        'alt_count',
    ]]

    allele_count.to_csv(allele_filename, sep='\t', index=False, header=False)


count_cols = [
    'segment_id',
    'chrom_idx',
    'start',
    'end',
    'count_tumour',
    'count_normal',
    'upper_bound',
    'lower_bound',
]


class ThetaWrapper(object):

    def __init__(self, install_directory):

        self.install_directory = os.path.abspath(install_directory)

        self.packages_directory = os.path.join(self.install_directory, 'packages')
        self.bin_directory = os.path.join(self.packages_directory, 'THetA', 'bin')

        self.theta_bin = os.path.join(self.bin_directory, 'RunTHetA')

        self.max_copynumber = 6


    def get_analysis_filename(self, *names):
        return os.path.realpath(os.path.join(self.analysis_directory, *names))


    def install(self, **kwargs):

        Sentinal = utils.SentinalFactory(os.path.join(self.install_directory, 'sentinal_'), kwargs)

        with Sentinal('download_theta') as sentinal:
            if sentinal.unfinished:
                with utils.CurrentDirectory(self.packages_directory):
                    utils.rmtree('THetA')
                    subprocess.check_call('git clone https://github.com/amcpherson/THetA.git', shell=True)

        with Sentinal('install_theta') as sentinal:
            if sentinal.unfinished:
                with utils.CurrentDirectory(os.path.join(self.packages_directory, 'THetA')):
                    subprocess.check_call('rm -rf bin', shell=True)
                    subprocess.check_call('mkdir bin', shell=True)
                    subprocess.check_call('cp python/RunTHetA bin', shell=True)
                    subprocess.check_call('ant', shell=True)
                    subprocess.check_call('cp python/CreateExomeInput bin', shell=True)
                    subprocess.check_call('cp matlab/runBAFGaussianModel.m bin', shell=True)


    def init(self, analysis_directory, normal_filename, tumour_filename, perfect_segment_filename=None, **kwargs):

        self.analysis_directory = analysis_directory

        utils.makedirs(self.analysis_directory)

        segments = pd.read_csv(perfect_segment_filename, sep='\t', converters={'chromosome':str})

        normal_segment_count = calculate_segment_counts(normal_filename, segments)
        tumour_segment_count = calculate_segment_counts(tumour_filename, segments)

        count_data = pd.merge(normal_segment_count, tumour_segment_count,
            on=['chromosome', 'start', 'end'],
            suffixes=('_normal', '_tumour'))

        # Create a mapping from chromosome names to indices
        chromosomes = count_data['chromosome'].unique()

        chrom_idx = pd.DataFrame({
            'chromosome':chromosomes,
            'chrom_idx':xrange(len(chromosomes)),
        })

        chrom_idx_filename = self.get_analysis_filename('chrom_idx.tsv')
        chrom_idx.to_csv(chrom_idx_filename, sep='\t', index=False)

        # Add chromosome index
        count_data = count_data.merge(chrom_idx).drop('chromosome', axis=1)

        # Add segment index
        count_data = count_data.reset_index().rename(columns={'index':'segment_id'})

        # Add upper and lower bound
        count_data['upper_bound'] = self.max_copynumber
        count_data['lower_bound'] = 0

        # Reorder columns and output
        count_data = count_data[count_cols]

        count_data_filename = self.get_analysis_filename('counts.tsv')
        count_data.to_csv(count_data_filename, sep='\t', index=False, header=False)

        # Write alleles in preparation for theta2
        normal_allele_filename = self.get_analysis_filename('normal_alleles.tsv')
        tumour_allele_filename = self.get_analysis_filename('tumour_alleles.tsv')

        normal_allele_count = calculate_allele_counts(normal_filename)
        tumour_allele_count = calculate_allele_counts(tumour_filename)

        normal_allele_count = normal_allele_count.merge(chrom_idx)
        tumour_allele_count = tumour_allele_count.merge(chrom_idx)

        write_theta_format_alleles(normal_allele_filename, normal_allele_count)
        write_theta_format_alleles(tumour_allele_filename, tumour_allele_count)


    def run(self, analysis_directory, init_param_idx):

        if init_param_idx != 0:
            raise utils.InvalidInitParam()

        self.analysis_directory = analysis_directory

        chrom_idx_filename = self.get_analysis_filename('chrom_idx.tsv')
        chrom_idx = pd.read_csv(chrom_idx_filename, sep='\t')

        count_data_filename = self.get_analysis_filename('counts.tsv')
        theta_prefix = self.get_analysis_filename('theta')

        # Run theta
        theta_cmd = [
            self.theta_bin,
            count_data_filename,
            '--FORCE',
            '--OUTPUT_PREFIX', theta_prefix
        ]
        subprocess.check_call(' '.join(theta_cmd), shell=True)

        results_filename = theta_prefix + '.n3.results'

        normal_allele_filename = self.get_analysis_filename('normal_alleles.tsv')
        tumour_allele_filename = self.get_analysis_filename('tumour_alleles.tsv')

        theta2_prefix = self.get_analysis_filename('theta2')

        # Run theta2 VAF
        run_baf_args = [
            '\'' + tumour_allele_filename + '\'',
            '\'' + normal_allele_filename + '\'',
            '\'' + count_data_filename + '\'',
            '\'' + results_filename + '\'',
            '[' + ','.join([str(a) for a in chrom_idx['chrom_idx'].values]) + ']',
            '\'' + theta2_prefix + '\'',
            '[11,8]',
            '\'none\'',
        ]

        octave_eval = (
            'cd ' + self.bin_directory + '; ' +
            'runBAFGaussianModel(' + ','.join(run_baf_args) + ')'
        )

        octave_cmd = 'octave --eval "' + octave_eval + '"'

        subprocess.check_call(octave_cmd, shell=True)


    def report(self, analysis_directory, output_cn_filename, output_mix_filename):

        self.analysis_directory = analysis_directory

        theta2_results_filename = self.get_analysis_filename('theta2.BAF.NLL.results')

        theta2_results = pd.read_csv(theta2_results_filename, sep='\t').rename(columns={'#NLL':'NLL'})
        theta2_results['Total_NLL'] = theta2_results['NLL'] + theta2_results['BAF_NLL']

        best_idx = theta2_results['Total_NLL'].argmin()
        
        best_frac = theta2_results.loc[best_idx, 'mu']
        best_frac = best_frac.split(',')

        with open(output_mix_filename, 'w') as output_mix_file:
            output_mix_file.write('\t'.join(best_frac))

        best_cn = theta2_results.loc[best_idx, 'C']
        best_cn = [a.split(',') for a in best_cn.split(':')]
        best_cn = np.array(best_cn).astype(int).T

        chrom_idx_filename = self.get_analysis_filename('chrom_idx.tsv')
        chrom_idx = pd.read_csv(chrom_idx_filename, sep='\t')

        count_data_filename = self.get_analysis_filename('counts.tsv')
        cn_data = pd.read_csv(count_data_filename, sep='\t', header=None, names=count_cols)
        cn_data = cn_data.merge(chrom_idx)

        cn_data['total_1'] = best_cn[0]
        cn_data['total_2'] = best_cn[1]

        cn_data = cn_data[[
            'chromosome',
            'start',
            'end',
            'total_1',
            'total_2',
        ]]

        cn_data.to_csv(output_cn_filename, sep='\t', index=False, header=True)


if __name__ == '__main__':

    cmdline.interface(ThetaWrapper)



