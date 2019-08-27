import glob
import shutil
import os
import sys
import subprocess
import tarfile
import argparse
import itertools
import numpy as np
import pandas as pd

import utils
import cmdline

import remixt.seqdataio
import remixt.segalg
import remixt.analysis.haplotype


def calculate_segment_counts(seqdata_filename, segments):

    segment_counts = list()
    
    chromosomes = remixt.seqdataio.read_chromosomes(seqdata_filename)

    for chrom, chrom_segs in segments.groupby('chromosome'):

        try:
            chrom_reads = next(remixt.seqdataio.read_read_data(seqdata_filename, chromosome=chrom))
        except StopIteration:
            chrom_reads = pd.DataFrame(columns=['start', 'end'])

        chrom_segs.sort_values('start', inplace=True)
        chrom_reads.sort_values('start', inplace=True)

        chrom_segs['count'] = remixt.segalg.contained_counts(
            chrom_segs[['start', 'end']].values,
            chrom_reads[['start', 'end']].values,
        )

        chrom_segs['count'] = chrom_segs['count'].astype(int)

        segment_counts.append(chrom_segs)

    segment_counts = pd.concat(segment_counts, ignore_index=True)

    return segment_counts


def calculate_allele_counts(seqdata_filename):

    allele_counts = list()
    
    chromosomes = remixt.seqdataio.read_chromosomes(seqdata_filename)

    for chrom in chromosomes:

        chrom_allele_counts = remixt.analysis.haplotype.read_snp_counts(seqdata_filename, chrom)

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



class ThetaTool(object):

    def __init__(self, install_directory):

        self.install_directory = os.path.abspath(install_directory)

        self.packages_directory = os.path.join(self.install_directory, 'packages')
        self.bin_directory = os.path.join(self.packages_directory, 'THetA', 'bin')

        self.theta_bin = os.path.join(self.bin_directory, 'RunTHetA')

        self.max_copynumber = 6


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


    def create_analysis(self, analysis_directory):
        return ThetaAnalysis(self, analysis_directory)



class ThetaAnalysis(object):

    def __init__(self, tool, analysis_directory):
        self.tool = tool
        self.analysis_directory = analysis_directory
        utils.makedirs(self.analysis_directory)


    def get_analysis_filename(self, *names):
        return os.path.realpath(os.path.join(self.analysis_directory, *names))


    def prepare(self, normal_filename, tumour_filename, perfect_segment_filename=None, **kwargs):

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
            'chrom_idx':range(len(chromosomes)),
        })

        chrom_idx_filename = self.get_analysis_filename('chrom_idx.tsv')
        chrom_idx.to_csv(chrom_idx_filename, sep='\t', index=False)

        # Add chromosome index
        count_data = count_data.merge(chrom_idx).drop('chromosome', axis=1)

        # Add segment index
        count_data = count_data.reset_index().rename(columns={'index':'segment_id'})

        # Add upper and lower bound
        count_data['upper_bound'] = self.tool.max_copynumber
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

        return 1


    def run(self, init_param_idx):

        if init_param_idx != 0:
            raise utils.InvalidInitParam()

        chrom_idx_filename = self.get_analysis_filename('chrom_idx.tsv')
        chrom_idx = pd.read_csv(chrom_idx_filename, sep='\t')

        count_data_filename = self.get_analysis_filename('counts.tsv')
        theta_prefix = self.get_analysis_filename('theta')

        # Run theta
        theta_cmd = [
            self.tool.theta_bin,
            count_data_filename,
            '--FORCE',
            '--NUM_INTERVALS', '15',
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
            'cd ' + self.tool.bin_directory + '; ' +
            'runBAFGaussianModel(' + ','.join(run_baf_args) + ')'
        )

        octave_cmd = 'octave --eval "' + octave_eval + '"'

        subprocess.check_call(octave_cmd, shell=True)


    def report(self, output_cn_filename, output_mix_filename):

        theta2_results_filename = self.get_analysis_filename('theta2.BAF.NLL.results')

        theta2_results = pd.read_csv(theta2_results_filename, sep='\t').rename(columns={'#NLL':'NLL'})
        theta2_results['Total_NLL'] = theta2_results['NLL'] + theta2_results['BAF_NLL']

        best_idx = theta2_results['Total_NLL'].argmin()
        
        best_frac = theta2_results.loc[best_idx, 'mu']
        best_frac = best_frac.split(',')

        with open(output_mix_filename, 'w') as output_mix_file:
            output_mix_file.write('\t'.join(best_frac) + '\n')

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

    cmdline.interface(ThetaTool)



