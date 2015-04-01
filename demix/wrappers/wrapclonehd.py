import glob
import shutil
import os
import sys
import subprocess
import tarfile
import argparse
import itertools
import platform
import numpy as np
import pandas as pd
import scipy.stats

import utils
import cmdline

import demix.seqdataio
import demix.segalg
import demix.analysis.haplotype


def read_chromosome_lengths(chrom_info_filename):

    chrom_info = pd.read_csv(chrom_info_filename, sep='\t', compression='gzip', names=['chrom', 'length', 'twobit'])

    chrom_info['chrom'] = chrom_info['chrom'].str.replace('chr', '')

    return chrom_info.set_index('chrom')['length']


def create_segments(chrom_length, segment_length=1000):

    seg_start = np.arange(0, chrom_length, segment_length)
    seg_end = seg_start + segment_length

    segments = pd.DataFrame({'start':seg_start, 'end':seg_end})

    return segments


def write_cna(cna_filename, seqdata_filename, chromosome_lengths, segment_length=1000):

    with open(cna_filename, 'w') as cna:

        chromosomes = demix.seqdataio.read_chromosomes(seqdata_filename)

        for chrom in chromosomes:

            reads = next(demix.seqdataio.read_read_data(seqdata_filename, chromosome=chrom))

            reads.sort('start', inplace=True)

            segments = create_segments(chromosome_lengths[chrom], segment_length)

            segments['count'] = demix.segalg.contained_counts(
                segments[['start', 'end']].values,
                reads[['start', 'end']].values,
            )

            segments['chromosome'] = chrom

            segments.to_csv(cna, sep='\t', index=False, header=False,
                columns=['chromosome', 'end', 'count'])


def write_tumour_baf(baf_filename, normal_filename, tumour_filename):

    with open(baf_filename, 'w') as baf_file:

        chromosomes = demix.seqdataio.read_chromosomes(normal_filename)

        for chrom in chromosomes:

            normal_allele_count = demix.analysis.haplotype.read_snp_counts(normal_filename, chrom)

            demix.analysis.haplotype.infer_snp_genotype(normal_allele_count)

            het_positions = normal_allele_count.loc[normal_allele_count['AB'] == 1, ['position']]

            tumour_allele_count = demix.analysis.haplotype.read_snp_counts(tumour_filename, chrom)
            tumour_allele_count = tumour_allele_count.merge(het_positions)

            tumour_allele_count['minor_count'] = np.minimum(
                tumour_allele_count['ref_count'],
                tumour_allele_count['alt_count'],
            )

            tumour_allele_count['total_count'] = (
                tumour_allele_count['ref_count'] +
                tumour_allele_count['alt_count']
            )

            tumour_allele_count['chromosome'] = chrom

            tumour_allele_count.to_csv(baf_file, sep='\t', index=False, header=False,
                columns=['chromosome', 'position', 'minor_count', 'total_count'])


class CloneHDTool(object):

    def __init__(self, install_directory):

        self.install_directory = os.path.abspath(install_directory)

        self.packages_directory = os.path.join(self.install_directory, 'packages')
        self.data_directory = os.path.join(self.install_directory, 'data')
        self.bin_directory = os.path.join(self.install_directory, 'bin')

        self.package_url = {
            'Linux':'https://github.com/andrej-fischer/cloneHD/releases/download/v1.17.8/cloneHD-v1.17.8.tar.gz',
            'Darwin':'https://github.com/andrej-fischer/cloneHD/releases/download/v1.17.8/build-MacOSX.tar.gz',
        }

        self.package_filename = {
            'Linux':'cloneHD-v1.17.8.tar.gz',
            'Darwin':'build-MacOSX.tar.gz',
        }

        self.package_bin_subdir = {
            'Linux':os.path.join('cloneHD-v1.17.8', 'build'),
            'Darwin':'build',
        }

        self.chrom_info_filename = os.path.join(self.data_directory, 'chromInfo.txt.gz')

        self.filterhd_bin = os.path.join(self.bin_directory, 'filterHD')
        self.clonehd_bin = os.path.join(self.bin_directory, 'cloneHD')


    def get_analysis_filename(self, *names):
        return os.path.realpath(os.path.join(self.analysis_directory, *names))


    def install(self, **kwargs):

        utils.makedirs(self.install_directory)
        utils.makedirs(self.packages_directory)
        utils.makedirs(self.bin_directory)

        Sentinal = utils.SentinalFactory(os.path.join(self.install_directory, 'sentinal_'), kwargs)

        with Sentinal('install_clonehd') as sentinal:
            if sentinal.unfinished:
                with utils.CurrentDirectory(self.packages_directory):
                    install_sys = platform.system()
                    subprocess.check_call('wget --no-check-certificate ' + self.package_url[install_sys] + ' -O ' + self.package_filename[install_sys], shell=True)
                    try:
                        subprocess.check_call('tar -xzvf '+self.package_filename[install_sys], shell=True)
                    except subprocess.CalledProcessError as e:
                        print e
                    for binary in ('cloneHD', 'filterHD', 'pre-filter'):
                        binary_filename = os.path.abspath(os.path.join(self.package_bin_subdir[install_sys], binary))
                        utils.symlink(binary_filename, link_directory=self.bin_directory)

        with Sentinal('download_chrom_info') as sentinal:
            if sentinal.unfinished:
                with utils.CurrentDirectory(self.data_directory):
                    subprocess.check_call('wget ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/chromInfo.txt.gz', shell=True)


    def create_analysis(self, analysis_directory):
        return CloneHDAnalysis(self, analysis_directory)



class CloneHDAnalysis(object):

    def __init__(self, tool, analysis_directory):
        self.tool = tool
        self.analysis_directory = analysis_directory
        utils.makedirs(self.analysis_directory)


    def get_analysis_filename(self, *names):
        return os.path.realpath(os.path.join(self.analysis_directory, *names))


    def prepare(self, normal_filename, tumour_filename, **kwargs):
        """ Initialize analysis
        """

        utils.makedirs(self.analysis_directory)

        chromosome_lengths = read_chromosome_lengths(self.tool.chrom_info_filename)

        write_cna(self.get_analysis_filename('normal.cna.txt'), normal_filename, chromosome_lengths)
        write_cna(self.get_analysis_filename('tumour.cna.txt'), tumour_filename, chromosome_lengths)

        write_tumour_baf(self.get_analysis_filename('tumour.cna.txt'), normal_filename, tumour_filename)

        return 1


    def run(self, init_param_idx):
        """ Run the analysis with specific initialization parameters

        """

        if init_param_idx != 0:
            raise utils.InvalidInitParam()

        subprocess.check_call([
            self.tool.filterhd_bin,
            '--data', self.get_analysis_filename('normal.cna.txt'),
            '--mode', '3',
            '--pre', self.get_analysis_filename('normal.cna'),
            '--rnd', '0',
        ])

        subprocess.check_call([
            self.tool.filterhd_bin,
            '--data', self.get_analysis_filename('tumor.cna.txt'),
            '--mode', '3',
            '--pre', self.get_analysis_filename('tumor.cna'),
            '--rnd', '0',
        ])

        subprocess.check_call([
            self.tool.filterhd_bin,
            '--data', self.get_analysis_filename('tumor.cna'),
            '--mode', '3',
            '--pre', self.get_analysis_filename('tumor.cna.bias'),
            '--bias', self.get_analysis_filename('normal.cna.posterior-1.txt'),
            '--sigma', '0',
            '--jumps', '1',
            '--rnd', '0',
        ])

        subprocess.check_call([
            self.tool.filterhd_bin,
            '--data', self.get_analysis_filename('tumor.baf.txt'),
            '--mode', '1',
            '--pre', self.get_analysis_filename('tumor.baf'),
            '--sigma', '0',
            '--jumps', '1',
            '--reflect', '1',
            '--dist', '1',
            '--rnd', '0',
        ])

        subprocess.check_call([
            self.tool.clonehd_bin,
            '--cna', self.get_analysis_filename('tumor.cna.txt'),
            '--baf', self.get_analysis_filename('tumor.baf.txt'),
            '--pre', self.get_analysis_filename('tumor'),
            '--bias', self.get_analysis_filename('normal.cna.posterior-1.txt'),
            '--seed', '123',
            '--trials', '2',
            '--nmax', '3',
            '--force',
            '--max-tcn', '4',
            '--cna-jumps', self.get_analysis_filename('tumor.cna.bias.jumps.txt'),
            '--baf-jumps', self.get_analysis_filename('tumor.baf.jumps.txt'),
            '--min-jump', '0.01',
            '--restarts', '10',
            '--mass-gauging', '1',
        ])


    def report(self, output_cn_filename, output_mix_filename):
        """ Report optimal copy number and mixture

        """

        pass


if __name__ == '__main__':

    cmdline.interface(CloneHDTool)



