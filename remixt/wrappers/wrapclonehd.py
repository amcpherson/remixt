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
import scipy.stats

import utils
import cmdline

import remixt.seqdataio
import remixt.segalg
import remixt.analysis.haplotype


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

        chromosomes = remixt.seqdataio.read_chromosomes(seqdata_filename)

        for chrom in chromosomes:

            reads = next(remixt.seqdataio.read_read_data(seqdata_filename, chromosome=chrom))

            reads.sort_values('start', inplace=True)

            segments = create_segments(chromosome_lengths[chrom], segment_length)

            segments['count'] = remixt.segalg.contained_counts(
                segments[['start', 'end']].values,
                reads[['start', 'end']].values,
            )

            segments['chromosome'] = chrom
            segments['num_obs'] = 1

            segments.to_csv(cna, sep='\t', index=False, header=False,
                columns=['chromosome', 'end', 'count', 'num_obs'])


def write_tumour_baf(baf_filename, normal_filename, tumour_filename):

    with open(baf_filename, 'w') as baf_file:

        chromosomes = remixt.seqdataio.read_chromosomes(normal_filename)

        for chrom in chromosomes:

            normal_allele_count = remixt.analysis.haplotype.read_snp_counts(normal_filename, chrom)

            remixt.analysis.haplotype.infer_snp_genotype(normal_allele_count)

            het_positions = normal_allele_count.loc[normal_allele_count['AB'] == 1, ['position']]

            tumour_allele_count = remixt.analysis.haplotype.read_snp_counts(tumour_filename, chrom)
            tumour_allele_count = tumour_allele_count.merge(het_positions)

            tumour_allele_count['ref_count'] = tumour_allele_count['ref_count'].astype(int)
            tumour_allele_count['alt_count'] = tumour_allele_count['alt_count'].astype(int)

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

        self.git_repo_url = 'https://github.com/andrej-fischer/cloneHD.git'
        self.git_tag = 'v1.17.8'

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

        with Sentinal('install_gsl') as sentinal:
            if sentinal.unfinished:
                with utils.CurrentDirectory(self.packages_directory):
                    subprocess.check_call(['wget', 'ftp://ftp.gnu.org/gnu/gsl/gsl-1.16.tar.gz'])
                    subprocess.check_call(['tar', '-xzvf', 'gsl-1.16.tar.gz'])
                    with utils.CurrentDirectory('gsl-1.16'):
                        subprocess.check_call(['./configure', '--prefix', self.packages_directory])
                        subprocess.check_call(['make'])
                        subprocess.check_call(['make', 'install'])

        with Sentinal('install_clonehd') as sentinal:
            if sentinal.unfinished:
                with utils.CurrentDirectory(self.packages_directory):
                    subprocess.check_call(['git', 'clone', self.git_repo_url])
                    with utils.CurrentDirectory('cloneHD'):
                        subprocess.check_call(['git', 'checkout', self.git_tag])
                        utils.makedirs('build')
                        with utils.CurrentDirectory('src'):
                            os.environ['CPLUS_INCLUDE_PATH'] = os.path.join(self.packages_directory, 'include')
                            subprocess.check_call(['make',
                                'GSL='+os.path.join(self.packages_directory, 'lib', 'libgsl.a')+' '
                                      +os.path.join(self.packages_directory, 'lib', 'libgslcblas.a'),
                                'CC=g++'])
                        for binary in ('cloneHD', 'filterHD', 'pre-filter'):
                            binary_filename = os.path.abspath(os.path.join('build', binary))
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

        write_tumour_baf(self.get_analysis_filename('tumour.baf.txt'), normal_filename, tumour_filename)

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
            '--data', self.get_analysis_filename('tumour.cna.txt'),
            '--mode', '3',
            '--pre', self.get_analysis_filename('tumour.cna'),
            '--rnd', '0',
        ])

        subprocess.check_call([
            self.tool.filterhd_bin,
            '--data', self.get_analysis_filename('tumour.cna.txt'),
            '--mode', '3',
            '--pre', self.get_analysis_filename('tumour.cna.bias'),
            '--bias', self.get_analysis_filename('normal.cna.posterior-1.txt'),
            '--sigma', '0',
            '--jumps', '1',
            '--rnd', '0',
        ])

        subprocess.check_call([
            self.tool.filterhd_bin,
            '--data', self.get_analysis_filename('tumour.baf.txt'),
            '--mode', '1',
            '--pre', self.get_analysis_filename('tumour.baf'),
            '--sigma', '0',
            '--jumps', '1',
            '--reflect', '1',
            '--dist', '1',
            '--rnd', '0',
        ])

        subprocess.check_call([
            self.tool.clonehd_bin,
            '--cna', self.get_analysis_filename('tumour.cna.txt'),
            '--baf', self.get_analysis_filename('tumour.baf.txt'),
            '--pre', self.get_analysis_filename('tumour'),
            '--bias', self.get_analysis_filename('normal.cna.posterior-1.txt'),
            '--seed', '123',
            '--trials', '2',
            '--nmax', '3',
            '--force', '2',
            '--max-tcn', '4',
            '--cna-jumps', self.get_analysis_filename('tumour.cna.bias.jumps.txt'),
            '--baf-jumps', self.get_analysis_filename('tumour.baf.jumps.txt'),
            '--min-jump', '0.01',
            '--restarts', '10',
            '--mass-gauging', '1',
        ])


    def report(self, output_cn_filename, output_mix_filename):
        """ Report optimal copy number and mixture

        """

        segment_length = 1000

        with open(self.get_analysis_filename('tumour.summary.txt'), 'r') as summary_file:

            summary_info = dict()

            names = list()
            for line in summary_file:
                if line.startswith('#'):
                    names = line[1:].split()
                    if len(names) == 2 and names[1] == 'clones':
                        summary_info['num_clones'] = int(names[0])
                        names = ['mass'] + ['frac_'+str(i+1) for i in range(int(names[0]))]
                else:
                    values = line.split()
                    summary_info.update(dict(zip(names, values)))

        with open(output_mix_filename, 'w') as output_mix_file:

            mix = [float(summary_info['frac_'+str(i+1)]) for i in range(summary_info['num_clones'])]
            mix = [1-sum(mix)] + mix

            output_mix_file.write('\t'.join([str(a) for a in mix]) + '\n')

        cn_table = None

        for clone_id in range(1, summary_info['num_clones']+1):

            cna_filename = self.get_analysis_filename('tumour.cna.subclone-{0}.txt'.format(clone_id))

            cna_data = pd.read_csv(cna_filename, delim_whitespace=True)
            cna_data.rename(columns={'#chr':'chromosome', 'first-locus':'start', 'last-locus':'end'}, inplace=True)
            cna_data.drop(['nloci'], axis=1, inplace=True)
            # Unclear from the documentation, though techically the first datapoint is an endpoint
            # however, expanding regions results in inconsistencies
            cna_data['start'] -= segment_length
            cna_data.set_index(['chromosome', 'start', 'end'], inplace=True)
            cna_data = cna_data.idxmax(axis=1).astype(int)
            cna_data.name = 'total'
            cna_data = cna_data.reset_index()

            baf_filename = self.get_analysis_filename('tumour.baf.subclone-{0}.txt'.format(clone_id))

            baf_data = pd.read_csv(baf_filename, delim_whitespace=True)
            baf_data.rename(columns={'#chr':'chromosome', 'first-locus':'start', 'last-locus':'end'}, inplace=True)
            baf_data.drop(['nloci'], axis=1, inplace=True)
            baf_data.set_index(['chromosome', 'start', 'end'], inplace=True)
            baf_data = baf_data.fillna(0).idxmax(axis=1).astype(int)
            baf_data.name = 'allele'
            baf_data = baf_data.reset_index()

            data = remixt.segalg.reindex_segments(cna_data, baf_data)
            data = data.merge(cna_data[['total']], left_on='idx_1', right_index=True)
            data = data.merge(baf_data[['allele']], left_on='idx_2', right_index=True)

            data['major'] = np.maximum(data['allele'], data['total'] - data['allele'])
            data['minor'] = np.minimum(data['allele'], data['total'] - data['allele'])
            data.drop(['idx_1', 'idx_2', 'allele'], axis=1, inplace=True)

            # Having minor copies < 0 is common enough in the results
            # that we need to correct for it
            data['minor'] = np.maximum(data['minor'], 0)

            if not (data['minor'] >= 0).all():
                error = 'Negative minor copies\n'
                error += data[data['minor'] < 0].to_string()
                raise Exception(error)

            data.rename(inplace=True, columns={
                'total':'total_{0}'.format(clone_id),
                'minor':'minor_{0}'.format(clone_id),
                'major':'major_{0}'.format(clone_id),
            })

            if cn_table is None:
                cn_table = data

            else:
                cn_table_prev = cn_table
                cn_table = remixt.segalg.reindex_segments(cn_table_prev, data)

                cn_table_prev.drop(['chromosome', 'start', 'end'], axis=1, inplace=True)
                data.drop(['chromosome', 'start', 'end'], axis=1, inplace=True)

                cn_table = cn_table.merge(cn_table_prev, left_on='idx_1', right_index=True)
                cn_table = cn_table.merge(data, left_on='idx_2', right_index=True)

                cn_table.drop(['idx_1', 'idx_2'], axis=1, inplace=True)

        cn_table.to_csv(output_cn_filename, sep='\t', index=False)


if __name__ == '__main__':

    cmdline.interface(CloneHDTool)



