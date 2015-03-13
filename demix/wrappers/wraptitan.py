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
import scipy.stats

import utils
import cmdline

import demix.seqdataio
import demix.segalg


def read_chromosome_lengths(chrom_info_filename):

    chrom_info = pd.read_csv(chrom_info_filename, sep='\t', compression='gzip', names=['chrom', 'length', 'twobit'])

    chrom_info['chrom'] = chrom_info['chrom'].str.replace('chr', '')

    return chrom_info.set_index('chrom')['length']


def create_segments(chrom_length, segment_length=1000):

    seg_start = np.arange(0, chrom_length, segment_length)
    seg_end = seg_start + segment_length

    segments = np.array([seg_start, seg_end]).T

    return segments


def write_segment_count_wig(wig_filename, read_data_filename, chromosome_lengths, segment_length=1000):

    with open(wig_filename, 'w') as wig:

        with tarfile.open(read_data_filename, 'r:gz') as tar:

            chromosomes = demix.seqdataio.read_chromosomes(tar)

            for chrom in chromosomes:

                wig.write('fixedStep chrom={0} start=1 step={1} span={1}\n'.format(chrom, segment_length))

                chrom_reads = next(demix.seqdataio.read_read_data(tar, chromosome=chrom))

                chrom_reads.sort('start', inplace=True)

                chrom_segments = create_segments(chromosome_lengths[chrom], segment_length)

                seg_count = demix.segalg.contained_counts(
                    chrom_segments,
                    chrom_reads[['start', 'end']].values,
                )

                wig.write('\n'.join([str(c) for c in seg_count]))


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


def infer_genotype(allele_counts, base_call_error=0.005, call_threshold=0.9):

    allele_counts['total_count'] = allele_counts['ref_count'] + allele_counts['alt_count']

    allele_counts['likelihood_AA'] = scipy.stats.binom.pmf(allele_counts['alt_count'], allele_counts['total_count'], float(base_call_error))
    allele_counts['likelihood_AB'] = scipy.stats.binom.pmf(allele_counts['alt_count'], allele_counts['total_count'], 0.5)
    allele_counts['likelihood_BB'] = scipy.stats.binom.pmf(allele_counts['ref_count'], allele_counts['total_count'], float(base_call_error))
    allele_counts['evidence'] = allele_counts['likelihood_AA'] + allele_counts['likelihood_AB'] + allele_counts['likelihood_BB']

    allele_counts['posterior_AA'] = allele_counts['likelihood_AA'] / allele_counts['evidence']
    allele_counts['posterior_AB'] = allele_counts['likelihood_AB'] / allele_counts['evidence']
    allele_counts['posterior_BB'] = allele_counts['likelihood_BB'] / allele_counts['evidence']

    allele_counts['AA'] = (allele_counts['posterior_AA'] >= call_threshold) * 1
    allele_counts['AB'] = (allele_counts['posterior_AB'] >= call_threshold) * 1
    allele_counts['BB'] = (allele_counts['posterior_BB'] >= call_threshold) * 1

    return allele_counts


def write_titan_format_alleles(allele_filename, allele_count):

    allele_count = allele_count.rename(columns={
        'chromosome':'chr',
        'ref_count':'refCount',
        'alt_count':'NrefCount',
    })

    allele_count['refBase'] = 'A'
    allele_count['NrefBase'] = 'T'

    allele_count = allele_count[[
        'chr',
        'position',
        'refBase',
        'refCount',
        'NrefBase',
        'NrefCount',
    ]]

    allele_count.to_csv(allele_filename, sep='\t', index=False, header=False)


def read_titan_params(params_filename):

    params = dict()

    with open(params_filename, 'r') as params_file:
        for line in params_file:
            key, value = line.split(':')
            params[key] = np.array(value.split()).astype(float)

    return params



class TitanTool(object):

    def __init__(self, install_directory):

        self.install_directory = os.path.abspath(install_directory)

        self.packages_directory = os.path.join(self.install_directory, 'packages')
        self.data_directory = os.path.join(self.install_directory, 'data')

        self.chrom_info_filename = os.path.join(self.data_directory, 'chromInfo.txt.gz')

        self.wrapper_directory = os.path.realpath(os.path.dirname(__file__))
        self.bin_directory = os.path.join(self.wrapper_directory, 'bin')
        self.run_titan_script = os.path.join(self.bin_directory, 'run_titan.R')
        self.parse_segments_script = os.path.join(self.bin_directory, 'parse_titan_segments.py')

        self.max_copy_number = 5


    def get_analysis_filename(self, *names):
        return os.path.realpath(os.path.join(self.analysis_directory, *names))


    def install(self, **kwargs):

        utils.makedirs(self.install_directory)
        utils.makedirs(self.packages_directory)

        Sentinal = utils.SentinalFactory(os.path.join(self.install_directory, 'sentinal_'), kwargs)

        r_packages = [
            'foreach',
            'argparse',
            'yaml',
        ]

        for pkg in r_packages:
            with Sentinal('install_r_'+pkg) as sentinal:
                if sentinal.unfinished:
                    subprocess.check_call('R -q -e "install.packages(\'{0}\', repos=\'http://cran.us.r-project.org\')"'.format(pkg), shell=True)

        bioconductor_packages = [
            'HMMcopy',
            'IRanges',
            'Rsamtools',
            'GenomeInfoDb',
            'doMC',
            'TitanCNA',
        ]

        for pkg in bioconductor_packages:
            with Sentinal('install_r_'+pkg) as sentinal:
                if sentinal.unfinished:
                    subprocess.check_call('R -q -e "source(\'http://bioconductor.org/biocLite.R\'); biocLite(\'{0}\')"'.format(pkg), shell=True)

        with Sentinal('download_chrom_info') as sentinal:
            if sentinal.unfinished:
                with utils.CurrentDirectory(self.data_directory):
                    subprocess.check_call('wget ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/chromInfo.txt.gz', shell=True)


    def create_analysis(self, analysis_directory):
        return TitanAnalysis(self, analysis_directory)



class TitanAnalysis(object):

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

        normal_wig_filename = self.get_analysis_filename('normal.wig')
        tumour_wig_filename = self.get_analysis_filename('tumour.wig')

        write_segment_count_wig(normal_wig_filename, normal_filename, chromosome_lengths)
        write_segment_count_wig(tumour_wig_filename, tumour_filename, chromosome_lengths)

        normal_allele_count = calculate_allele_counts(normal_filename)
        tumour_allele_count = calculate_allele_counts(tumour_filename)

        # Identify het from normal
        normal_allele_count = infer_genotype(normal_allele_count)
        normal_allele_count = normal_allele_count[normal_allele_count['AB'] == 1]

        # Filter tumour for het positions
        tumour_allele_count = tumour_allele_count.merge(
            normal_allele_count[['chromosome', 'position']],
            how='right',
        )

        tumour_allele_filename = self.get_analysis_filename('alleles.tsv')
        write_titan_format_alleles(tumour_allele_filename, tumour_allele_count)

        normal_contamination = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        num_clusters = [2]
        ploidy = [1, 2, 3, 4]

        init_params = itertools.product(
            normal_contamination,
            num_clusters,
            ploidy,
        )

        init_params_cols = [
            'normal_contamination',
            'num_clusters',
            'ploidy',
        ]

        init_params = pd.DataFrame(list(init_params), columns=init_params_cols)

        init_params_filename = self.get_analysis_filename('init_params.tsv')
        init_params.to_csv(init_params_filename, sep='\t', index=False)

        return init_params


    def run(self, init_param_idx):
        """ Run the analysis with specific initialization parameters

        """

        init_params_filename = self.get_analysis_filename('init_params.tsv')
        init_params = pd.read_csv(init_params_filename, sep='\t')

        if init_param_idx not in init_params.index:
            raise utils.InvalidInitParam()

        init_param_subdir = 'init_{0}'.format(init_param_idx)

        utils.makedirs(self.get_analysis_filename(init_param_subdir))

        titan_cmd = [
            'Rscript',
            self.tool.run_titan_script,
            self.get_analysis_filename('alleles.tsv'),
            self.get_analysis_filename('normal.wig'),
            self.get_analysis_filename('tumour.wig'),
            self.get_analysis_filename(init_param_subdir, 'cn.tsv'),
            self.get_analysis_filename(init_param_subdir, 'params.txt'),
            '--estimate_clonal_prevalence',
            '--estimate_normal_contamination',
            '--estimate_ploidy',
            '--max_copy_number', str(self.tool.max_copy_number),
            '--normal_contamination', str(init_params['normal_contamination'].loc[init_param_idx]),
            '--num_clusters', str(init_params['num_clusters'].loc[init_param_idx]),
            '--ploidy', str(init_params['ploidy'].loc[init_param_idx]),
        ]

        subprocess.check_call(titan_cmd, shell=False)


    def report(self, output_cn_filename, output_mix_filename):
        """ Report optimal copy number and mixture

        """

        init_params_filename = self.get_analysis_filename('init_params.tsv')
        init_params = pd.read_csv(init_params_filename, sep='\t')

        for init_param_idx, row in init_params.iterrows():

            init_param_subdir = 'init_{0}'.format(init_param_idx)
            titan_params_filename = self.get_analysis_filename(init_param_subdir, 'params.txt')

            titan_params = read_titan_params(titan_params_filename)

            init_params.loc[init_param_idx, 'model_selection_index'] = titan_params['S_Dbw validity index'][0]
            init_params.loc[init_param_idx, 'norm_contam_est'] = titan_params['Normal contamination estimate'][0]

            cell_prev_field = 'Clonal cluster cellular prevalence Z={0}'.format(int(row['num_clusters']))
            for idx, cell_prev in enumerate(titan_params[cell_prev_field]):
                init_params.loc[init_param_idx, 'cell_prev_est_{0}'.format(idx+1)] = cell_prev

        best_idx = init_params['model_selection_index'].argmin()

        n = init_params.loc[best_idx, 'norm_contam_est']

        if init_params.loc[best_idx, 'num_clusters'] == 1:
            t_1 = init_params.loc[best_idx, 'cell_prev_est_1']
            mix = [n, (1-n) * t_1]
        elif init_params.loc[best_idx, 'num_clusters'] == 2:
            t_1 = init_params.loc[best_idx, 'cell_prev_est_1']
            t_2 = init_params.loc[best_idx, 'cell_prev_est_2']
            mix = [n, (1-n) * t_2, (1-n) * abs(t_1 - t_2)]

        with open(output_mix_filename, 'w') as output_mix_file:
            output_mix_file.write('\t'.join([str(a) for a in mix]))

        subprocess.check_call([
            'python',
            self.tool.parse_segments_script,
            self.get_analysis_filename('init_{0}'.format(best_idx), 'cn.tsv'),
            output_cn_filename,
            '--max_copy_number', '{0}'.format(self.tool.max_copy_number),
        ])



if __name__ == '__main__':

    cmdline.interface(TitanTool)



