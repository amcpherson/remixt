import glob
import shutil
import os
import sys
import subprocess
import tarfile
import argparse
import itertools
import collections
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

    segments = np.array([seg_start, seg_end]).T

    return segments


def write_segment_count_wig(wig_filename, seqdata_filename, chromosome_lengths, segment_length=1000):

    with open(wig_filename, 'w') as wig:

        chromosomes = remixt.seqdataio.read_chromosomes(seqdata_filename)

        for chrom in chromosomes:

            wig.write('fixedStep chrom={0} start=1 step={1} span={1}\n'.format(chrom, segment_length))

            chrom_reads = next(remixt.seqdataio.read_read_data(seqdata_filename, chromosome=chrom))

            chrom_reads.sort_values('start', inplace=True)

            chrom_segments = create_segments(chromosome_lengths[chrom], segment_length)

            seg_count = remixt.segalg.contained_counts(
                chrom_segments,
                chrom_reads[['start', 'end']].values,
            )

            wig.write('\n'.join([str(c) for c in seg_count]))
            wig.write('\n')


def calculate_allele_counts(seqdata_filename):

    allele_counts = list()
    
    chromosomes = remixt.seqdataio.read_chromosomes(seqdata_filename)

    for chrom in chromosomes:

        chrom_allele_counts = remixt.analysis.haplotype.read_snp_counts(seqdata_filename, chrom)
        
        chrom_allele_counts['chromosome'] = chrom

        allele_counts.append(chrom_allele_counts)

    allele_counts = pd.concat(allele_counts, ignore_index=True)

    return allele_counts


def infer_het_positions(seqdata_filename):

    allele_count = calculate_allele_counts(seqdata_filename)
    
    remixt.analysis.haplotype.infer_snp_genotype(allele_count)

    het_positions = allele_count.loc[allele_count['AB'] == 1, ['chromosome', 'position']]

    return het_positions


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

    allele_count['refCount'] = allele_count['refCount'].astype(int)
    allele_count['NrefCount'] = allele_count['NrefCount'].astype(int)

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
        self.bin_directory = os.path.join(self.install_directory, 'bin')

        self.chrom_info_filename = os.path.join(self.data_directory, 'chromInfo.txt.gz')

        self.wrapper_directory = os.path.realpath(os.path.dirname(__file__))
        self.run_titan_script = os.path.join(self.wrapper_directory, 'bin', 'run_titan.R')
        self.create_segments_script = os.path.join(self.bin_directory, 'createTITANsegmentfiles.pl')

        self.max_copy_number = 5


    def get_analysis_filename(self, *names):
        return os.path.realpath(os.path.join(self.analysis_directory, *names))


    def install(self, **kwargs):

        utils.makedirs(self.install_directory)
        utils.makedirs(self.packages_directory)
        utils.makedirs(self.bin_directory)

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

        with Sentinal('clone_titan') as sentinal:
            if sentinal.unfinished:
                with utils.CurrentDirectory(self.packages_directory):
                    subprocess.check_call('git clone https://github.com/gavinha/TitanCNA', shell=True)
                    with utils.CurrentDirectory('TitanCNA'):
                        subprocess.check_call('git checkout 30fceb911b99a281ccbe3fac29d154f567127410', shell=True)
                    subprocess.check_call('R CMD INSTALL TitanCNA', shell=True)

        with Sentinal('install_titan_tools') as sentinal:
            if sentinal.unfinished:
                with utils.CurrentDirectory(self.packages_directory):
                    subprocess.check_call('git clone https://github.com/gavinha/TitanCNA-utils', shell=True)
                    with utils.CurrentDirectory('TitanCNA-utils'):
                        subprocess.check_call('git checkout 4cfd6155e620dade4090322d71f45f5a39cb688e', shell=True)
                        utils.symlink('titan_scripts/createTITANsegmentfiles.pl', link_directory=self.bin_directory)

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

        # Identify het from normal
        het_positions = infer_het_positions(normal_filename)

        # Filter tumour for het positions
        tumour_allele_count = calculate_allele_counts(tumour_filename).merge(het_positions)

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

        return len(init_params.index)


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
            output_mix_file.write('\t'.join([str(a) for a in mix]) + '\n')

        subprocess.check_call([
            'perl',
            self.tool.create_segments_script,
            '-i', self.get_analysis_filename('init_{0}'.format(best_idx), 'cn.tsv'),
            '-o', self.get_analysis_filename('cn_best.tsv'),
            '-igv', self.get_analysis_filename('cn_best.igv'),
        ])

        cn_data = pd.read_csv(
            self.get_analysis_filename('cn_best.tsv'),
            sep='\t', converters={'Chromosome':str}
        )

        cn_columns = {
            'Chromosome':'chromosome',
            'Start_Position(bp)':'start',
            'End_Position(bp)':'end',
            'Copy_Number':'total_1',
            'MajorCN':'major_1',
            'MinorCN':'minor_1',
            'Clonal_Cluster':'clone',
        }

        cn_data = cn_data.rename(columns=cn_columns)[cn_columns.values()]

        cn_data['clone'] = cn_data['clone'].fillna(1).astype(int)

        cn_data['total_2'] = np.where(cn_data['clone'] == 1, cn_data['total_1'], 2)
        cn_data['major_2'] = np.where(cn_data['clone'] == 1, cn_data['major_1'], 1)
        cn_data['minor_2'] = np.where(cn_data['clone'] == 1, cn_data['minor_1'], 1)

        cn_data = cn_data.drop(['clone'], axis=1)

        cn_data.to_csv(output_cn_filename, sep='\t', index=False)



if __name__ == '__main__':

    cmdline.interface(TitanTool)



