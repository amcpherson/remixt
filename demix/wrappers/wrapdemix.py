import glob
import shutil
import os
import sys
import subprocess
import tarfile
import argparse
import itertools
import pickle
import numpy as np
import pandas as pd

import utils
import cmdline

import demix.analysis.segment
import demix.analysis.haplotype
import demix.analysis.pipeline
import demix.utils



class DemixTool(object):

    def __init__(self, install_directory):

        self.install_directory = os.path.abspath(install_directory)

        self.packages_directory = os.path.join(self.install_directory, 'packages')
        self.ref_data_directory = os.path.join(self.install_directory, 'ref_data')


    def install(self, **kwargs):
        pass

    def create_analysis(self, analysis_directory):
        return DemixAnalysis(self, analysis_directory)



class DemixAnalysis(object):

    def __init__(self, tool, analysis_directory):
        self.tool = tool
        self.analysis_directory = analysis_directory
        utils.makedirs(self.analysis_directory)


    def get_analysis_filename(self, *names):
        return os.path.realpath(os.path.join(self.analysis_directory, *names))


    def prepare(self, normal_filename, tumour_filename, segment_filename=None, breakpoint_filename=None, haplotype_filename=None, **kwargs):

        segments = pd.read_csv(segment_filename, sep='\t', converters={'chromosome':str})

        segment_counts = demix.analysis.segment.create_segment_counts(
            segments,
            tumour_filename,
        )

        segment_counts['length'] = segment_counts['end'] - segment_counts['start']

        allele_counts = demix.analysis.haplotype.create_allele_counts(
            segments,
            tumour_filename,
            haplotype_filename,
        )

        phased_allele_counts, = demix.analysis.haplotype.phase_segments(allele_counts)

        segment_allele_counts = demix.analysis.segment.create_segment_allele_counts(
            segment_counts,
            phased_allele_counts,
        )

        segment_allele_counts.to_csv(self.get_analysis_filename('counts.tsv'), sep='\t', index=False)

        h_idxs = set()
        def candidate_h_filename_callback(h_idx):
            h_idxs.add(h_idx)
            return self.get_analysis_filename('h_init_{0}.pickle'.format(h_idx))

        demix.analysis.pipeline.init(
            self.get_analysis_filename('experiment_learn.pickle'),
            self.get_analysis_filename('model_learn.pickle'),
            self.get_analysis_filename('experiment_infer.pickle'),
            self.get_analysis_filename('model_infer.pickle'),
            candidate_h_filename_callback,
            self.get_analysis_filename('candidate_h_plot.pdf'),
            self.get_analysis_filename('counts.tsv'),
            breakpoint_filename,
        )

        with open(self.get_analysis_filename('h_idxs.pickle'), 'w') as f:
            pickle.dump(h_idxs, f)

        return len(h_idxs)


    def run(self, init_param_idx):

        with open(self.get_analysis_filename('h_idxs.pickle'), 'r') as f:
            h_idxs = pickle.load(f)

        if init_param_idx not in h_idxs:
            raise utils.InvalidInitParam()

        demix.analysis.pipeline.learn_h(
            self.get_analysis_filename('h_opt_{0}.pickle'.format(init_param_idx)),
            self.get_analysis_filename('experiment_learn.pickle'),
            self.get_analysis_filename('model_learn.pickle'),
            self.get_analysis_filename('h_init_{0}.pickle'.format(init_param_idx)),
        )


    def report(self, output_cn_filename, output_mix_filename):

        with open(self.get_analysis_filename('h_idxs.pickle'), 'r') as f:
            h_idxs = pickle.load(f)

        h_opt_filenames = dict()
        for h_idx in h_idxs:
            h_opt_filenames[h_idx] = self.get_analysis_filename('h_opt_{0}.pickle'.format(h_idx))

        demix.analysis.pipeline.tabulate_h(self.get_analysis_filename('h_table.tsv'), h_opt_filenames)

        demix.analysis.pipeline.infer_cn(
            output_cn_filename,
            self.get_analysis_filename('brk_cn.tsv'),
            self.get_analysis_filename('experiment_plot.pdf'),
            output_mix_filename,
            self.get_analysis_filename('experiment_infer.pickle'),
            self.get_analysis_filename('model_infer.pickle'),
            self.get_analysis_filename('h_table.tsv'),
        )


if __name__ == '__main__':

    cmdline.interface(DemixTool)



