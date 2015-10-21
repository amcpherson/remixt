import csv
import sys
import logging
import os
import re
import itertools
import subprocess
import argparse
import string
import gzip
from collections import *

import pypeliner


remixt_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))

default_config_filename = os.path.join(remixt_directory, 'defaultconfig.py')


def wget_gunzip(url, filename):
    temp_filename = filename + '.tmp'
    pypeliner.commandline.execute('wget', url, '-c', '-O', temp_filename + '.gz')
    pypeliner.commandline.execute('gunzip', temp_filename + '.gz')
    os.rename(temp_filename, filename)

def wget(url, filename):
    temp_filename = filename + '.tmp'
    pypeliner.commandline.execute('wget', url, '-c', '-O', temp_filename)
    os.rename(temp_filename, filename)

class AutoSentinal(object):
    def __init__(self, sentinal_prefix):
        self.sentinal_prefix = sentinal_prefix
    def run(self, func):
        sentinal_filename = self.sentinal_prefix + func.__name__
        if os.path.exists(sentinal_filename):
            return
        func()
        with open(sentinal_filename, 'w') as sentinal_file:
            pass

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('-c', '--config',
        help='Configuration filename')

    args = argparser.parse_args()

    args = vars(argparser.parse_args())

    config = {'ref_data_directory':args['ref_data_dir']}
    execfile(default_config_filename, {}, config)

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    try:
        os.makedirs(args['ref_data_dir'])
    except OSError:
        pass

    auto_sentinal = AutoSentinal(args['ref_data_dir'] + '/sentinal.')

    temp_directory = os.path.join(args['ref_data_dir'], 'tmp')

    try:
        os.makedirs(temp_directory)
    except OSError:
        pass

    def wget_genome_fasta():
        with open(config['genome_fasta'], 'w') as genome_file:
            for assembly in config['ensembl_assemblies']:
                assembly_url = config['ensembl_assembly_url'].format(assembly)
                assembly_fasta = os.path.join(temp_directory, 'dna.assembly.{0}.fa'.format(assembly))
                if not os.path.exists(assembly_fasta):
                    wget_gunzip(assembly_url, assembly_fasta)
                with open(assembly_fasta, 'r') as assembly_file:
                    for line in assembly_file:
                        if line[0] == '>':
                            line = line.split()[0] + '\n'
                        genome_file.write(line)
    auto_sentinal.run(wget_genome_fasta)

    def bwa_index():
        pypeliner.commandline.execute('bwa', 'index', config['genome_fasta'])
    auto_sentinal.run(bwa_index)

    def samtools_faidx():
        pypeliner.commandline.execute('samtools', 'faidx', config['genome_fasta'])
    auto_sentinal.run(samtools_faidx)

    def wget_thousand_genomes():
        tar_filename = os.path.join(temp_directory, 'thousand_genomes_download.tar.gz')
        wget(config['thousand_genomes_impute_url'], tar_filename)
        pypeliner.commandline.execute('tar', '-C', args['ref_data_dir'], '-xzvf', tar_filename)
        os.remove(tar_filename)
    auto_sentinal.run(wget_thousand_genomes)

    def create_snp_positions():
        with open(config['snp_positions'], 'w') as snp_positions_file:
            for chromosome in config['chromosomes']:
                phased_chromosome = chromosome
                if chromosome == 'X':
                    phased_chromosome = config['phased_chromosome_x']
                legend_filename = config['legend_template'].format(phased_chromosome)
                with gzip.open(legend_filename, 'r') as legend_file:
                    for line in legend_file:
                        if line.startswith('id'):
                            continue
                        row = line.split()
                        rs_id = row[0]
                        position = row[1]
                        a0 = row[2]
                        a1 = row[3]
                        if len(a0) != 1 or len(a1) != 1:
                            continue
                        snp_positions_file.write('\t'.join([chromosome, position, a0, a1]) + '\n')
    auto_sentinal.run(create_snp_positions)

