import os
import argparse
import gzip

import pypeliner

import remixt.config


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
        with open(sentinal_filename, 'w'):
            pass

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('-c', '--config',
        help='Configuration filename')

    args = argparser.parse_args()

    args = vars(argparser.parse_args())

    config = {'ref_data_directory': args['ref_data_dir']}

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
        with open(remixt.config.get_filename(config, 'genome_fasta'), 'w') as genome_file:
            for assembly in remixt.config.get_param(config, 'ensembl_assemblies'):
                assembly_url = remixt.config.get_filename(config, 'ensembl_assembly_url', ensembl_assembly=assembly)
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
        pypeliner.commandline.execute('bwa', 'index', remixt.config.get_filename(config, 'genome_fasta'))
    auto_sentinal.run(bwa_index)

    def samtools_faidx():
        pypeliner.commandline.execute('samtools', 'faidx', remixt.config.get_filename(config, 'genome_fasta'))
    auto_sentinal.run(samtools_faidx)

    def wget_thousand_genomes():
        tar_filename = os.path.join(temp_directory, 'thousand_genomes_download.tar.gz')
        wget(remixt.config.get_param(config, 'thousand_genomes_impute_url'), tar_filename)
        pypeliner.commandline.execute('tar', '-C', args['ref_data_dir'], '-xzvf', tar_filename)
        os.remove(tar_filename)
    auto_sentinal.run(wget_thousand_genomes)

    def create_snp_positions():
        with open(remixt.config.get_filename(config, 'snp_positions'), 'w') as snp_positions_file:
            for chromosome in remixt.config.get_param(config, 'chromosomes'):
                phased_chromosome = chromosome
                if chromosome == 'X':
                    phased_chromosome = remixt.config.get_param(config, 'phased_chromosome_x')
                legend_filename = remixt.config.get_filename(config, 'legend', chromosome=phased_chromosome)
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

