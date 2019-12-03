import os
import gzip
import pypeliner.commandline

import remixt.config
import remixt.utils


def create_ref_data(config, ref_data_dir, ref_data_sentinal, bwa_index_genome=False):
    try:
        os.makedirs(ref_data_dir)
    except OSError:
        pass

    auto_sentinal = remixt.utils.AutoSentinal(ref_data_dir + '/sentinal.')

    temp_directory = os.path.join(ref_data_dir, 'tmp')

    try:
        os.makedirs(temp_directory)
    except OSError:
        pass

    def wget_genome_fasta():
        with open(remixt.config.get_filename(config, ref_data_dir, 'genome_fasta'), 'w') as genome_file:
            for assembly in remixt.config.get_param(config, 'ensembl_assemblies'):
                assembly_url = remixt.config.get_filename(config, ref_data_dir, 'ensembl_assembly_url', ensembl_assembly=assembly)
                assembly_fasta = os.path.join(temp_directory, 'dna.assembly.{0}.fa'.format(assembly))
                if not os.path.exists(assembly_fasta):
                    remixt.utils.wget_gunzip(assembly_url, assembly_fasta)
                with open(assembly_fasta, 'r') as assembly_file:
                    for line in assembly_file:
                        if line[0] == '>':
                            line = line.split()[0] + '\n'
                        genome_file.write(line)
    auto_sentinal.run(wget_genome_fasta)

    def wget_gap_table():
        remixt.utils.wget(remixt.config.get_filename(config, ref_data_dir, 'gap_url'), remixt.config.get_filename(config, ref_data_dir, 'gap_table'))
    auto_sentinal.run(wget_gap_table)

    if bwa_index_genome:
        def bwa_index():
            pypeliner.commandline.execute('bwa', 'index', remixt.config.get_filename(config, ref_data_dir, 'genome_fasta'))
        auto_sentinal.run(bwa_index)

    def samtools_faidx():
        pypeliner.commandline.execute('samtools', 'faidx', remixt.config.get_filename(config, ref_data_dir, 'genome_fasta'))
    auto_sentinal.run(samtools_faidx)

    def wget_thousand_genomes():
        tar_filename = os.path.join(temp_directory, 'thousand_genomes_download.tar.gz')
        remixt.utils.wget(remixt.config.get_param(config, 'thousand_genomes_impute_url'), tar_filename)
        pypeliner.commandline.execute('tar', '-C', ref_data_dir, '-xzvf', tar_filename)
        os.remove(tar_filename)
    auto_sentinal.run(wget_thousand_genomes)

    def create_snp_positions():
        with open(remixt.config.get_filename(config, ref_data_dir, 'snp_positions'), 'w') as snp_positions_file:
            for chromosome in remixt.config.get_chromosomes(config, ref_data_dir):
                phased_chromosome = chromosome
                if chromosome == 'X':
                    phased_chromosome = remixt.config.get_param(config, 'phased_chromosome_x')
                legend_filename = remixt.config.get_filename(config, ref_data_dir, 'legend', chromosome=phased_chromosome)
                with gzip.open(legend_filename, 'rt') as legend_file:
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

    with open(ref_data_sentinal, 'w'):
        pass

