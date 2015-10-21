import sys
import csv
import subprocess
from collections import *
import math
import random
import argparse
import uuid
import os
import pickle
import gzip
import itertools

import pypeliner
import pypeliner.managed as mgd

import remixt
import remixt.utils


remixt_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))

default_config_filename = os.path.join(remixt_directory, 'defaultconfig.py')


if __name__ == '__main__':
    
    import mappability_bwa

    argparser = argparse.ArgumentParser()

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    config = {'ref_data_directory':args['ref_data_dir']}
    execfile(default_config_filename, {}, config)

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([mappability_bwa], config)

    ctx = {'mem':6}
    
    pyp.sch.transform('create_kmers', (), ctx,
        mappability_bwa.create_kmers,
        None,
        mgd.InputFile(config['genome_fasta']),
        int(config['mappability_length']),
        mgd.TempOutputFile('kmers'))

    pyp.sch.transform('split_kmers', (), ctx,
        mappability_bwa.split_file_byline,
        None,
        mgd.TempInputFile('kmers'),
        4000000,
        mgd.TempOutputFile('kmers', 'bykmer'))

    pyp.sch.commandline('bwa_aln_kmers', ('bykmer',), ctx,
        'bwa',
        'aln',
        mgd.InputFile(config['genome_fasta']),
        mgd.TempInputFile('kmers', 'bykmer'),
        '>',
        mgd.TempOutputFile('sai', 'bykmer'))

    pyp.sch.commandline('bwa_samse_kmers', ('bykmer',), ctx,
        'bwa',
        'samse',
        mgd.InputFile(config['genome_fasta']),
        mgd.TempInputFile('sai', 'bykmer'),
        mgd.TempInputFile('kmers', 'bykmer'),
        '>',
        mgd.TempOutputFile('alignments', 'bykmer'))

    pyp.sch.transform('create_bedgraph', ('bykmer',), ctx,
        mappability_bwa.create_bedgraph,
        None,
        mgd.TempInputFile('alignments', 'bykmer'),
        mgd.TempOutputFile('bedgraph', 'bykmer'))

    pyp.sch.transform('merge_bedgraph', (), ctx,
        mappability_bwa.merge_files_by_line,
        None,
        mgd.TempInputFile('bedgraph', 'bykmer'),
        mgd.OutputFile(config['mappability_filename']))

    pyp.run()
    
def create_kmers(genome_fasta, k, kmers_filename):
    with open(kmers_filename, 'w') as kmers_file:
        genome_sequences = dict(remixt.utils.read_sequences(genome_fasta))
        for chromosome, sequence in genome_sequences.iteritems():
            chromosome = chromosome.split()[0]
            for start in xrange(len(sequence)):
                kmer = sequence[start:start+k].upper()
                if len(kmer) < k:
                    continue
                if 'N' in kmer:
                    continue
                kmers_file.write('>{0}:{1}\n{2}\n'.format(chromosome, start, kmer))

def split_file_byline(in_filename, lines_per_file, out_filename_callback):
    with open(in_filename, 'r') as in_file:
        file_number = 0
        out_file = None
        out_file_lines = None
        try:
            for line in in_file:
                if out_file is None or out_file_lines == lines_per_file:
                    if out_file is not None:
                        out_file.close()
                    out_file = open(out_filename_callback(file_number), 'w')
                    out_file_lines = 0
                    file_number += 1
                out_file.write(line)
                out_file_lines += 1
        finally:
            if out_file is not None:
                out_file.close()

def create_bedgraph(alignment_filename, bedgraph_filename):
    mappable = defaultdict(list)
    with open(alignment_filename, 'r') as alignment_file:
        for row in csv.reader(alignment_file, delimiter='\t'):
            if row[0][0] == '@':
                continue
            if int(row[4]) != 0:
                chromosome, position = row[0].split(':')
                position = int(position)
                mappable[chromosome].append(position)
    with open(bedgraph_filename, 'w') as bedgraph_file:
        for chromosome, positions in mappable.iteritems():
            positions = sorted(positions)
            for idx, adjacent_positions in itertools.groupby(positions, lambda n, c=itertools.count(positions[0]): n-next(c)):
                adjacent_positions = list(adjacent_positions)
                bedgraph_file.write('{0}\t{1}\t{2}\t{3}\n'.format(chromosome, min(adjacent_positions), max(adjacent_positions) + 1, 1))
                
def merge_files_by_line(in_filenames, out_filename):
    pypeliner.commandline.execute(*(['cat'] + in_filenames.values() + ['>', out_filename]))

