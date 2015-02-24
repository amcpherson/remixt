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

import utils.io

if __name__ == '__main__':
	
	import mappability_bwa
	
	argparser = argparse.ArgumentParser()
	pypeliner.easypypeliner.add_arguments(argparser)
	
	cfg = pypeliner.easypypeliner.Config(vars(argparser.parse_args()))
	pyp = pypeliner.easypypeliner.EasyPypeliner([mappability_bwa], cfg)
	ctx = {'mem':6}
	
	pyp.sch.transform('create_kmers', (), ctx, mappability_bwa.create_kmers, None, pyp.sch.input(cfg.genome_fasta), int(cfg.mappability_length), pyp.sch.ofile('kmers'))
	pyp.sch.transform('split_kmers', (), ctx, mappability_bwa.split_file_byline, None, pyp.sch.ifile('kmers'), 4000000, pyp.sch.ofile('kmers', ('bykmer',)))
	pyp.sch.commandline('bwa_aln_kmers', ('bykmer',), ctx, cfg.bwa_bin, 'aln', pyp.sch.input(cfg.genome_fasta), pyp.sch.ifile('kmers', ('bykmer',)), '>', pyp.sch.ofile('sai', ('bykmer',)))
	pyp.sch.commandline('bwa_samse_kmers', ('bykmer',), ctx, cfg.bwa_bin, 'samse', pyp.sch.input(cfg.genome_fasta), pyp.sch.ifile('sai', ('bykmer',)), pyp.sch.ifile('kmers', ('bykmer',)), '>', pyp.sch.ofile('alignments', ('bykmer',)))
	pyp.sch.transform('create_bedgraph', ('bykmer',), ctx, mappability_bwa.create_bedgraph, None, pyp.sch.ifile('alignments', ('bykmer',)), pyp.sch.ofile('bedgraph', ('bykmer',)))
	pyp.sch.transform('merge_bedgraph', (), ctx, mappability_bwa.merge_files_by_line, None, pyp.sch.ifile('bedgraph', ('bykmer',)), pyp.sch.output(cfg.mappability_filename))
	pyp.run()
	
def create_kmers(genome_fasta, k, kmers_filename):
	with open(kmers_filename, 'w') as kmers_file:
		genome_sequences = dict(utils.io.read_sequences(open(genome_fasta, 'r')))
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

