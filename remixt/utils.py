import csv
import os
import string
import shutil
import itertools
import collections
import bisect
import numpy as np
import pandas as pd
import pypeliner.commandline


class TempRandomSeed(object):
    def __init__(self, seed=1234):
        self.seed = seed
    def __enter__(self):
        self.rng_state = np.random.get_state()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            np.random.set_state(self.rng_state)


def weighted_resample(data, weights, num_samples=10000):
    norm_weights = weights.astype(float) / float(weights.sum())
    with TempRandomSeed():
        counts = np.random.multinomial(num_samples, norm_weights)
    samples = np.repeat(data, counts)
    return samples


def weighted_percentile(data, weights, percentile, num_samples=10000):
    data = weighted_resample(data, weights, num_samples=num_samples)
    return np.percentile(data, percentile)


def read_sequences(fasta_filename):
    with open(fasta_filename, 'rt') as fasta_file:
        seq_id = None
        sequences = []
        for line in fasta_file.readlines():
            line = str(line.rstrip())
            if len(line) == 0:
                continue
            if line[0] == '>':
                if seq_id is not None:
                    yield (seq_id, ''.join(sequences))
                seq_id = line[1:].split()[0]
                sequences = []
            else:
                sequences.append(line)
        if seq_id is not None:
            yield (seq_id, ''.join(sequences))


def write_sequence(fasta, seq_id, sequence):
    fasta.write('>{0}\n'.format(seq_id))
    idx = 0
    while idx < len(sequence):
        line_seq = sequence[idx:idx+80]
        idx += 80
        if line_seq == '':
            continue
        fasta.write(line_seq)
        fasta.write('\n')


def reverse_complement(sequence):
    return sequence[::-1].translate(string.maketrans('ACTGactg','TGACtgac'))


def read_chromosome_lengths(genome_fai_filename):
    chromosome_lengths = dict()
    with open(genome_fai_filename, 'r') as genome_fai_file:
        for row in csv.reader(genome_fai_file, delimiter='\t'):
            chromosome = row[0]
            length = int(row[1])
            chromosome_lengths[chromosome] = length
    return chromosome_lengths


def merge_files(output_filename, *input_filenames):
    with open(output_filename, 'w') as output_file:
        for input_filename in input_filenames:
            with open(input_filename, 'r') as input_file:
                shutil.copyfileobj(input_file, output_file)


def read_table_raw(filename):
    peek = pd.read_csv(filename, sep='\t', nrows=1)
    columns = peek.columns
    dtypes = dict(zip(columns, itertools.repeat(str, len(columns))))
    return pd.read_csv(filename, sep='\t', dtype=dtypes)


def split_table(output_filenames, input_filename, num_rows):
    input_data = read_table_raw(input_filename)
    for idx, start_row in enumerate(range(0, len(input_data.index), num_rows)):
        input_data.iloc[start_row:start_row+num_rows,].to_csv(output_filenames[idx], sep='\t', index=False)


def merge_tables(output_filename, *input_filenames):
    if len(input_filenames) == 1 and isinstance(input_filenames[0], dict):
        input_filenames = input_filenames[0].values()
    input_data = [read_table_raw(fname) for fname in input_filenames]
    pd.concat(input_data).to_csv(output_filename, sep='\t', index=False)


def link_file(target_filename, link_filename):
    try:
        os.remove(link_filename)
    except OSError:
        pass
    os.symlink(os.path.abspath(target_filename), link_filename)


def sort_chromosome_names(chromosomes):
    def get_chromosome_key(chromosome):
        try:
            return (0, int(chromosome))
        except ValueError:
            return (1, chromosome)
    return [chromosome for chromosome in sorted(chromosomes, key=get_chromosome_key)]


class BreakpointDatabase(object):
    def __init__(self, breakpoints):
        """ Create a database of breakpoints.

        Args:
            breakpoints (pandas.DataFrame): table of breakpoints

        Breakpoints table expects the following columns:
            'prediction_id', 'chromosome_1', 'strand_1', 'position_1',
            'chromosome_2', 'strand_2', 'position_2'
        """
        self.positions = collections.defaultdict(list)
        self.prediction_ids = collections.defaultdict(set)
        cols = [
            'prediction_id',
            'chromosome_1', 'strand_1', 'position_1',
            'chromosome_2', 'strand_2', 'position_2'
        ]
        for idx, row in breakpoints[cols].drop_duplicates().iterrows():
            for side in ('1', '2'):
                self.positions[(row['chromosome_'+side], row['strand_'+side])].append(row['position_'+side])
                self.prediction_ids[(row['chromosome_'+side], row['strand_'+side], row['position_'+side])].add((row['prediction_id'], side))
        for key in self.positions.keys():
            self.positions[key] = sorted(self.positions[key])

    def query(self, row, extend=0):
        """ Query the database for a breakpoint.

        Args:
            row (mapping): breakpoint information

        KwArgs:
            extend (int): inexact search range

        Breakpoint information expects the following keys:
            'chromosome_1', 'strand_1', 'position_1',
            'chromosome_2', 'strand_2', 'position_2'
        """
        matched_ids = list()
        for side in ('1', '2'):
            chrom_strand_positions = self.positions[(row['chromosome_'+side], row['strand_'+side])]
            idx = bisect.bisect_left(chrom_strand_positions, row['position_'+side] - extend)
            side_matched_ids = list()
            while idx < len(chrom_strand_positions):
                pos = chrom_strand_positions[idx]
                dist = abs(pos - row['position_'+side])
                if pos >= row['position_'+side] - extend and pos <= row['position_'+side] + extend:
                    for prediction_id in self.prediction_ids[(row['chromosome_'+side], row['strand_'+side], pos)]:
                        side_matched_ids.append((prediction_id, dist))
                if pos > row['position_'+side] + extend:
                    break
                idx += 1
            matched_ids.append(side_matched_ids)
        matched_ids_bypos = list()
        for matched_id_1, dist_1 in matched_ids[0]:
            for matched_id_2, dist_2 in matched_ids[1]:
                if matched_id_1[0] == matched_id_2[0] and matched_id_1[1] != matched_id_2[1]:
                    matched_ids_bypos.append((dist_1 + dist_2, matched_id_1[0]))
        if len(matched_ids_bypos) == 0:
            return None
        return sorted(matched_ids_bypos)[0][1]


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

