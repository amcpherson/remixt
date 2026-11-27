import csv
import os
import itertools
import numpy as np
import pandas as pd
import subprocess


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


def reverse_complement(sequence):
    return sequence[::-1].translate(str.maketrans('ACTGactg', 'TGACtgac'))


def read_chromosome_lengths(genome_fai_filename):
    chromosome_lengths = dict()
    with open(genome_fai_filename, 'r') as genome_fai_file:
        for row in csv.reader(genome_fai_file, delimiter='\t'):
            chromosome = row[0]
            length = int(row[1])
            chromosome_lengths[chromosome] = length
    return chromosome_lengths


def read_selected_chromosome_lengths(genome_fai_filename, chromosomes, chr_name_prefix=''):
    """ Read lengths of a selection of chromosomes from a .fai file, in .fai order
    """
    chromosome_lengths = read_chromosome_lengths(genome_fai_filename)

    # Check chromosomes against chromosome lengths from .fai file
    assert set(chromosomes).issubset(set(chromosome_lengths.keys()))

    # Validate chr name prefix
    if chr_name_prefix == 'chr':
        assert all(c.startswith('chr') for c in chromosomes)
    elif chr_name_prefix == '':
        assert not any(c.startswith('chr') for c in chromosomes)
    else:
        raise ValueError(f'unrecognized chr_name_prefix {chr_name_prefix}')

    return {c: l for c, l in chromosome_lengths.items() if c in chromosomes}


breakpoint_columns = [
    'prediction_id',
    'chromosome_1',
    'strand_1',
    'position_1',
    'chromosome_2',
    'strand_2',
    'position_2',
]


def read_breakpoints(breakpoint_filename):
    """ Read a breakpoints TSV, allowing for no breakpoints.

    Args:
        breakpoint_filename (str or None): breakpoints TSV file

    Returns:
        pandas.DataFrame: breakpoints with columns given by `breakpoint_columns`

    No breakpoints can be given as None, an empty file, or a file with only a
    header, and results in an empty table with the expected columns.

    """

    if breakpoint_filename is None or os.path.getsize(breakpoint_filename) == 0:
        breakpoints = pd.DataFrame(columns=breakpoint_columns)

    else:
        breakpoints = pd.read_csv(
            breakpoint_filename, sep='\t',
            dtype={'chromosome_1': str, 'chromosome_2': str, 'strand_1': str, 'strand_2': str},
        )

        missing_columns = set(breakpoint_columns) - set(breakpoints.columns)
        if missing_columns:
            raise ValueError(f'breakpoint file {breakpoint_filename} missing columns {sorted(missing_columns)}')

    return breakpoints.astype({
        'chromosome_1': str, 'chromosome_2': str,
        'strand_1': str, 'strand_2': str,
        'position_1': int, 'position_2': int,
    })


def read_table_raw(filename):
    peek = pd.read_csv(filename, sep='\t', nrows=1)
    columns = peek.columns
    dtypes = dict(zip(columns, itertools.repeat(str, len(columns))))
    return pd.read_csv(filename, sep='\t', dtype=dtypes)


def merge_tables(output_filename, *input_filenames):
    if len(input_filenames) == 1 and isinstance(input_filenames[0], dict):
        input_filenames = input_filenames[0].values()
    input_data = [read_table_raw(fname) for fname in input_filenames]
    pd.concat(input_data).to_csv(output_filename, sep='\t', index=False)


def sort_chromosome_names(chromosomes):
    def get_chromosome_key(chromosome):
        try:
            return (0, int(chromosome))
        except ValueError:
            return (1, chromosome)
    return [chromosome for chromosome in sorted(chromosomes, key=get_chromosome_key)]


def execute(*args):
    """ Execute a command line, raising subprocess.CalledProcessError on failure

    Empty string arguments are dropped, allowing optional flags to be passed as ''.
    """
    subprocess.run([str(arg) for arg in args if arg != ''], check=True)


def wget_gunzip(url, filename):
    temp_filename = filename + '.tmp'
    execute('wget', url, '-c', '-O', temp_filename + '.gz')
    execute('gunzip', temp_filename + '.gz')
    os.rename(temp_filename, filename)


def wget(url, filename):
    temp_filename = filename + '.tmp'
    execute('wget', url, '-c', '-O', temp_filename)
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
