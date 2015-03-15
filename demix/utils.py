import csv
import string
import shutil
import scipy
import scipy.stats
import itertools
import numpy as np
import pandas as pd


class gaussian_kde_set_covariance(scipy.stats.gaussian_kde):
    def __init__(self, dataset, covariance):
        self.covariance = covariance
        scipy.stats.gaussian_kde.__init__(self, dataset)
    def _compute_covariance(self):
        self.inv_cov = 1.0 / self.covariance
        self._norm_factor = np.sqrt(2*np.pi*self.covariance) * self.n


def filled_density(ax, data, c, a, xmin, xmax, cov):
    density = gaussian_kde_set_covariance(data, cov)
    xs = [xmin] + list(np.linspace(xmin, xmax, 2000)) + [xmax]
    ys = density(xs)
    ys[0] = 0.0
    ys[-1] = 0.0
    ax.plot(xs, ys, color=c, alpha=a)
    ax.fill(xs, ys, color=c, alpha=a)


def weighted_resample(data, weights, num_samples=10000, randomize=False):
    norm_weights = weights.astype(float) / float(weights.sum())
    if randomize:
        counts = np.random.multinomial(num_samples, norm_weights)
    else:
        counts = np.round(norm_weights * float(num_samples)).astype(int)
    samples = np.repeat(data, counts)
    return samples


def filled_density_weighted(ax, data, weights, c, a, xmim, xmax, cov):
    weights = weights.astype(float)
    resample_prob = weights / weights.sum()
    samples = np.random.choice(data, size=10000, replace=True, p=resample_prob)
    filled_density(ax, samples, c, a, xmim, xmax, cov)


def read_sequences(fasta_filename):
    with open(fasta_filename, 'r') as fasta_file:
        seq_id = None
        sequences = []
        for line in fasta_file:
            line = line.rstrip()
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
            if chromosome.startswith('GL'):
                continue
            if chromosome == 'MT':
                continue
            chromosome_lengths[chromosome] = length
    return chromosome_lengths


def merge_files(output_filename, *input_filenames):
    with open(output_filename, 'w') as output_file:
        for input_filename in input_filenames:
            with open(input_filename, 'r') as input_file:
                shutil.copyfileobj(input_file, output_file)


def merge_tables(output_filename, *input_filenames):
    if len(input_filenames) == 1 and isinstance(input_filenames[0], dict):
        input_filenames = input_filenames[0].values()
    input_data = [pd.read_csv(fname, sep='\t', dtype=str) for fname in input_filenames]
    pd.concat(input_data).to_csv(output_filename, sep='\t', index=False)


def link_file(target_filename, link_filename):
    try:
        os.remove(link_filename)
    except OSError:
        pass
    os.symlink(os.path.abspath(target_filename), link_filename)



