import string
import scipy
import scipy.stats
import itertools
import numpy as np

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


def weighted_resample(data, weights, num_samples):
    weights = weights.astype(float)
    idx_counts = np.random.multinomial(num_samples, weights / weights.sum())
    samples = np.array(list(itertools.chain(*[itertools.repeat(idx, cnt) for idx, cnt in enumerate(np.random.multinomial(10000, weights.astype(float) / weights.sum()))])))


def filled_density_weighted(ax, data, weights, c, a, xmim, xmax, cov):
    weights = weights.astype(float)
    resample_prob = weights / weights.sum()
    samples = np.random.choice(data, size=10000, replace=True, p=resample_prob)
    filled_density(ax, samples, c, a, xmim, xmax, cov)


def read_sequences(fasta):
    seq_id = None
    sequences = []
    for line in fasta:
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


