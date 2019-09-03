import math
import collections
import numpy as np
import pandas as pd
import scipy
import scipy.stats

import remixt.utils
import remixt.likelihood
import remixt.analysis.experiment
import remixt.simulations.balanced


MAX_SEED = 2**32

class RearrangedGenome(object):
    """ Rearranged genome with stored history.

    Attributes:
        default_params (dict): dictionary of default simulation parameters
        chromosomes (list of list of tuple): list of chromosome, each chromosome a list of 'segment copy'
        wt_adj (set): set of 'breakpoints' representing wild type adjacencies
        init_params (dict): parameters for initializing chromosome structure.
        init_seed (int) seed for random initialization of chromosome structure.
        event_params (list of dict): list of parameters for randomly selected events.
        event_seeds (list of int): list of seeds used to generate randomly selected events.

    A 'segment copy' is represented as the tuple (('segment', 'allele'), 'orientation').

    A 'breakend' is represented as the tuple (('segment', 'allele'), 'side').

    A 'breakpoint' is represented as the frozenset (['breakend_1', 'breakend_2'])

    """

    default_params = {
        'genome_length':3e9,
        'seg_length_concentration':1.0,
        'seg_length_min':50000,
        'num_chromosomes':20,
        'chrom_length_concentration':5.,
        'chromosome_lengths':None,
        'event_type':['dcj', 'dup', 'del', 'wgd'],
        'event_prob':[0.19, 0.3, 0.5, 0.01],
        'del_prop_len':0.5,
        'dup_prop_len':0.5,
        'wgd_prop_dup':0.8,
    }

    def __init__(self, N):
        """ Create an empty genome.

        Args:
            N (int): number of segments

        """
        self.N = N

        self.init_params = None
        self.init_seed = None

        self.event_params = list()
        self.event_seeds = list()


    def copy(self):
        """ Create a copy of the genome.

        Creates a copy such that rearrangement of the copy will not
        affect the original.
        """

        genome = RearrangedGenome(self.N)

        # References to fixed attributes of each genome
        genome.init_params = self.init_params
        genome.init_seed = self.init_seed
        genome.segment_start = self.segment_start
        genome.segment_end = self.segment_end
        genome.segment_chromosome_id = self.segment_chromosome_id
        genome.l = self.l
        genome.wt_adj = self.wt_adj

        # Copies of mutable attributes of each genome
        genome.event_params = list(self.event_params)
        genome.event_seeds = list(self.event_seeds)
        genome.chromosomes = list(self.chromosomes)

        return genome


    def create(self, params):
        """ Create a new non-rearranged genome.

        Args:
            params (dict): parameters for random chromosome creation

        Sets the seed and updates the init seed and params.

        """
        seed = np.random.randint(MAX_SEED - 1)

        np.random.seed(seed)

        self.random_chromosomes(params)

        self.init_params = params
        self.init_seed = seed


    def rewind(self, num_events):
        """ Rewind event list to more ancestral genome.

        Args:
            num_events (int): number of ancestral events from which to create genome

        """
        self.event_params = self.event_params[:num_events]
        self.event_seeds = self.event_seeds[:num_events]

        self.recreate()


    def recreate(self):
        """ Recreate a genome based on event list.
        """
        np.random.seed(self.init_seed)

        self.random_chromosomes(self.init_params)

        for params, seed in zip(self.event_params, self.event_seeds):

            np.random.seed(seed)

            self.random_event(params)


    def random_chromosomes(self, params):
        """ Create a random set of chromosomes.

        Args:
            params (dict): parameters for random chromosome creation

        """
        if params.get('chromosome_lengths', None) is not None:

            chromosome_ids = list(params['chromosome_lengths'].keys())
            chromosome_lengths = np.array(list(params['chromosome_lengths'].values()))

        else:

            num_chroms = params['num_chromosomes']
            genome_length = params['genome_length']
            chrom_length_concentration = params['chrom_length_concentration']

            chromosome_ids = [str(a) for a in range(1, num_chroms + 1)]
            chromosome_lengths = np.random.dirichlet([chrom_length_concentration] * num_chroms) * genome_length
            chromosome_lengths.sort_values()
            chromosome_lengths = chromosome_lengths[::-1]

        chrom_pvals = chromosome_lengths.astype(float) / float(chromosome_lengths.sum())
        chrom_num_segments = np.random.multinomial(self.N - len(chromosome_lengths), pvals=chrom_pvals)
        chrom_num_segments += 1

        seg_length_concentration = params['seg_length_concentration']
        seg_length_min = params['seg_length_min']

        self.l = np.array([])

        self.segment_chromosome_id = np.array([], dtype=str)
        self.segment_start = np.array([], dtype=int)
        self.segment_end = np.array([], dtype=int)

        for chrom_id, chrom_length, num_segments in zip(chromosome_ids, chromosome_lengths, chrom_num_segments):

            length_proportions = np.random.dirichlet([seg_length_concentration] * num_segments)
            length_proportions = np.maximum(length_proportions, float(seg_length_min) / chrom_length)
            length_proportions /= length_proportions.sum()
            lengths = length_proportions * chrom_length
            lengths = lengths.astype(int)
            lengths[-1] = chrom_length - lengths[:-1].sum()

            assert lengths[-1] > 0

            chrom_ids = [chrom_id] * num_segments
            ends = lengths.cumsum()
            starts = ends - lengths

            self.l = np.concatenate((self.l, lengths))

            self.segment_chromosome_id = np.concatenate((self.segment_chromosome_id, chrom_ids))
            self.segment_start = np.concatenate((self.segment_start, starts))
            self.segment_end = np.concatenate((self.segment_end, ends))

        segment_idx = 0

        self.chromosomes = list()

        for num_seg in chrom_num_segments:

            for allele in (0, 1):

                chrom_segs = range(segment_idx, segment_idx+num_seg)
                chrom_alleles = [allele]*num_seg
                chrom_orient = [1]*num_seg

                self.chromosomes.append(tuple(zip(zip(chrom_segs, chrom_alleles), chrom_orient)))

            segment_idx += num_seg

        self.wt_adj = set()
        self.wt_adj = set(self.breakpoints)


    def generate_cuts(self):
        """ Generate a list of possible cuts.
        
        Cuts are triples of chromosome index, segment index where
        the segment index is the second in an adjacent pair

        """
        for chromosome_idx, chromosome in enumerate(self.chromosomes):
            for segment_idx in range(len(chromosome)):
                next_segment_idx = (segment_idx + 1) % len(chromosome)
                yield (chromosome_idx, next_segment_idx)


    def random_cut(self):
        """ Sample a random cut
        """
        cuts = list(self.generate_cuts())
        idx = np.random.choice(range(len(cuts)))
        return cuts[idx]

    
    def random_cut_pair(self):
        """ Sample a random cut pair without replacement
        """
        cuts = list(self.generate_cuts())
        idx1, idx2 = np.random.choice(range(len(cuts)), size=2, replace=False)
        return (cuts[idx1], cuts[idx2])

    
    def reverse_segment(self, segment):
        """ Reverse the sign of a segment.
        """
        return (segment[0], segment[1] * -1)
        

    def reverse_chromosome(self, chromosome):
        """ Reverse the order of segments in a chromosome, and the sign of each segment.
        """
        return tuple([self.reverse_segment(a) for a in reversed(chromosome)])


    def rearrange(self, params):
        """ Apply random rearrangement event.

        Args:
            params (dict): dictionary of modification params

        Sets the seed and appends the seed and params to the event lists.

        """
        seed = np.random.randint(MAX_SEED - 1)

        np.random.seed(seed)

        self.random_event(params)

        self.event_params.append(params)
        self.event_seeds.append(seed)


    def random_event(self, params):
        """ Randomly apply rearrangement event.

        Args:
            params (dict): dictionary of modification params

        """
        event = np.random.choice(params['event_type'], p=params['event_prob'])

        if event == 'dcj':
            self.random_double_cut_join(params)
        elif event == 'dup':
            self.random_duplication(params)
        elif event == 'del':
            self.random_deletion(params)
        elif event == 'wgd':
            self.random_whole_genome_doubling(params)


    def random_double_cut_join(self, params):
        """ Randomly break the genome at two locations and rejoin.

        Args:
            params (dict): dictionary of modification params

        """
        if len(self.chromosomes) < 2:
            return

        breakpoint_1, breakpoint_2 = sorted(self.random_cut_pair())
        
        dcj_flip = np.random.choice([True, False])
        
        if breakpoint_1[0] != breakpoint_2[0]:
            
            chromosome_1 = self.chromosomes[breakpoint_1[0]]
            chromosome_2 = self.chromosomes[breakpoint_2[0]]
            
            del self.chromosomes[breakpoint_1[0]]
            del self.chromosomes[breakpoint_2[0] - 1]
            
            if dcj_flip:
                
                # Create a new chromosome with chromosome 2 segments reversed
                new_chromosome = chromosome_1[:breakpoint_1[1]] + \
                                 self.reverse_chromosome(chromosome_2[:breakpoint_2[1]]) + \
                                 self.reverse_chromosome(chromosome_2[breakpoint_2[1]:]) + \
                                 chromosome_1[breakpoint_1[1]:]
                assert len(new_chromosome) > 0
            
                self.chromosomes.append(new_chromosome)

            else:
            
                # Create a new chromosome with orientation preserved
                new_chromosome = chromosome_1[:breakpoint_1[1]] + \
                                 chromosome_2[breakpoint_2[1]:] + \
                                 chromosome_2[:breakpoint_2[1]] + \
                                 chromosome_1[breakpoint_1[1]:]
                assert len(new_chromosome) > 0

                self.chromosomes.append(new_chromosome)
        
        else:
            
            chromosome = self.chromosomes[breakpoint_1[0]]
            
            del self.chromosomes[breakpoint_1[0]]

            if dcj_flip:
                
                # Create a new chromosome with an inversion of some segments
                new_chromosome = chromosome[:breakpoint_1[1]] + \
                                 self.reverse_chromosome(chromosome[breakpoint_1[1]:breakpoint_2[1]]) + \
                                 chromosome[breakpoint_2[1]:]
                assert len(new_chromosome) > 0

                
                self.chromosomes.append(new_chromosome)

            else:
                
                # Create two new chromosomes with orientation preserved
                new_chromosome_1 = chromosome[:breakpoint_1[1]] + \
                                   chromosome[breakpoint_2[1]:]
                new_chromosome_2 = chromosome[breakpoint_1[1]:breakpoint_2[1]]
                assert len(new_chromosome_1) > 0
                assert len(new_chromosome_2) > 0

                self.chromosomes.append(new_chromosome_1)
                self.chromosomes.append(new_chromosome_2)
    
    
    def random_deletion(self, params):
        """ Randomly delete consecutive segments of a chromosome.

        Args:
            params (dict): dictionary of modification params

        """
        if len(self.chromosomes) == 0:
            return
        
        breakpoint_1 = self.random_cut()

        chromosome = self.chromosomes[breakpoint_1[0]]

        del self.chromosomes[breakpoint_1[0]]

        chrom_length = len(chromosome)
        
        deletion_length = np.random.randint(0, math.ceil(params['del_prop_len'] * chrom_length))
        
        if deletion_length == 0:
            return
        
        breakpoint_2 = (breakpoint_1[0], (breakpoint_1[1] + deletion_length) % chrom_length)
        
        if breakpoint_1[1] < breakpoint_2[1]:

            new_chromosome = chromosome[:breakpoint_1[1]] + \
                             chromosome[breakpoint_2[1]:]

            self.chromosomes.append(new_chromosome)
        
        else:
            
            new_chromosome = chromosome[breakpoint_2[1]:breakpoint_1[1]]
    
            self.chromosomes.append(new_chromosome)

    
    def random_duplication(self, params):
        """ Randomly duplicate consecutive segments of a chromosome.

        Args:
            params (dict): dictionary of modification params

        """
        if len(self.chromosomes) == 0:
            return
        
        breakpoint_1 = self.random_cut()

        chromosome = self.chromosomes[breakpoint_1[0]]

        del self.chromosomes[breakpoint_1[0]]

        chrom_length = len(chromosome)
        
        duplication_length = np.random.randint(0, math.ceil(params['dup_prop_len'] * chrom_length))
        
        breakpoint_2 = (breakpoint_1[0], (breakpoint_1[1] + duplication_length) % chrom_length)
        
        if breakpoint_1[1] < breakpoint_2[1]:

            new_chromosome = chromosome[:breakpoint_2[1]] + \
                             chromosome[breakpoint_1[1]:]

            self.chromosomes.append(new_chromosome)
        
        else:
            
            new_chromosome = chromosome + \
                             chromosome[:breakpoint_2[1]] + \
                             chromosome[breakpoint_1[1]:]
    
            self.chromosomes.append(new_chromosome)


    def random_whole_genome_doubling(self, params):
        """ Randomly select chromosomes to be duplicated.

        Args:
            params (dict): dictionary of modification params

        """

        duplicated_chromosomes = []
        for chromosome in self.chromosomes:
            if np.random.rand() < params['wgd_prop_dup']:
                duplicated_chromosomes.append(chromosome)

        self.chromosomes.extend(duplicated_chromosomes)


    @property
    def segment_copy_number(self):
        """ Segment copy number matrix (numpy.array).
        """
        cn_matrix = np.zeros((self.N, 2))

        for chromosome in self.chromosomes:
            for segment in chromosome:
                cn_matrix[segment[0][0], segment[0][1]] += 1.0
                
        return cn_matrix


    @property
    def breakpoint_copy_number(self):
        """ Breakpoint copy number (dict of breakpoint to integer copy number).
        """
        brk_cn = collections.Counter()

        for chromosome_idx, segment_idx_2 in self.generate_cuts():

            segment_idx_1 = (segment_idx_2 - 1) % len(self.chromosomes[chromosome_idx])

            segment_1 = self.chromosomes[chromosome_idx][segment_idx_1]
            segment_2 = self.chromosomes[chromosome_idx][segment_idx_2]

            side_1 = (0, 1)[segment_1[1] == 1]
            side_2 = (1, 0)[segment_2[1] == 1]

            brkend_1 = (segment_1[0], side_1)
            brkend_2 = (segment_2[0], side_2)

            breakpoint = frozenset([brkend_1, brkend_2])

            if breakpoint in self.wt_adj:
                continue

            brk_cn[breakpoint] += 1

        return brk_cn
    
    
    @property
    def breakpoints(self):
        """ Breakpoint list.
        """
        return list(self.breakpoint_copy_number.keys())


    def length_loh(self):
        """ Length of the genome with LOH.
        """
        cn = self.segment_copy_number
        loh = (cn.min(axis=1) == 0) * 1
        return (loh * self.l).sum()


    def proportion_loh(self):
        """ Proportion of the genome with LOH.
        """
        return self.length_loh() / float(self.l.sum())


    def length_hdel(self):
        """ Length of the genome homozygously deleted.
        """
        cn = self.segment_copy_number
        hdel = (cn.max(axis=1) == 0) * 1
        return (hdel * self.l).sum()


    def proportion_hdel(self):
        """ Proportion of the genome homozygously deleted.
        """
        return self.length_hdel() / float(self.l.sum())


    def length_hlamp(self, hlamp_min=6):
        """ Length of the genome with high level amplification.
        """
        cn = self.segment_copy_number
        hlamp = (cn.sum(axis=1) >= hlamp_min) * 1
        return (hlamp * self.l).sum()


    def proportion_hlamp(self, hlamp_min=6):
        """ Proportion of the genome with high level amplification.
        """
        return self.length_hlamp(hlamp_min=hlamp_min) / float(self.l.sum())


    def length_divergent(self, other):
        """ Length of the genome divergent from another genome.
        """
        cn = self.segment_copy_number
        other_cn = other.segment_copy_number
        divergent = ((cn - other_cn > 0) * 1).sum(axis=1)
        return (divergent * self.l).sum()


    def proportion_divergent(self, other):
        """ Proportion of the genome divergent from another genome.
        """
        return self.length_divergent(other) / float(self.l.sum())


    def ploidy(self):
        """ Average number of copies of each nucleotide.
        """
        cn = self.segment_copy_number
        cn = cn.sum(axis=1)
        return (cn * self.l).sum() / self.l.sum()


    def proportion_minor_state(self, cn_max=6):
        """ Proportion of the genome with minor in each cn state.
        """
        minor = self.segment_copy_number.min(axis=1)
        minor[minor > cn_max] = cn_max
        return np.bincount(minor.flatten().astype(int), weights=self.l, minlength=cn_max+1) / self.l.sum()
 
    def proportion_major_state(self, cn_max=6):
        """ Proportion of the genome with major in each cn state.
        """
        major = self.segment_copy_number.max(axis=1)
        major[major > cn_max] = cn_max
        return np.bincount(major.flatten().astype(int), weights=self.l, minlength=cn_max+1) / self.l.sum()


    def create_chromosome_sequences(self, germline_genome):
        """ Create sequences of rearranged chromosomes

        Args:
            ref_genome (dict): germline chromosome allele sequences keyed by (chromosome, allele_id)

        Returns:
            list: rearranged chromosome allele sequences

        """
        rearranged_genome = list()

        for chrom in self.chromosomes:

            rearranged_chromosome = list()

            for ((segment_idx, allele_id), orientation) in chrom:

                chromosome_id = self.segment_chromosome_id[segment_idx]
                start = self.segment_start[segment_idx]
                end = self.segment_end[segment_idx]

                segment_sequence = germline_genome[(chromosome_id, allele_id)][start:end]

                if orientation < 0:
                    segment_sequence = remixt.utils.reverse_complement(segment_sequence)

                rearranged_chromosome.append(segment_sequence)

            rearranged_genome.append(''.join(rearranged_chromosome))

        return rearranged_genome


def log_multinomial_likelihood(x, p):
    return scipy.special.gammaln(np.sum(x+1)) - np.sum(scipy.special.gammaln(x+1)) + np.sum(x * np.log(p))


class RearrangementHistorySampler(object):
    """ Simulate a random rearrangement history, accounting for relative fitness
    """

    def __init__(self, params):

        self.N = params.get('N', 1000)

        self.genome_params = RearrangedGenome.default_params

        for key in self.genome_params.keys():
            if key in params:
                self.genome_params[key] = params[key]

        self.proportion_hdel = params.get('proportion_hdel', 0.)
        self.proportion_hdel_stddev = params.get('proportion_hdel_stddev', 0.001)

        self.proportion_hlamp = params.get('proportion_hlamp', 0.)
        self.proportion_hlamp_stddev = params.get('proportion_hlamp_stddev', 0.001)

        self.ploidy = params.get('ploidy', 2.5)
        self.ploidy_stddev = params.get('ploidy_stddev', 0.1)

        self.proportion_loh = params.get('proportion_loh', 0.2)
        self.proportion_loh_stddev = params.get('proportion_loh_stddev', 0.02)

        self.num_swarm = 100


    def genome_fitness(self, genome, fitness_callback=None):
        """ Calculate fitness of a genome based on loh, hdel and hlamp proportions.

        Args:
            genome (RearrangedGenome): Genome to calculate fitness

        Kwargs:
            fitness_callback (callable): modify fitness callback 

        """

        hdel_log_p = scipy.stats.norm.logpdf(genome.proportion_hdel(), loc=self.proportion_hdel, scale=self.proportion_hdel_stddev)
        hlamp_log_p = scipy.stats.norm.logpdf(genome.proportion_hlamp(), loc=self.proportion_hlamp, scale=self.proportion_hlamp_stddev)
        ploidy_log_p = scipy.stats.norm.logpdf(genome.ploidy(), loc=self.ploidy, scale=self.ploidy_stddev)
        loh_log_p = scipy.stats.norm.logpdf(genome.proportion_loh(), loc=self.proportion_loh, scale=self.proportion_loh_stddev)

        fitness = hdel_log_p + hlamp_log_p + ploidy_log_p + loh_log_p

        if fitness_callback is not None:
            fitness = fitness_callback(genome, fitness)

        return fitness


    def resample_probs(self, genomes, fitness_callback=None):
        """ Calculate resampling probabilities.

        Args:
            genomes (list of RearrangedGenome): list of rearranged genomes to calculate prob from fitnesses

        Kwargs:
            fitness_callback (callable): modify fitness callback 

        """

        fitnesses = list()
        for genome in genomes:
            fitnesses.append(self.genome_fitness(genome, fitness_callback))

        fitnesses = np.array(fitnesses)

        prob = np.exp(fitnesses - scipy.misc.logsumexp(fitnesses))

        return prob


    def sample_wild_type(self):
        """ Sample a wild type genome.

        Returns:
            RearrangedGenome: a wild type genome with no rearrangements

        """

        wt_genome = RearrangedGenome(self.N)
        wt_genome.create(self.genome_params)

        return wt_genome


    def sample_rearrangement_history(self, genome_init, num_events, fitness_callback=None):
        """ Sample a rearrangement history specified by a random seed, and select based on fitness.

        Args:
            genome_init (RearrangedGenome): initial genome to which rearrangements will be applied
            num_events (int): number of rearrangement events

        Kwargs:
            fitness_callback (callable): modify fitness callback 

        Returns:
            list of RearrangedGenome: a list genome evolved through a series of rearrangements,
                                      sorted by the resample probablity of those genomes

        """

        swarm = [genome_init] * self.num_swarm

        for _ in range(num_events):
            new_swarm = list()
            for genome in swarm:
                genome = genome.copy()
                genome.rearrange(self.genome_params)
                new_swarm.append(genome)

            resample_p = self.resample_probs(new_swarm, fitness_callback=fitness_callback)
            resampled_swarm = np.random.choice(new_swarm, size=self.num_swarm, p=resample_p)

            swarm = list(resampled_swarm)

        prob = self.resample_probs(swarm)
        swarm = list(np.array(swarm)[np.argsort(prob)[::-1]])

        return swarm


def _collapse_allele_bp(allele_bp):
    ((n_1, ell_1), side_1), ((n_2, ell_2), side_2) = allele_bp
    return frozenset([(n_1, side_1), (n_2, side_2)])


def _sum_brk_cn_alleles(allele_brk_cn):
    total_brk_cn = {}
    for allele_bp, cn in allele_brk_cn.items():
        total_bp = _collapse_allele_bp(allele_bp)
        if total_bp not in total_brk_cn:
            total_brk_cn[total_bp] = cn
        else:
            total_brk_cn[total_bp] += cn
    return total_brk_cn


def _collapse_allele_bps(allele_bps):
    total_bps = set()
    for allele_bp in allele_bps:
        total_bps.add(_collapse_allele_bp(allele_bp))
    return total_bps


class GenomeCollection(object):
    """
    Collection of normal and tumour clone genomes.
    """
    def __init__(self, genomes):

        self.genomes = genomes

        self.cn = np.array([genome.segment_copy_number for genome in self.genomes])
        self.cn = self.cn.swapaxes(0, 1)

        self.adjacencies = set()
        for breakends in self.genomes[0].wt_adj:
            adj = [None, None]
            for breakend in breakends:
                adj[1-breakend[1]] = breakend[0][0]
            assert None not in adj
            self.adjacencies.add(tuple(adj))

        self.breakpoints = set()
        for genome in self.genomes[1:]:
            for brkend_1, brkend_2 in genome.breakpoints:
                brkend_1 = (brkend_1[0][0], brkend_1[1])
                brkend_2 = (brkend_2[0][0], brkend_2[1])
                self.breakpoints.add(frozenset([brkend_1, brkend_2]))

        self.breakpoint_copy_number = collections.defaultdict(lambda: np.zeros(self.M))
        for m in range(self.M):
            for breakpoint, brk_cn in self.genomes[m].breakpoint_copy_number.items():
                self.breakpoint_copy_number[breakpoint][m] = brk_cn
        self.breakpoint_copy_number = dict(self.breakpoint_copy_number)

        self.balanced_breakpoints = set()
        for breakpoint, brk_cn in self.breakpoint_copy_number.items():
            brk_cn_sum = 0
            for (n, ell), side_1 in breakpoint:
                if side_1 == 1:
                    n_2 = (n + 1) % self.N
                else:
                    n_2 = (n - 1) % self.N
                brk_cn_sum += abs((self.cn[n,:,ell] - self.cn[n_2,:,ell]).sum())
            if brk_cn_sum == 0:
                self.balanced_breakpoints.add(breakpoint)

    @property
    def N(self):
        return self.genomes[0].N

    @property
    def M(self):
        return len(self.genomes)

    @property
    def l(self):
        return self.genomes[0].l

    @property
    def segment_chromosome_id(self):
        return self.genomes[0].segment_chromosome_id

    @property
    def segment_start(self):
        return self.genomes[0].segment_start

    @property
    def segment_end(self):
        return self.genomes[0].segment_end

    def length_divergent(self):
        return self.genomes[1].length_divergent(self.genomes[2])

    def length_loh(self):
        return [g.length_loh() for g in self.genomes]

    def length_hdel(self):
        return [g.length_hdel() for g in self.genomes]

    def length_hlamp(self, hlamp_min=6):
        return [g.length_hlamp() for g in self.genomes]

    def collapsed_breakpoint_copy_number(self):
        return _sum_brk_cn_alleles(self.breakpoint_copy_number)

    def collapsed_minimal_breakpoint_copy_number(self):
        minimal_breakpoint_copy_number = remixt.simulations.balanced.minimize_breakpoint_copies(
            self.adjacencies, self.breakpoint_copy_number)
        return _sum_brk_cn_alleles(minimal_breakpoint_copy_number)

    def collapsed_balanced_breakpoints(self):
        return _collapse_allele_bps(self.balanced_breakpoints)


class GenomeCollectionSampler(object):
    """
    Samples a collection of two genomes with rearrangement histories related by a chain phylogeny.
    """

    def __init__(self, rearrangement_history_sampler, params):

        self.rh_sampler = rearrangement_history_sampler

        self.num_ancestral_events = params.get('num_ancestral_events', 25)
        self.num_descendent_events = params.get('num_descendent_events', 10)

        self.M = params['M']

        self.ploidy = params.get('ploidy', 2.5)
        self.ploidy_max_error = params.get('ploidy_max_error', 0.2)

        self.proportion_loh = params.get('proportion_loh', 0.2)
        self.proportion_loh_max_error = params.get('proportion_loh_max_error', 0.02)

        self.proportion_subclonal = params.get('proportion_subclonal', 0.3)
        self.proportion_subclonal_max_error = params.get('proportion_subclonal_max_error', 0.02)
        self.proportion_subclonal_stddev = params.get('proportion_subclonal_stddev', 0.02)

    def sample_genome_collection(self):
        """ Sample a collection of genomes.

        Returns:
            GenomeCollection: randomly generated collection of genomes

        """

        wt_genome = self.rh_sampler.sample_wild_type()

        genomes = [wt_genome]

        success = False
        ancestral_genome = None
        for anc_iter in range(100):
            ancestral_genomes = self.rh_sampler.sample_rearrangement_history(wt_genome, self.num_ancestral_events)
            ancestral_genomes = np.array(ancestral_genomes)

            ploidys = np.array([genome.ploidy() for genome in ancestral_genomes])
            ploidy_errors = np.absolute(ploidys - self.ploidy)
            ancestral_genomes = ancestral_genomes[ploidy_errors < self.ploidy_max_error]

            if len(ancestral_genomes) == 0:
                print ('failed ploidy')
                continue

            loh_proportions = np.array([genome.proportion_loh() for genome in ancestral_genomes])
            loh_errors = np.absolute(loh_proportions - self.proportion_loh)
            ancestral_genomes = ancestral_genomes[loh_errors < self.proportion_loh_max_error]

            if len(ancestral_genomes) == 0:
                print ('failed loh')
                continue

            ancestral_genome = ancestral_genomes[0]
            genomes.append(ancestral_genomes[0])
            success = True
            break

        if not success:
            raise ValueError('unable to simulate ancestral genome')

        def subclone_fitness(genome, fitness):
            divergent_log_p = scipy.stats.norm.logpdf(
                genome.proportion_divergent(ancestral_genome), loc=self.proportion_subclonal, scale=self.proportion_subclonal_stddev)
            return fitness + divergent_log_p

        for m in range(self.M - 2, self.M):
            success = False
            for desc_iter in range(100):
                descendent_genomes = self.rh_sampler.sample_rearrangement_history(
                    ancestral_genome, self.num_descendent_events, fitness_callback=subclone_fitness)
                descendent_genomes = np.array(descendent_genomes)

                subclonal_proportions = np.array([genome.proportion_divergent(ancestral_genome) for genome in descendent_genomes])
                subclonal_errors = np.absolute(subclonal_proportions - self.proportion_subclonal)
                descendent_genomes = descendent_genomes[subclonal_errors < self.proportion_subclonal_max_error]

                if len(descendent_genomes) == 0:
                    print ('failed subclonal')
                    continue

                genomes.append(descendent_genomes[0])

                success = True
                break

            if not success:
                raise ValueError('unable to simulate descendant genome')

        return GenomeCollection(genomes)


class GenomeMixture(object):
    """
    Mixture of normal and tumour clone genomes.
    """
    def __init__(self, genome_collection, frac, detected_breakpoints):

        self.genome_collection = genome_collection
        self.frac = frac
        self.detected_breakpoints = detected_breakpoints

        # Table of breakpoint information including chromosome position
        # info and prediction ids
        breakpoint_segment_data = list()
        for prediction_id, breakpoint in self.detected_breakpoints.items():
            breakpoint_info = {'prediction_id': prediction_id}
            for breakend_idx, breakend in enumerate(breakpoint):
                n, side = breakend
                if side == 0:
                    strand = '-'
                    position = self.segment_start[n]
                elif side == 1:
                    strand = '+'
                    position = self.segment_end[n]
                else:
                    raise Exception('unexpected side value')
                breakpoint_info['n_{}'.format(breakend_idx + 1)] = n
                breakpoint_info['side_{}'.format(breakend_idx + 1)] = side
                breakpoint_info['chromosome_{}'.format(breakend_idx + 1)] = self.segment_chromosome_id[n]
                breakpoint_info['position_{}'.format(breakend_idx + 1)] = position
                breakpoint_info['strand_{}'.format(breakend_idx + 1)] = strand
            breakpoint_segment_data.append(breakpoint_info)
        self.breakpoint_segment_data = pd.DataFrame(breakpoint_segment_data)

    @property
    def N(self):
        return self.genome_collection.N

    @property
    def M(self):
        return self.genome_collection.M

    @property
    def l(self):
        return self.genome_collection.l

    @property
    def segment_chromosome_id(self):
        return self.genome_collection.segment_chromosome_id

    @property
    def segment_start(self):
        return self.genome_collection.segment_start

    @property
    def segment_end(self):
        return self.genome_collection.segment_end

    @property
    def cn(self):
        return self.genome_collection.cn

    @property
    def adjacencies(self):
        return self.genome_collection.adjacencies

    @property
    def breakpoints(self):
        return self.genome_collection.breakpoints


def sample_random_breakpoints(N, num_breakpoints, adjacencies, excluded_breakpoints=None):
    breakpoints = set()
    while len(breakpoints) < num_breakpoints:

        n_1 = np.random.randint(N)
        n_2 = np.random.randint(N)

        side_1 = np.random.randint(2)
        side_2 = np.random.randint(2)

        # Do not add wild type adjacencies
        if (n_1, n_2) in adjacencies and side_1 == 1 and side_2 == 0:
            continue
        if (n_2, n_1) in adjacencies and side_2 == 1 and side_1 == 0:
            continue

        # TODO: fold back inversions
        if (n_1, side_1) == (n_2, side_2):
            continue

        breakpoint = frozenset([(n_1, side_1), (n_2, side_2)])

        # Do not add excluded breakpoints
        if excluded_breakpoints is not None and breakpoint in excluded_breakpoints:
            continue

        breakpoints.add(breakpoint)

    return breakpoints


class GenomeMixtureSampler(object):
    """ Sampler for genome mixtures.
    """

    def __init__(self, params):

        self.frac_normal = params.get('frac_normal', 0.4)
        self.frac_clone_concentration = params.get('frac_clone_concentration', 1.)
        self.frac_clone_1 = params.get('frac_clone_1', None)
        self.num_false_breakpoints = params.get('num_false_breakpoints', 50)
        self.proportion_breakpoints_detected = params.get('proportion_breakpoints_detected', 0.9)

    def sample_genome_mixture(self, genome_collection):
        """ Sample a genome mixture.

        Args:
            genome_collection (GenomeCollection): collection of genomes to mix

        Returns:
            GenomeMixture: mixed genomes

        """

        M = genome_collection.M

        frac = np.zeros((M,))
        
        frac[0] = self.frac_normal

        if self.frac_clone_1 is None:
            frac[1:] = np.random.dirichlet([self.frac_clone_concentration] * (M-1)) * (1 - self.frac_normal)
        elif M == 3:
            frac[1:] = np.array([self.frac_clone_1, 1. - self.frac_normal - self.frac_clone_1])
        elif M == 4:
            frac_clones_2_3 = 1. - self.frac_normal - self.frac_clone_1
            frac_clones_2_3 = np.random.dirichlet([self.frac_clone_concentration] * (M-2)) * frac_clones_2_3
            frac[1:] = np.array([self.frac_clone_1] + list(frac_clones_2_3))
        else:
            raise Exception('Case not handled')

        assert abs(1. - np.sum(frac)) < 1e-8

        num_breakpoints_detected = int(self.proportion_breakpoints_detected * len(genome_collection.breakpoints))
        detected_breakpoints = list(genome_collection.breakpoints)
        np.random.shuffle(detected_breakpoints)
        detected_breakpoints = detected_breakpoints[:num_breakpoints_detected]

        false_breakpoints = sample_random_breakpoints(
            genome_collection.N,
            self.num_false_breakpoints,
            genome_collection.adjacencies,
            excluded_breakpoints=genome_collection.breakpoints,
        )

        detected_breakpoints.extend(false_breakpoints)

        # Create a dictionary of detected breakpoints, keyed by a unique id
        detected_breakpoints = dict(enumerate(detected_breakpoints))

        return GenomeMixture(genome_collection, frac, detected_breakpoints)


class Experiment(object):
    """ Sequencing experiment read counts.
    """

    def __init__(self, genome_mixture, h, phi, x, h_pred, **kwargs):

        self.genome_mixture = genome_mixture
        self.h = h
        self.phi = phi
        self.x = x
        self.h_pred = h_pred

        self.__dict__.update(kwargs)

    @property
    def N(self):
        return self.genome_mixture.N

    @property
    def M(self):
        return self.genome_mixture.M

    @property
    def l(self):
        return self.genome_mixture.l

    @property
    def segment_chromosome_id(self):
        return self.genome_mixture.segment_chromosome_id

    @property
    def segment_start(self):
        return self.genome_mixture.segment_start

    @property
    def segment_end(self):
        return self.genome_mixture.segment_end

    @property
    def cn(self):
        return self.genome_mixture.cn

    @property
    def adjacencies(self):
        return self.genome_mixture.adjacencies

    @property
    def chains(self):
        chain_start = [0]
        chain_end = [self.N]
        for idx in range(self.N - 1):
            if (idx, idx+1) not in self.adjacencies:
                chain_end.append(idx+1)  # Half-open interval indexing [start, end)
                chain_start.append(idx+1)
        return zip(sorted(chain_start), sorted(chain_end))

    @property
    def breakpoints(self):
        return self.genome_mixture.detected_breakpoints

    @property
    def breakpoint_segment_data(self):
        return self.genome_mixture.breakpoint_segment_data


def _sample_negbin(mu, r):
    mu += 1e-16
    inv_p = r / (r + mu)
    x = np.array([np.random.negative_binomial(r, a) for a in inv_p]).reshape(mu.shape)
    return x


def _sample_negbin_mix(mu, r_0, r_1, mix):    
    x_0 = _sample_negbin(mu, r_0)
    x_1 = _sample_negbin(mu, r_1)
    is_0 = np.random.random(size=x_0.shape) > mix
    x = np.where(is_0, x_0, x_1)
    return x, is_0


def _sample_betabin(n, p, M):
    p_binom = np.random.beta(M * p, M * (1 - p))
    x = np.random.binomial(n, p_binom)
    return x


def _sample_betabin_mix(n, p, M_0, M_1, mix):
    x_0 = _sample_betabin(n, p, M_0)
    x_1 = _sample_betabin(n, p, M_1)
    is_0 = np.random.random(size=x_0.shape) > mix
    x = np.where(is_0, x_0, x_1)
    return x, is_0


class ExperimentSampler(object):
    """ Sampler for sequencing experiments.
    """

    def __init__(self, params):

        self.h_total = params.get('h_total', 0.1)

        self.phi_min = params.get('phi_min', 0.05)
        self.phi_max = params.get('phi_max', 0.2)

        self.emission_model = params.get('emission_model', 'negbin_betabin')

        if self.emission_model not in ('poisson', 'negbin', 'normal', 'full', 'negbin_betabin'):
            raise ValueError('emission_model must be one of "poisson", "negbin", "normal", "full"')

        self.frac_beta_noise_stddev = params.get('frac_beta_noise_stddev', None)

        self.params = params.copy()


    def sample_experiment(self, genome_mixture):
        """ Sample a sequencing experiment.

        Args:
            genome_mixture (GenomeMixture): mixture to sample read counts representing a sequencing experiment

        Returns:
            Experiment: sequencing experiment read counts

        """

        N = genome_mixture.N
        l = genome_mixture.l
        cn = genome_mixture.cn

        h = genome_mixture.frac * self.h_total

        phi = np.random.uniform(low=self.phi_min, high=self.phi_max, size=N)

        mu = remixt.likelihood.expected_read_count(l, cn, h, phi)

        extra_params = dict()

        if self.emission_model == 'poisson':
            mu_poisson = mu + 1e-16

            x = np.array([np.random.poisson(a) for a in mu_poisson]).reshape(mu_poisson.shape)

        elif self.emission_model == 'negbin':
            mu_negbin = mu + 1e-16

            nb_inv_p = self.negbin_r / (self.negbin_r + mu_negbin)

            x = np.array([np.random.negative_binomial(self.negbin_r, a) for a in nb_inv_p]).reshape(mu_negbin.shape)

            extra_params['negbin_r'] = self.negbin_r

        elif self.emission_model == 'negbin_betabin':
            x = np.zeros(mu.shape)

            negbin_r_0 = self.params.get('negbin_r_0', 1000.)
            negbin_r_1 = self.params.get('negbin_r_1', 10.)
            negbin_mix = self.params.get('negbin_mix', 0.01)

            betabin_M_0 = self.params.get('betabin_M_0', 2000.)
            betabin_M_1 = self.params.get('betabin_M_1', 10.)
            betabin_mix = self.params.get('betabin_mix', 0.01)

            x_total, x_total_is_0 = _sample_negbin_mix(mu[:, 2] + 1e-16, negbin_r_0, negbin_r_1, negbin_mix)

            allele_total = (phi * x_total).astype(int)
            p_true = mu[:, 0] / (mu[:, 0:2].sum(axis=1) + 1e-16)
            x_allele_1, x_allele_1_is_0 = _sample_betabin_mix(allele_total, p_true, betabin_M_0, betabin_M_1, betabin_mix)
            x_allele_2 = allele_total - x_allele_1

            x[:, 2] = x_total
            x[:, 0] = x_allele_1
            x[:, 1] = x_allele_2

            extra_params['is_outlier_total'] = ~x_total_is_0
            extra_params['is_outlier_allele'] = ~x_allele_1_is_0

        elif self.emission_model == 'normal':
            x = np.zeros(mu.shape)

            mu_total = mu[:, 2]
            a, b = 0.14514556880927346, 1.3745893696636038
            variance = a * mu_total**b
            variance[variance == 0] = 50.

            x[:, 2] = np.random.normal(loc=mu_total, scale=variance**0.5)
            
            mu_allele = mu[:, 0:2]
            a, b = 0.040819090849598873, 1.4981089638117262
            variance = a * mu_allele**b
            variance[variance == 0] = 50.

            x[:, 0:2] = np.random.normal(loc=mu_allele, scale=variance**0.5)

            x[x < 0] = 0
            x = x.round().astype(int)

            if self.noise_prior is not None:
                noise_range_total = mu[:, 2].max()
                noise_range_total *= 1.25

                is_outlier_total = np.random.choice(
                    [True, False], size=mu[:, 2].shape,
                    p=[self.noise_prior, 1. - self.noise_prior])

                x[is_outlier_total, 2] = (np.random.randint(noise_range_total, size=mu[:, 2].shape))[is_outlier_total]

                is_outlier_allele = np.random.choice(
                    [True, False], size=mu[:, 0].shape,
                    p=[self.noise_prior, 1. - self.noise_prior])

                outlier_allele_ratio = np.random.beta(2, 2, size=mu[:, 0].shape)

                x[is_outlier_allele, 0] = (outlier_allele_ratio * x[:, 0:2].sum(axis=1))[is_outlier_allele]
                x[is_outlier_allele, 1] = ((1. - outlier_allele_ratio) * x[:, 0:2].sum(axis=1))[is_outlier_allele]
                
        elif self.emission_model == 'full':
            x = np.zeros(mu.shape)

            mu_total = mu[:, 2]
            a, b = 0.14514556880927346, 1.3745893696636038
            variance = a * mu_total**b
            variance[variance == 0] = 50.

            x[:, 2] = np.random.normal(loc=mu_total, scale=variance**0.5)

            M = 1200
            loh_p = 0.01
            noise_prior = 0.03

            mu_total = mu[:, 2]
            p_true = mu[:, 0] / mu[:, 0:2].sum(axis=1)
            
            p_true[p_true == 0] = loh_p
            p_true[p_true == 1] = loh_p

            p_dispersion = np.random.beta(M * p_true, M * (1 - p_true))
            p_noise = np.random.random(size=p_true.shape)
            p_is_noise = np.random.random(size=p_true.shape) <= noise_prior
            p_binomial = np.where(p_is_noise, p_noise, p_dispersion)
            
            allele_reads = (x[:, 2] * phi).astype(int)
            
            x[:, 0] = np.random.binomial(allele_reads, p_binomial)
            x[:, 1] = allele_reads.astype(float) - x[:, 0]

        # Reorder x as segment * major/minor/total and record
        # whether the major allele refers to the first (a) allele
        # TODO: perhaps easiers to just do minor/major/total
        major_is_allele_a = x[:, 0] > x[:, 1]
        for n in range(N):
            if not major_is_allele_a[n]:
                x[n, 0], x[n, 1] = x[n, 1], x[n, 0]

        extra_params['segment_major_is_allele_a'] = major_is_allele_a * 1

        def add_beta_noise(mu, var):
            if np.any(var >= mu * (1. - mu)):
                raise ValueError('var >= mu * (1. - mu)')
            nu = mu * (1. - mu) / var - 1.
            a = mu * nu
            b = (1 - mu) * nu
            return np.array([np.random.beta(_a, _b) for _a, _b in zip(a, b)])

        if self.frac_beta_noise_stddev is not None:
            frac = add_beta_noise(genome_mixture.frac, self.frac_beta_noise_stddev**2.)
        else:
            frac = genome_mixture.frac

        h_pred = frac * self.h_total

        return Experiment(genome_mixture, h, phi, x, h_pred, **extra_params)


