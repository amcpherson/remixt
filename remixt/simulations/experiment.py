import sys
import copy
import math
import collections
import numpy as np
import scipy
import scipy.stats

import remixt.utils
import remixt.likelihood


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
        'seg_length_concentration':1.,
        'num_chromosomes':20,
        'chrom_length_concentration':5.,
        'chromosome_lengths':None,
        'event_type':['dcj', 'dup', 'del'],
        'event_prob':[0.2, 0.4, 0.4],
        'del_prop_len':0.5,
        'dup_prop_len':0.5,
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


    def create(self, params):
        """ Create a new non-rearranged genome.

        Args:
            params (dict): parameters for random chromosome creation

        Sets the seed and updates the init seed and params.

        """
        seed = np.random.randint(MAX_SEED - 1)

        np.random.seed(seed)

        self.random_chromosomes(params)

        self.init_params = copy.deepcopy(params)
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

            chromosome_ids = [str(a) for a in xrange(num_chroms)]
            chromosome_lengths = np.random.dirichlet([chrom_length_concentration] * num_chroms) * genome_length
            chromosome_lengths.sort()
            chromosome_lengths = chromosome_lengths[::-1]

        chrom_pvals = chromosome_lengths.astype(float) / float(chromosome_lengths.sum())
        chrom_num_segments = np.random.multinomial(self.N - len(chromosome_lengths), pvals=chrom_pvals)
        chrom_num_segments += 1

        seg_length_concentration = params['seg_length_concentration']

        self.l = np.array([])

        self.segment_chromosome_id = np.array([], dtype=str)
        self.segment_start = np.array([], dtype=int)
        self.segment_end = np.array([], dtype=int)

        for chrom_id, chrom_length, num_segments in zip(chromosome_ids, chromosome_lengths, chrom_num_segments):

            lengths = np.random.dirichlet([seg_length_concentration] * num_segments) * chrom_length
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

                chrom_segs = xrange(segment_idx, segment_idx+num_seg)
                chrom_alleles = [allele]*num_seg
                chrom_orient = [1]*num_seg

                self.chromosomes.append(list(zip(zip(chrom_segs, chrom_alleles), chrom_orient)))

            segment_idx += num_seg

        self.wt_adj = set()
        self.wt_adj = set(self.breakpoints)


    def generate_cuts(self):
        """ Generate a list of possible cuts.
        
        Cuts are triples of chromosome index, segment index where
        the segment index is the second in an adjacent pair

        """
        for chromosome_idx, chromosome in enumerate(self.chromosomes):
            for segment_idx in xrange(len(chromosome)):
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
        return [self.reverse_segment(a) for a in reversed(chromosome)]


    def rearrange(self, params):
        """ Apply random rearrangement event.

        Args:
            params (dict): dictionary of modification params

        Sets the seed and appends the seed and params to the event lists.

        """
        seed = np.random.randint(MAX_SEED - 1)

        np.random.seed(seed)

        self.random_event(params)

        self.event_params.append(copy.deepcopy(params))
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
    
    
    def proportion_loh(self):
        """ Proportion of the genome with LOH.
        """
        cn = self.segment_copy_number
        return np.sum(cn.min(axis=1) == 0) / float(cn.shape[0])
    

    def proportion_hdel(self):
        """ Proportion of the genome homozygously deleted.
        """
        cn = self.segment_copy_number
        return np.sum(cn.max(axis=1) == 0) / float(cn.shape[0])

    
    def proportion_hlamp(self, hlamp_min=6):
        """ Proportion of the genome with high level amplification.
        """
        cn = self.segment_copy_number
        return np.sum(cn.max(axis=1) >= hlamp_min) / float(cn.shape[0])


    def proportion_minor_state(self, cn_max=6):
        """ Proportion of the genome with minor in each cn state.
        """
        minor = self.segment_copy_number.min(axis=1)
        minor[minor > cn_max] = cn_max
        return np.bincount(minor.flatten().astype(int), minlength=cn_max+1)


    def proportion_major_state(self, cn_max=6):
        """ Proportion of the genome with major in each cn state.
        """
        major = self.segment_copy_number.max(axis=1)
        major[major > cn_max] = cn_max
        return np.bincount(major.flatten().astype(int), minlength=cn_max+1)


    def proportion_divergent(self, other, cn_max=6):
        """ Proportion of the genome with high level amplification.
        """
        cn = self.segment_copy_number
        other_cn = other.segment_copy_number
        divergence = np.absolute(cn - other_cn)
        divergence[divergence > cn_max] = cn_max
        return np.bincount(divergence.flatten().astype(int), minlength=cn_max+1)


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

    hgs_major_cn_proportions = [
        0.0009,
        0.3749,
        0.4091,
        0.1246,
        0.0291,
        0.0161,
        0.0453,
    ]

    hgs_minor_cn_proportions = [
        0.2587,
        0.5460,
        0.1586,
        0.0097,
        0.0054,
        0.0029,
        0.0187,
    ]

    def __init__(self, params):

        self.N = params.get('N', 1000)

        self.genome_params = RearrangedGenome.default_params

        for key in self.genome_params.keys():
            if key in params:
                self.genome_params[key] = params[key]

        self.major_cn_proportions = params.get('major_cn_proportions', self.hgs_major_cn_proportions)
        self.minor_cn_proportions = params.get('minor_cn_proportions', self.hgs_minor_cn_proportions)
        
        self.num_swarm = 100


    def genome_fitness(self, genome, fitness_callback=None):
        """ Calculate fitness of a genome based on loh, hdel and hlamp proportions.

        Args:
            genome (RearrangedGenome): Genome to calculate fitness

        Kwargs:
            fitness_callback (callable): modify fitness callback 

        """
        
        log_minor_p = log_multinomial_likelihood(genome.proportion_minor_state(), self.minor_cn_proportions)
        log_major_p = log_multinomial_likelihood(genome.proportion_major_state(), self.major_cn_proportions)

        fitness = log_minor_p + log_major_p
        
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
            RearrangedGenome: a genome evolved through a series of rearrangements

        """

        swarm = [genome_init] * self.num_swarm

        for _ in xrange(num_events):

            print '.',

            new_swarm = list()
            for genome in swarm:
                genome = copy.deepcopy(genome)
                genome.rearrange(self.genome_params)
                new_swarm.append(genome)

            resample_p = self.resample_probs(new_swarm, fitness_callback=fitness_callback)
            resampled_swarm = np.random.choice(new_swarm, size=self.num_swarm, p=resample_p)

            swarm = list(resampled_swarm)

        print

        prob = self.resample_probs(swarm)
        result = np.random.choice(swarm, size=1, p=prob)[0]

        return result


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
        for m in xrange(self.M):
            for breakpoint, brk_cn in self.genomes[m].breakpoint_copy_number.iteritems():
                self.breakpoint_copy_number[breakpoint][m] = brk_cn
        self.breakpoint_copy_number = dict(self.breakpoint_copy_number)

        self.balanced_breakpoints = set()
        for breakpoint, brk_cn in self.breakpoint_copy_number.iteritems():
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

    def proportion_divergent(self):
        diver = self.genomes[1].proportion_divergent(self.genomes[2])
        return float(sum(diver[1:])) / float(sum(diver))

    def proportion_loh(self):
        return [g.proportion_loh() for g in self.genomes]

    def proportion_hdel(self):
        return [g.proportion_hdel() for g in self.genomes]

    def proportion_hlamp(self, hlamp_min=6):
        return [g.proportion_hlamp() for g in self.genomes]


class GenomeCollectionSampler(object):
    """
    Samples a collection of two genomes with rearrangement histories related by a chain phylogeny.
    """

    def __init__(self, rearrangement_history_sampler, params):

        self.rh_sampler = rearrangement_history_sampler

        self.num_ancestral_events = params.get('num_ancestral_events', 25)
        self.num_descendent_events = params.get('num_descendent_events', 10)

        self.proportion_subclonal = params.get('proportion_subclonal', 0.3)

        self.divergence_proportions = [
                                       1.0 - self.proportion_subclonal,
                                       self.proportion_subclonal,
                                       1e-9,
                                       1e-9,
                                       1e-9,
                                       1e-9,
                                       1e-9,
                                      ]
        self.divergence_proportions = np.array(self.divergence_proportions)
        self.divergence_proportions /= self.divergence_proportions.sum()

    def sample_genome_collection(self):
        """ Sample a collection of genomes.

        Returns:
            GenomeCollection: randomly generated collection of genomes

        """

        wt_genome = self.rh_sampler.sample_wild_type()

        ancestral_genome = self.rh_sampler.sample_rearrangement_history(wt_genome, self.num_ancestral_events)

        def descendent_genome_fitness(genome, fitness):

            log_diverg_p = log_multinomial_likelihood(ancestral_genome.proportion_divergent(genome), self.divergence_proportions)

            return log_diverg_p + fitness
        
        descendent_genome = self.rh_sampler.sample_rearrangement_history(ancestral_genome, self.num_descendent_events, fitness_callback=descendent_genome_fitness)

        return GenomeCollection([wt_genome, ancestral_genome, descendent_genome])


class GenomeMixture(object):
    """
    Mixture of normal and tumour clone genomes.
    """
    def __init__(self, genome_collection, frac):

        self.genome_collection = genome_collection
        self.frac = frac

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


class GenomeMixtureSampler(object):
    """ Sampler for genome mixtures.
    """

    def __init__(self, params):

        self.frac_normal = params.get('frac_normal', 0.4)
        self.frac_clone_concentration = params.get('frac_clone_concentration', 1.)

        self.frac_clone = params.get('frac_clone', None)

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

        if self.frac_clone is None:
            frac[1:] = np.random.dirichlet([self.frac_clone_concentration] * (M-1)) * (1 - self.frac_normal)
        else:
            frac[1:] = np.array(self.frac_clone)

        frac[1:] = np.sort(frac[1:])[::-1]

        return GenomeMixture(genome_collection, frac)


class Experiment(object):
    """ Sequencing experiment read counts.
    """

    def __init__(self, genome_mixture, h, phi, x, breakpoints, h_pred, **kwargs):

        self.genome_mixture = genome_mixture
        self.h = h
        self.phi = phi
        self.x = x
        self.breakpoints = breakpoints
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


class ExperimentSampler(object):
    """ Sampler for sequencing experiments.
    """

    def __init__(self, params):

        self.h_total = params.get('h_total', 0.1)

        self.phi_min = params.get('phi_min', 0.05)
        self.phi_max = params.get('phi_max', 0.2)

        self.emission_model = params.get('emission_model', 'negbin')

        if self.emission_model not in ('poisson', 'negbin'):
            raise ValueError('emission_model must be one of "poisson", "negbin"')

        self.negbin_r = np.array([
            params.get('negbin_r_allele', 200.0),
            params.get('negbin_r_allele', 200.0),
            params.get('negbin_r_total', 200.0)])

        self.num_false_breakpoints = params.get('num_false_breakpoints', 50)

        self.frac_beta_noise_stddev = params.get('frac_beta_noise_stddev', None)


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

        model = remixt.likelihood.ReadCountLikelihood()
        model.h = h
        model.phi = phi

        mu = model.expected_read_count(l, cn)

        extra_params = dict()

        if self.emission_model == 'poisson':

            x = np.array([np.random.poisson(a) for a in mu]).reshape(mu.shape)

        elif self.emission_model == 'negbin':

            nb_inv_p = self.negbin_r / (self.negbin_r + mu)

            x = np.array([np.random.negative_binomial(self.negbin_r, a) for a in nb_inv_p]).reshape(mu.shape)

            extra_params['negbin_r'] = self.negbin_r

        # Reorder x as segment * major/minor/total
        x[:,0:2].sort(axis=1)
        x[:,0:2] = x[:,2:0:-1]

        breakpoints = genome_mixture.breakpoints.copy()

        num_breakpoints = len(breakpoints) + self.num_false_breakpoints

        while len(breakpoints) < num_breakpoints:

            n_1 = np.random.randint(N)
            n_2 = np.random.randint(N)

            side_1 = np.random.randint(2)
            side_2 = np.random.randint(2)

            # Do not add wild type adjacencies
            if (n_1, n_2) in genome_mixture.adjacencies and side_1 == 1 and side_2 == 0:
                continue
            if (n_2, n_1) in genome_mixture.adjacencies and side_2 == 1 and side_1 == 0:
                continue

            # TODO: fold back inversions
            if (n_1, side_1) == (n_2, side_2):
                continue

            breakpoints.add(frozenset([(n_1, side_1), (n_2, side_2)]))

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

        return Experiment(genome_mixture, h, phi, x, breakpoints, h_pred, **extra_params)


def compare_segment_copy_number(true_cn, pred_cn):

    true_cn = true_cn[:,1:,:].copy()
    pred_cn = pred_cn[:,1:,:].copy()

    stats = dict()

    true_is_clonal = (np.array([true_cn[:,0,:]] * true_cn.shape[1]).swapaxes(0, 1) == true_cn).all(axis=(1,2))
    pred_is_clonal = (np.array([pred_cn[:,0,:]] * pred_cn.shape[1]).swapaxes(0, 1) == pred_cn).all(axis=(1,2))
    stats['num_seg_is_clonal_incorrect'] = (true_is_clonal != pred_is_clonal).sum()

    print true_is_clonal
    print pred_is_clonal

    return stats


def compare_brk_copy_number(true_brk_cn, pred_brk_cn):

    stats = dict()
    stats['num_brk_dominant_correct'] = 0
    stats['num_brk_subclonal_correct'] = 0

    for bp, true_cn in true_brk_cn.iteritems():

        try:
            pred_cn = pred_brk_cn[bp]
        except KeyError:
            continue
            
        true_dominant = False
        true_subclonal = False
        
        if np.all(true_cn[1:] > 0):
            true_dominant = True
        elif np.any(true_cn[1:] > 0):
            true_subclonal = True

        pred_dominant = False
        pred_subclonal = False

        if np.all(pred_cn[1:] > 0):
            pred_dominant = True
        elif np.any(pred_cn[1:] > 0):
            pred_subclonal = True

        if true_dominant and pred_dominant:
            stats['num_brk_dominant_correct'] += 1

        if true_subclonal and pred_subclonal:
            stats['num_brk_subclonal_correct'] += 1

    return stats


