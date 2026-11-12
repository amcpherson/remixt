import itertools
import collections
import numpy as np
import pandas as pd
import scipy.misc
import pickle
import contextlib
import datetime
import statsmodels.tools.numdiff

import remixt.bpmodel


def _get_brkend_seg_orient(breakend):
    n, side = breakend
    if side == 1:
        n_left = n
        orient = +1
    elif side == 0:
        n_left = n - 1
        orient = -1
    return n_left, orient


def _gettime():
    return datetime.datetime.now().time().isoformat()


class OptimizationError(ValueError):
    """ Model fitting failed to optimize a parameter.
    """


class BreakpointModel(object):

    def __init__(self, x, l, adjacencies, breakpoints, **kwargs):
        """ Create a copy number model.

        Args:
            h_init (numpy.array): per clone haploid read depth
            x (numpy.array): observed minor, major, total reads
            l (numpy.array): observed segment lengths
            adjacencies (list of tuple): pairs of adjacent segments
            breakpoints (dict of frozenset of tuples): breakpoints as segment extremity pairs

        KwArgs:
            max_copy_number (int): maximum copy number of HMM state space
            normal_contamination (bool): whether the sample is contaminated by normal
            divergence_weight (float): clone segment divergence prior parameter
            min_segment_length (float): minimum size of segments for segment likelihood mask
            min_proportion_genotyped (float): minimum proportion genotyped reads for segment likelihood mask
            max_depth (float): maximum depth for segment likelihood mask
            transition_log_prob (float): penalty on transitions, per copy number change
            disable_breakpoints (bool): disable integrated breakpoint copy number inference
            normal_copies (numpy.array): germline copy number
            transition_saturation (float): if set, transition penalties saturate at this copy number change
            amp_windows (bool): after amp_window_start_iter iterations, remove the depth mask and give high
                copy number segments windows of states around their read depth estimate of copy number
            amp_window_start_iter (int): number of masked iterations before switching to windows
            amp_window_major_radius (int): window radius for the major allele
            amp_window_minor_radius (int): window radius for the minor allele

        """
        
        # Observed data should be ordered as major, minor, total
        assert np.all(x[:, 1] <= x[:, 0])

        self.N = x.shape[0]

        self.breakpoint_ids, self.breakpoints = zip(*breakpoints.items())
        
        self.max_copy_number = kwargs.get('max_copy_number', 6)
        self.max_copy_number_diff = kwargs.get('max_copy_number_diff', 1)
        self.normal_contamination = kwargs.get('normal_contamination', True)
        self.is_female = kwargs.get('is_female', True)
        self.divergence_weight = kwargs.get('divergence_weight', 1e6)
        self.min_segment_length = kwargs.get('min_segment_length', 10000)
        self.min_proportion_genotyped = kwargs.get('min_proportion_genotyped', 0.01)
        self.max_depth = kwargs.get('max_depth')
        self.transition_log_prob = kwargs.get('transition_log_prob', 10.)
        self.transition_model = kwargs.get('transition_model', 0)
        self.disable_breakpoints = kwargs.get('disable_breakpoints', False)
        self.breakpoint_init = kwargs.get('breakpoint_init', None)
        self.normal_copies = kwargs.get('normal_copies', np.array([[1, 1]] * self.N))
        self.do_h_update = kwargs.get('do_h_update', True)
        self.transition_saturation = kwargs.get('transition_saturation', None)
        self.amp_windows = kwargs.get('amp_windows', False)
        self.amp_window_start_iter = kwargs.get('amp_window_start_iter', 2)
        self.amp_window_major_radius = kwargs.get('amp_window_major_radius', 3)
        self.amp_window_minor_radius = kwargs.get('amp_window_minor_radius', 2)
        self.amp_window_recenter = kwargs.get('amp_window_recenter', 'depth')
        self.amp_window_seed_short = kwargs.get('amp_window_seed_short', False)
        self.amp_window_sweeps = kwargs.get('amp_window_sweeps', 3)
        # Lower bound on the overdispersion of the total read count outlier component.
        # A very small value makes the outlier component so broad that the depth of
        # long, deeply covered segments barely constrains their copy number.
        self.negbin_r_1_min = kwargs.get('negbin_r_1_min', 1.)

        if self.transition_saturation is not None:
            self.transition_model = 2

        if self.max_depth is None:
            raise ValueError('must specify max depth')

        if not self.normal_contamination:
            self.normal_copies = self.normal_copies * 0

        # The factor graph model for breakpoint copy number allows only a single breakend
        # interposed between each pair of adjacent segments.  Where multiple breakends are
        # involved, additional zero lenght dummy segments must be added between those
        # breakends
        breakpoint_segment = collections.defaultdict(set)
        for bp_idx, breakpoint in enumerate(self.breakpoints):
            for be_idx, breakend in enumerate(breakpoint):
                n, orient = _get_brkend_seg_orient(breakend)
                breakpoint_segment[n].add((bp_idx, be_idx, orient))

        # Count the number of segments in the new segmentation
        self.N1 = 0
        for n in range(-1, self.N):
            if n in breakpoint_segment:
                self.N1 += len(breakpoint_segment[n])
                if (n, n + 1) not in adjacencies:
                    self.N1 += 1
            elif n >= 0:
                self.N1 += 1

        # Mapping from old segment index to new segment index
        self.seg_fwd_remap = np.zeros(self.N, dtype=int)

        # New segment refers to an original segment
        self.seg_is_original = np.zeros(self.N1, dtype=bool)
        
        # Mapping from new segment index to old segment index
        self.seg_rev_remap = np.zeros(self.N1, dtype=int)
        
        # Create variables required for lower level model object
        self.num_breakpoints = len(self.breakpoints)
        self.is_telomere = np.ones(self.N1, dtype=int)
        self.breakpoint_idx = -np.ones(self.N1, dtype=int)
        self.breakpoint_orient = np.zeros(self.N1, dtype=int)

        # Index of new segmentation
        n_new = 0

        # There may be a breakend at the start of the first segment,
        # in which case there will be a breakend with segment n=-1
        for n in range(-1, self.N):
            if n >= 0:
                # Map old segment n to n_new
                self.seg_fwd_remap[n] = n_new
                self.seg_is_original[n_new] = True

            if n in breakpoint_segment:
                for bp_idx, be_idx, orient in breakpoint_segment[n]:
                    # Breakpoint index and orientation based on n_new
                    self.breakpoint_idx[n_new] = bp_idx
                    self.breakpoint_orient[n_new] = orient
                    self.seg_rev_remap[n_new] = n

                    # Breakpoint incident segments cannot be telomeres
                    self.is_telomere[n_new] = 0

                    # Next new segment, per introduced breakend
                    n_new += 1

                # If a breakend is at a telomere, create an additional new segment to be the telomere
                if (n, n + 1) not in adjacencies:
                    # Mark as a telomere
                    self.is_telomere[n_new] = 1
                    self.seg_rev_remap[n_new] = n

                    # Next new segment, after telomere
                    n_new += 1

            elif n >= 0:
                # If n is not a telomere, n_new is not a telomere
                if (n, n + 1) in adjacencies:
                    self.is_telomere[n_new] = 0

                self.seg_rev_remap[n_new] = n

                # Next new segment
                n_new += 1

        assert not np.any((self.breakpoint_idx >= 0) & (self.is_telomere == 1))
        assert np.all(np.bincount(self.breakpoint_idx[self.breakpoint_idx >= 0]) == 2)

        self.x1 = np.zeros((self.N1, x.shape[1]), dtype=float)
        self.l1 = np.zeros((self.N1,), dtype=float)

        self.x1[self.seg_fwd_remap, :] = x
        self.l1[self.seg_fwd_remap] = l

        # Mask likelihood of poorly modelled segments
        self._total_likelihood_mask = np.array([True] * len(self.l1))
        self._allele_likelihood_mask = np.array([True] * len(self.l1))

        # Add segment length mask
        self._length_mask = (self.l1 >= self.min_segment_length)
        self._total_likelihood_mask &= self._length_mask
        self._allele_likelihood_mask &= self._length_mask

        # Add proportion genotyped mask
        p = self.x1[:,:2].sum(axis=1).astype(float) / (self.x1[:,2].astype(float) + 1e-16)
        self._genotyped_mask = (p >= self.min_proportion_genotyped)
        self._allele_likelihood_mask &= self._genotyped_mask
        
        # Add amplification mask based on max depth
        depth = self.x1[:,2].astype(float) / (self.l1.astype(float) + 1e-16)
        self._depth_mask = (depth <= self.max_depth)
        self._total_likelihood_mask &= self._depth_mask
        self._allele_likelihood_mask &= self._depth_mask

        # Segments given windowed state spaces, excluded from parameter updates
        self._is_windowed = np.zeros(self.N1, dtype=bool)

        # Optionally disable integrated breakpoint copy number inference
        if self.disable_breakpoints:
            self.num_breakpoints = 0
            self.breakpoint_idx = -np.ones(self.breakpoint_idx.shape, dtype=int)
            self.breakpoint_orient = np.zeros(self.breakpoint_orient.shape, dtype=int)

        self.check_elbo = False
        self.prev_elbo = None
        self.prev_elbo_diff = None
        self.num_em_iter = 1
        self.num_update_iter = 1
        
        self.likelihood_params = [
            'negbin_r_0',
            'negbin_r_1',
            'betabin_M_0',
            'betabin_M_1',
        ]
        
        if not self.normal_contamination:
            self.likelihood_params.extend([
                'negbin_hdel_mu',
                'negbin_hdel_r_0',
                'negbin_hdel_r_1',
                'betabin_loh_p',
                'betabin_loh_M_0',
                'betabin_loh_M_1',
            ])

        self.likelihood_param_bounds = {
            'negbin_r_0': (10., 2000.),
            'negbin_r_1': (self.negbin_r_1_min, 2000.),
            'betabin_M_0': (10., 2000.),
            'betabin_M_1': (1., 2000.),
            'negbin_hdel_mu': (1e-9, 1e-4),
            'negbin_hdel_r_0': (10., 2000.),
            'negbin_hdel_r_1': (1., 200.),
            'betabin_loh_p': (1e-5, 1e-2),
            'betabin_loh_M_0': (10., 2000.),
            'betabin_loh_M_1': (1., 200.),
        }

    def create_cn_states(self, num_clones, num_alleles, cn_max, cn_diff_max):
        """ Create a list of allele specific copy number states for a single segment.
        """
        num_tumour_vars = (num_clones - 1) * num_alleles

        normal_cn = (1, 1)

        cn_states = dict()
        for cn in itertools.product(range(cn_max + 1), repeat=num_tumour_vars):
            cn = np.concatenate([normal_cn, cn]).reshape((num_clones, num_alleles))

            if not np.all(cn[1:, :].sum(axis=1) <= cn_max):
                continue

            if not np.all(cn[1:, :].max(axis=0) - cn[1:, :].min(axis=0) <= cn_diff_max):
                continue

            # Ensure states are non-redundant under swapping
            cn_key = tuple(cn[1:, :].flatten())
            cn_swapped_key = tuple(cn[1:, ::-1].flatten())

            cn_states[frozenset([cn_key, cn_swapped_key])] = cn

        cn_states = np.array(list(cn_states.values()))

        return cn_states

    def create_brk_states(self, num_clones, cn_max, cn_diff_max):
        """ Create a list of allele specific breakpoint copy number states.
        """
        num_tumour_vars = num_clones - 1
            
        normal_cn = (0,)

        brk_states = list()
        for cn in itertools.product(range(cn_max + 1), repeat=num_tumour_vars):
            cn = np.array(normal_cn + cn).reshape((num_clones,))

            if not np.all(cn <= cn_max):
                continue

            if not np.all(cn[1:].max() - cn[1:].min() <= cn_diff_max):
                continue

            brk_states.append(cn)

        brk_states = np.array(brk_states)

        return brk_states

    def get_likelihood_param_values(self):
        """ Get current likelihood parameter values.
        """
        likelihood_param_values = {}
        for name in self.likelihood_params:
            likelihood_param_values[name] = getattr(self.model, name)
        return likelihood_param_values

    def get_model_data(self):
        data = {}
        for a in dir(self.model):
            if a in ['__doc__', '__pyx_vtable__']:
                continue
            if callable(getattr(self.model, a)):
                continue
            if type(getattr(self.model, a)).__name__ == '_memoryviewslice':
                data[a] = np.asarray(getattr(self.model, a))
            else:
                data[a] = getattr(self.model, a)
        return data

    def _write_model(self, model_filename):
        data = self.get_model_data()
        pickle.dump(data, open(model_filename, 'w'))

    def _read_model(self, model_filename):
        data = pickle.load(open(model_filename, 'r'))
        for a in dir(self.model):
            if a in data:
                setattr(self.model, a, data[a])

    def _get_hdel_weights(self):
        mask = np.zeros(self.model.posterior_marginals.shape)
        n, s = np.where(np.asarray(self.model.is_hdel) == 1)
        mask[n, s] = 1
        weights = (np.asarray(self.model.posterior_marginals) * mask).sum(axis=-1)
        return weights

    def _get_loh_weights(self):
        mask = np.zeros(self.model.posterior_marginals.shape)
        n, s = np.where(np.asarray(self.model.is_loh) == 1)
        mask[n, s] = 1
        weights = (np.asarray(self.model.posterior_marginals) * mask).sum(axis=-1)
        return weights

    def get_param_sample_weight(self, name):
        """ Get segment specific weight for resampling segments
        for stochastic parameter estimation.
        """
        if name == 'negbin_r_0':
            weights = np.asarray(self.model.p_outlier_total[:, 0])
        elif name == 'negbin_r_1':
            weights = np.asarray(self.model.p_outlier_total[:, 1])
        elif name == 'betabin_M_0':
            weights = np.asarray(self.model.p_outlier_allele[:, 0])
        elif name == 'betabin_M_1':
            weights = np.asarray(self.model.p_outlier_allele[:, 1])
        elif name == 'negbin_hdel_mu':
            weights = self._get_hdel_weights()
        elif name == 'negbin_hdel_r_0':
            weights = self._get_hdel_weights() * np.asarray(self.model.p_outlier_total[:, 0])
        elif name == 'negbin_hdel_r_1':
            weights = self._get_hdel_weights() * np.asarray(self.model.p_outlier_total[:, 1])
        elif name == 'betabin_loh_p':
            weights = self._get_loh_weights()
        elif name == 'betabin_loh_M_0':
            weights = self._get_loh_weights() * np.asarray(self.model.p_outlier_allele[:, 0])
        elif name == 'betabin_loh_M_1':
            weights = self._get_loh_weights() * np.asarray(self.model.p_outlier_allele[:, 1])
        norm = weights.sum()
        if norm > 0.:
            return weights / norm
        else:
            print ('nothing for ' + name)
            return None

    def fit(self, h_init):
        """ Fit the model with a series of updates.
        """
        M = h_init.shape[0]

        cn_states = self.create_cn_states(M, 2, self.max_copy_number, self.max_copy_number_diff)
        cn_states = np.array([cn_states] * self.N)
        cn_states[:, :, 0, :] = self.normal_copies[:, np.newaxis, :]

        # Remap cn states
        cn_states = cn_states[self.seg_rev_remap, :, :, :]

        brk_states = self.create_brk_states(M, self.max_copy_number, self.max_copy_number_diff)

        self.model = remixt.bpmodel.RemixtModel(
            M,
            self.N1,
            self.num_breakpoints,
            self.normal_contamination,
            cn_states,
            brk_states,
            h_init,
            self.l1,
            self.x1[:, 2],
            self.x1[:, 0:2],
            self.is_telomere,
            self.breakpoint_idx,
            self.breakpoint_orient,
            self.transition_log_prob,
            self.divergence_weight,
        )

        self.model.total_likelihood_mask = self._total_likelihood_mask.astype(int)
        self.model.allele_likelihood_mask = self._allele_likelihood_mask.astype(int)
        self.model.negbin_r_1 = max(self.model.negbin_r_1, self.negbin_r_1_min)

        if self.breakpoint_init is not None:
            p_breakpoint = np.ones((self.model.self.num_breakpoints, self.model.num_brk_states))
            brk_states = np.array(self.model.brk_states)

            for k, bp in enumerate(self.breakpoints):
                cn = self.breakpoint_init[bp]

                for s in range(self.model.num_brk_states):
                    if np.all(cn == brk_states[s]):
                        p_breakpoint[k, s] = 1000.

            p_breakpoint /= np.sum(p_breakpoint, axis=-1)[:, np.newaxis]

            self.model.p_breakpoint = p_breakpoint

        self.model.transition_model = self.transition_model
        if self.transition_saturation is not None:
            self.model.transition_saturation = self.transition_saturation
        self.model.calculate_log_transmat(self.model.cached_log_transmat)

        if self.prev_elbo is None:
            self.prev_elbo = self.model.calculate_elbo()

        for i in range(self.num_em_iter):
            if self.amp_windows and i == self.amp_window_start_iter:
                self.switch_to_amp_windows()
            elif self.amp_windows and i > self.amp_window_start_iter:
                self.recenter_amp_windows()

            for j in range(self.num_update_iter):
                self.variational_update()

            if self.do_h_update:
                self.em_update_h()

            self.em_update_params()

            elbo = self.model.calculate_elbo()

            self.prev_elbo_diff = elbo - self.prev_elbo
            self.prev_elbo = elbo

            print ('[{}] completed iteration {}'.format(_gettime(), i))
            print ('[{}]     elbo: {:.10f}'.format(_gettime(), self.prev_elbo))
            print ('[{}]     elbo diff: {:.10f}'.format(_gettime(), self.prev_elbo_diff))
            print ('[{}]     h = {}'.format(_gettime(), np.asarray(self.model.h)))
            for name, value in self.get_likelihood_param_values().items():
                print ('[{}]     {} = {}'.format(_gettime(), name, value))

    @contextlib.contextmanager
    def elbo_check(self, name, threshold=-1e-6):
        print ('[{}] optimizing {}'.format(_gettime(), name))
        if not self.check_elbo:
            yield
            return
        elbo_before = self.model.calculate_elbo()
        yield
        elbo_after = self.model.calculate_elbo()
        print ('[{}]     elbo: {:.10f}'.format(_gettime(), elbo_after))
        print ('[{}]     elbo diff: {:.10f}'.format(_gettime(), elbo_after - elbo_before))
        if elbo_after - elbo_before < threshold:
            raise Exception('elbo error for step {}!'.format(name))

    def variational_update(self):
        """ Single update of all variational parameters.
        """
        with self.elbo_check('update_p_allele_swap'):
            self.model.update_p_allele_swap()

        with self.elbo_check('p_cn'):
            self.model.update_p_cn()

        with self.elbo_check('p_breakpoint'):
            self.model.update_p_breakpoint()

        with self.elbo_check('p_outlier_total'):
            self.model.update_p_outlier_total()

        with self.elbo_check('p_outlier_allele'):
            self.model.update_p_outlier_allele()

    def em_update_h(self):
        """ Single EM update of haploid read depth parameter.
        """
        with self.elbo_check('h'):
            self.update_h()

    def em_update_params(self):
        """ Single EM update of likelihood parameters.
        """
        for name in self.likelihood_params:
            with self.elbo_check(name):
                self.update_param(name)

    def _create_sample(self, weights=None):
        sample_size = int(min(200, self.model.num_segments / 10))
        sample_idxs = np.random.choice(self.model.num_segments, size=sample_size, replace=False, p=weights)
        sample = np.zeros((self.model.num_segments,), dtype=int)
        sample[sample_idxs] = 1
        return sample

    def update_h(self):
        """ Update haploid depths by optimizing expected likelihood.
        """
        def calculate_nll(h, model, sample):
            model.h = h
            nll = -model.calculate_expected_log_likelihood(sample)
            return nll

        def calculate_nll_partial_h(h, model, sample):
            model.h = h
            partial_h = np.zeros((model.num_clones,))
            model.calculate_expected_log_likelihood_partial_h(sample, partial_h)
            return -partial_h

        h_before = self.model.h
        elbo_before = self.model.calculate_expected_log_likelihood(np.ones((self.model.num_segments,), dtype=int))

        sample = self._create_sample(self._exclude_windowed(None))

        result = scipy.optimize.minimize(
            calculate_nll,
            self.model.h,
            method='L-BFGS-B',
            jac=calculate_nll_partial_h,
            bounds=[(1e-8, 10.)] * self.model.num_clones,
            args=(self.model, sample),
        )

        if not result.success:

            # Check the gradiant if the line search terminated abnormally, scipy < 1.15
            # reports 'ABNORMAL_TERMINATION_IN_LNSRCH', later versions 'ABNORMAL: '
            if str(result.message).startswith('ABNORMAL'):
                analytic_fprime = calculate_nll_partial_h(result.x, self.model, sample)
                numerical_fprime = statsmodels.tools.numdiff.approx_fprime(result.x, calculate_nll, args=(self.model, sample))

                if not np.allclose(analytic_fprime, numerical_fprime, atol=2.):
                    raise OptimizationError('gradiant error, analytic: {}, numerical: {}\n'.format(analytic_fprime, numerical_fprime))

            else:
                raise OptimizationError('optimization failed\n{}'.format(result))

        self.model.h = result.x
        elbo_after = self.model.calculate_expected_log_likelihood(np.ones((self.model.num_segments,), dtype=int))

        if elbo_after < elbo_before:
            print ('[{}] h rejected, elbo before: {}, after: {}'.format(_gettime(), elbo_before, elbo_after))
            self.model.h = h_before

        else:
            self.model.h = result.x

    def update_param(self, name):
        """ Update named param by optimizing expected likelihood.
        """
        bounds = self.likelihood_param_bounds[name]
        weights = self._exclude_windowed(self.get_param_sample_weight(name))

        def calculate_nll(value, model, name, bounds, sample):
            assert value.shape == (1,)
            value = float(value[0])
            if value < bounds[0] or value > bounds[1]:
                return np.inf
            setattr(model, name, value)
            nll = -model.calculate_expected_log_likelihood(sample)
            return nll

        value_before = getattr(self.model, name)
        elbo_before = self.model.calculate_expected_log_likelihood(np.ones((self.model.num_segments,), dtype=int))

        sample = self._create_sample(weights)

        result = scipy.optimize.brute(
            calculate_nll,
            args=(self.model, name, bounds, sample),
            ranges=[bounds],
            full_output=True,
        )

        assert result[0].shape == (1,)
        result_value = float(result[0][0])

        elbo_after = self.model.calculate_expected_log_likelihood(np.ones((self.model.num_segments,), dtype=int))
        if elbo_after < elbo_before:
            print ('[{}] {} rejected, elbo before: {}, after: {}'.format(_gettime(), name, elbo_before, elbo_after))
            setattr(self.model, name, value_before)

        else:
            setattr(self.model, name, result_value)

    def _exclude_windowed(self, weights):
        """ Zero sampling weights of windowed segments, so that high copy number
        segments do not drive the haploid depth and likelihood parameters.
        """
        if not np.any(self._is_windowed):
            return weights
        if weights is None:
            weights = np.ones(self.N1)
        weights = weights * (~self._is_windowed)
        norm = weights.sum()
        if norm <= 0:
            return None
        return weights / norm

    def estimate_allele_cn(self, h, allow_short=False):
        """ Read depth estimate of tumour major and minor copy number per segment,
        averaged over tumour clones.

        Returns:
            numpy.array: N1 x 2 major, minor estimates, nan for segments without reads
        """
        normal_copies = np.sort(self.normal_copies[self.seg_rev_remap], axis=1)[:, ::-1]
        depth = self.x1[:, 2] / (self.l1 + 1e-16)
        allelic = self.x1[:, 0] + self.x1[:, 1]
        frac_major = np.where(allelic > 0, self.x1[:, 0] / (allelic + 1e-16), 0.5)

        # Segments with few allele specific reads, for instance with no heterozygous
        # snps in a region of loss of heterozygosity, take the major allele fraction
        # of whichever nearest segment with enough reads, on either side, is closest
        # in depth, so that segments inside an amplicon do not borrow from its flanks
        is_reliable = (allelic >= 50) & self._genotyped_mask & self._length_mask
        log_depth = np.log(depth + 1e-16)
        reliable_frac = frac_major.copy()
        for n in np.where(~is_reliable)[0]:
            neighbours = []
            for step in (-1, 1):
                k = n
                while True:
                    boundary = k - 1 if step < 0 else k
                    if boundary < 0 or boundary >= self.N1 - 1 or self.is_telomere[boundary] != 0:
                        break
                    k += step
                    if is_reliable[k]:
                        neighbours.append(k)
                        break
            if not neighbours:
                reliable_frac[n] = 0.5
            elif self.l1[n] > 0 and self.x1[n, 2] > 0:
                reliable_frac[n] = frac_major[min(neighbours, key=lambda k: abs(log_depth[k] - log_depth[n]))]
            else:
                reliable_frac[n] = np.mean(frac_major[neighbours])
        frac_major = reliable_frac

        h_normal = h[0] if self.normal_contamination else 0.
        major = (depth * frac_major - h_normal * normal_copies[:, 0]) / h[1:].sum()
        minor = (depth * (1. - frac_major) - h_normal * normal_copies[:, 1]) / h[1:].sum()
        estimate = np.maximum(np.array([major, minor]).T, 0.)
        if allow_short:
            estimate[(self.l1 <= 0) | (self.x1[:, 2] <= 0)] = np.nan
        else:
            estimate[(self.l1 < self.min_segment_length) | (self.x1[:, 2] <= 0)] = np.nan
        return estimate

    def calculate_amp_windows(self, h):
        """ Find segments needing windowed states and the window centres.

        Segments with a read depth estimate of copy number close to or above the
        state space maximum are windowed around the estimate.  Segments too short
        for an estimate, such as the zero length segments between breakends, take
        windows around their nearest estimated neighbours on each side, if either
        is windowed.

        Returns:
            dict: segment index to list of (major, minor) window centres
        """
        estimate = self.estimate_allele_cn(h)
        is_estimated = ~np.isnan(estimate[:, 0])
        is_high = is_estimated & (np.nan_to_num(estimate.sum(axis=1)) > self.max_copy_number - 2)
        short_estimate = self.estimate_allele_cn(h, allow_short=True)

        centres = {}
        for n in np.where(is_high)[0]:
            centres[n] = [tuple(np.round(estimate[n]).astype(int))]

        # Nearest estimated neighbours without crossing a telomere
        for n in np.where(~is_estimated)[0]:
            neighbours = []
            k = n - 1
            while k >= 0 and self.is_telomere[k] == 0 and not is_estimated[k]:
                k -= 1
            if k >= 0 and self.is_telomere[k] == 0:
                neighbours.append(k)
            k = n
            while k < self.N1 - 1 and self.is_telomere[k] == 0 and not is_estimated[k + 1]:
                k += 1
            if k < self.N1 - 1 and self.is_telomere[k] == 0:
                neighbours.append(k + 1)
            own = []
            if self.amp_window_seed_short and not np.isnan(short_estimate[n, 0]):
                own = [tuple(np.round(short_estimate[n]).astype(int))]
            own_high = len(own) > 0 and sum(own[0]) > self.max_copy_number - 2
            if not own_high and not any(is_high[k] for k in neighbours):
                continue
            centres[n] = sorted(set(tuple(np.round(estimate[k]).astype(int)) for k in neighbours) | set(own))

        return centres

    def create_window_states(self, centres, num_clones, num_states):
        """ Create allele specific states in windows around centres, padded to
        num_states with a disabled copy of the first state.

        Returns:
            numpy.array: num_states x num_clones x 2 states, normal clone zero
            numpy.array: num_states log prior, 0 or a large penalty for padding
        """
        major_radius = self.amp_window_major_radius
        minor_radius = self.amp_window_minor_radius
        if len(centres) > 1:
            major_radius = max(1, major_radius - 1)
            minor_radius = max(1, minor_radius - 1)

        num_tumour = num_clones - 1
        states = dict()
        distance = dict()
        for centre_major, centre_minor in centres:
            for a in range(centre_major - major_radius, centre_major + major_radius + 1):
                for b in range(centre_minor - minor_radius, centre_minor + minor_radius + 1):
                    if a < 0 or b < 0:
                        continue
                    for deltas in itertools.product((-1, 0, 1), repeat=2 * (num_tumour - 1)):
                        cn = np.array([[a, b]] + [[a + deltas[2 * i], b + deltas[2 * i + 1]] for i in range(num_tumour - 1)])
                        if np.any(cn < 0):
                            continue
                        if not np.all(cn.max(axis=0) - cn.min(axis=0) <= self.max_copy_number_diff):
                            continue
                        key = frozenset([tuple(cn.flatten()), tuple(cn[:, ::-1].flatten())])
                        d = min(abs(a - cm) + abs(b - cb) for cm, cb in centres)
                        if key not in states or d < distance[key]:
                            states[key] = cn
                            distance[key] = d

        keys = sorted(states, key=lambda k: (distance[k], sorted(k)))[:num_states]
        cn_states = np.zeros((num_states, num_clones, 2), dtype=int)
        log_prior = np.zeros(num_states)
        for s in range(num_states):
            if s < len(keys):
                cn_states[s, 1:, :] = states[keys[s]]
            else:
                cn_states[s, 1:, :] = states[keys[0]]
                log_prior[s] = -1e6

        return cn_states, log_prior

    def _rebuild_model(self, cn_states, log_prior_state, brk_states, total_mask, allele_mask):
        """ Replace the model with one with new state spaces, keeping the other
        variational distributions and parameters.
        """
        old = self.model
        keep = {
            'h': np.asarray(old.h).copy(),
            'p_outlier_total': np.asarray(old.p_outlier_total).copy(),
            'p_outlier_allele': np.asarray(old.p_outlier_allele).copy(),
            'p_allele_swap': np.asarray(old.p_allele_swap).copy(),
            'prior_outlier_total': old.prior_outlier_total,
            'prior_outlier_allele': old.prior_outlier_allele,
            'transition_model': old.transition_model,
            'transition_saturation': old.transition_saturation,
        }
        for name in self.likelihood_param_bounds:
            keep[name] = getattr(old, name)

        old_brk_states = np.asarray(old.brk_states)
        old_p_breakpoint = np.asarray(old.p_breakpoint).copy()

        self.model = None
        del old

        model = remixt.bpmodel.RemixtModel(
            cn_states.shape[2],
            self.N1,
            self.num_breakpoints,
            self.normal_contamination,
            cn_states,
            brk_states,
            keep.pop('h'),
            self.l1,
            self.x1[:, 2],
            self.x1[:, 0:2],
            self.is_telomere,
            self.breakpoint_idx,
            self.breakpoint_orient,
            self.transition_log_prob,
            self.divergence_weight,
        )

        for name, value in keep.items():
            setattr(model, name, value)

        model.total_likelihood_mask = total_mask.astype(int)
        model.allele_likelihood_mask = allele_mask.astype(int)
        model.log_prior_state = log_prior_state

        # Carry over breakpoint probabilities for states in both spaces
        brk_index = {tuple(b): i for i, b in enumerate(old_brk_states)}
        p_breakpoint = np.zeros((self.num_breakpoints, brk_states.shape[0]))
        for s_b, b in enumerate(brk_states):
            if tuple(b) in brk_index:
                p_breakpoint[:, s_b] = old_p_breakpoint[:, brk_index[tuple(b)]]
        p_breakpoint += 1e-12
        p_breakpoint /= p_breakpoint.sum(axis=-1)[:, np.newaxis]
        model.p_breakpoint = p_breakpoint
        model.calculate_log_transmat(model.cached_log_transmat)

        self.model = model

        # Copy number posteriors are indexed by state, recompute before other updates
        self.model.update_p_cn()

    def _amp_window_state_space(self, h):
        """ State spaces with windows for high copy number segments at haploid depths h.
        """
        return self._window_state_space_from_centres(h.shape[0], self.calculate_amp_windows(h))

    def _window_state_space_from_centres(self, M, centres):
        """ State spaces with windows around the given centres per segment.
        """
        S = self._base_cn_states.shape[1]

        cn_states = self._base_cn_states.copy()
        log_prior_state = np.zeros((self.N1, S))
        is_windowed = np.zeros(self.N1, dtype=bool)
        for n, segment_centres in centres.items():
            states, log_prior = self.create_window_states(segment_centres, M, S)
            states[:, 0, :] = self._base_cn_states[n, 0, 0, :]
            cn_states[n] = states
            log_prior_state[n] = log_prior
            is_windowed[n] = True

        cn_max = max(self.max_copy_number, int(cn_states[:, :, 1:, :].sum(axis=-1).max()))
        brk_states = self.create_brk_states(M, cn_max, self.max_copy_number_diff)

        return centres, cn_states, log_prior_state, is_windowed, brk_states

    def switch_to_amp_windows(self):
        """ Remove the depth mask and give high copy number segments windowed states.
        """
        h = np.asarray(self.model.h).copy()
        self._base_cn_states = np.asarray(self.model.cn_states).copy()

        centres, cn_states, log_prior_state, is_windowed, brk_states = self._amp_window_state_space(h)
        self._amp_window_centres = centres
        self._is_windowed = is_windowed

        self._total_likelihood_mask = self._length_mask.copy()
        self._allele_likelihood_mask = self._length_mask & self._genotyped_mask

        print ('[{}] switching to amp windows for {} segments, max brk cn {}'.format(
            _gettime(), is_windowed.sum(), brk_states.max()))

        self._rebuild_model(cn_states, log_prior_state, brk_states,
            self._total_likelihood_mask, self._allele_likelihood_mask)

        self.prev_elbo = self.model.calculate_elbo()

    def recenter_amp_windows(self):
        """ Move windows, keeping the move only if the elbo increases.

        With amp_window_recenter 'depth', windows move to the read depth estimates
        at the current haploid depths.  With 'q', windows move to the states with
        the highest local variational score given the neighbouring posteriors, see
        propose_window_centres, repeated for a few sweeps so that adjacent windows
        can move together.
        """
        h = np.asarray(self.model.h).copy()

        elbo_before = self.model.calculate_elbo()
        previous = (
            np.asarray(self.model.cn_states).copy(),
            np.asarray(self.model.log_prior_state).copy(),
            np.asarray(self.model.brk_states).copy(),
            np.asarray(self.model.p_breakpoint).copy(),
            self._is_windowed.copy(),
            self._amp_window_centres,
        )

        num_moved = 0
        sweeps = 1 if self.amp_window_recenter == 'depth' else self.amp_window_sweeps
        for sweep in range(sweeps):
            if self.amp_window_recenter == 'depth':
                centres, cn_states, log_prior_state, is_windowed, brk_states = self._amp_window_state_space(h)
            else:
                centres = self.propose_window_centres()
                centres, cn_states, log_prior_state, is_windowed, brk_states = self._window_state_space_from_centres(h.shape[0], centres)

            moved = [n for n in set(centres) | set(self._amp_window_centres)
                if centres.get(n) != self._amp_window_centres.get(n)]
            if len(moved) == 0:
                break
            num_moved += len(moved)
            print ('[{}] sweep {} moving {} amp windows: {}'.format(_gettime(), sweep, len(moved),
                ', '.join('{}:{}->{}'.format(n, self._amp_window_centres.get(n), centres.get(n)) for n in moved[:20])))

            self._rebuild_model(cn_states, log_prior_state, brk_states,
                self._total_likelihood_mask, self._allele_likelihood_mask)
            self.model.update_p_breakpoint()
            self.model.update_p_cn()
            self.model.update_p_outlier_total()
            self.model.update_p_outlier_allele()
            self.model.update_p_cn()
            self._is_windowed = is_windowed
            self._amp_window_centres = centres

        if num_moved == 0:
            return

        elbo_after = self.model.calculate_elbo()

        if elbo_after >= elbo_before:
            print ('[{}] moved {} amp windows, elbo {} -> {}'.format(_gettime(), num_moved, elbo_before, elbo_after))
        else:
            print ('[{}] rejected moving {} amp windows, elbo {} -> {}'.format(_gettime(), num_moved, elbo_before, elbo_after))
            cn_states, log_prior_state, brk_states, p_breakpoint, self._is_windowed, self._amp_window_centres = previous
            self._rebuild_model(cn_states, log_prior_state, brk_states,
                self._total_likelihood_mask, self._allele_likelihood_mask)
            self.model.p_breakpoint = p_breakpoint
            self.model.calculate_log_transmat(self.model.cached_log_transmat)
            self.model.update_p_cn()

    def _transition_function(self, d):
        d = np.abs(d)
        if self.model.transition_model == 1:
            return (d > 0).astype(float)
        elif self.model.transition_model == 2:
            return np.minimum(d, self.model.transition_saturation)
        return d.astype(float)

    def _expected_junction_log_potential(self, j, cand, neighbour_states, q, cand_is_left):
        """ Expected log transition potential at junction j (between segments j and
        j + 1) for each candidate state, with respect to the posterior marginal q
        of the neighbouring segment and the breakpoint posterior.

        Args:
            cand: K x M x 2 candidate states
            neighbour_states: S x M x 2 neighbour states
            q: S neighbour posterior marginal
            cand_is_left: candidate segment is j, neighbour is j + 1
        Returns:
            numpy.array: K expected log potentials
        """
        keep = q > 1e-8
        neighbour_states = neighbour_states[keep]
        q = q[keep] / q[keep].sum()

        if cand_is_left:
            left, right = cand[:, np.newaxis], neighbour_states[np.newaxis, :]
        else:
            left, right = neighbour_states[np.newaxis, :], cand[:, np.newaxis]

        d_total = left.sum(axis=-1) - right.sum(axis=-1)   # K x S x M

        if self.model.breakpoint_idx[j] < 0:
            total_term = self._transition_function(d_total).sum(axis=-1)
        else:
            brk_states = np.asarray(self.model.brk_states)
            p_b = np.asarray(self.model.p_breakpoint)[self.model.breakpoint_idx[j]]
            orient = self.model.breakpoint_orient[j]
            d_range = np.arange(d_total.min(), d_total.max() + 1)
            total_term = np.zeros(d_total.shape[:2])
            for m in range(cand.shape[1]):
                g = (p_b[np.newaxis, :] * self._transition_function(
                    d_range[:, np.newaxis] - orient * brk_states[np.newaxis, :, m])).sum(axis=1)
                total_term += g[d_total[:, :, m] - d_range[0]]

        same = np.abs(left - right).astype(float)
        flipped = np.abs(left - right[..., ::-1]).astype(float)
        if self.model.transition_model != 0:
            same = self._transition_function(left - right)
            flipped = self._transition_function(left - right[..., ::-1])
        total_change = self._transition_function(d_total).sum(axis=-1)
        allele_term = np.minimum(same.sum(axis=(-1, -2)), flipped.sum(axis=(-1, -2))) - total_change

        return -self.model.transition_penalty * ((total_term + allele_term) * q[np.newaxis, :]).sum(axis=1)

    def _candidate_log_likelihood(self, n, cand):
        """ Log likelihood of each candidate state for segment n, with the outlier
        indicators summed out under their priors and the allele swap taken in
        expectation under its posterior.

        Summing out the outlier indicators, rather than weighting by their current
        posteriors, lets a segment whose depth is currently explained as an outlier
        move to the candidate that fits its depth.
        """
        import scipy.special
        h = np.asarray(self.model.h)
        ll = np.zeros(cand.shape[0])
        total = cand.sum(axis=-1)

        if self.model.total_likelihood_mask[n]:
            x = self.x1[n, 2]
            mu = self.l1[n] * (total * h[np.newaxis, :]).sum(axis=1)
            log_prior = (np.log(1. - self.model.prior_outlier_total), np.log(self.model.prior_outlier_total))
            terms = []
            for u, r in enumerate((self.model.negbin_r_0, self.model.negbin_r_1)):
                p = mu / (r + mu)
                p = np.where((p < 0) | (p > 1), 0.5, p)
                p = np.clip(p, 1e-300, 1. - 1e-16)
                nb = (scipy.special.gammaln(x + r) - scipy.special.gammaln(x + 1) - scipy.special.gammaln(r)
                    + x * np.log(p) + r * np.log(1. - p))
                terms.append(log_prior[u] + nb)
            ll += np.logaddexp(terms[0], terms[1])

        allelic = self.x1[n, 0] + self.x1[n, 1]
        if self.model.allele_likelihood_mask[n] and allelic > 0:
            ratio = (cand[:, :, 0] * h[np.newaxis, :]).sum(axis=1) / (total * h[np.newaxis, :]).sum(axis=1)
            ratio = np.clip(ratio, 1e-6, 1. - 1e-6)
            G = scipy.special.gammaln
            log_prior = (np.log(1. - self.model.prior_outlier_allele), np.log(self.model.prior_outlier_allele))
            terms = []
            for v, M in enumerate((self.model.betabin_M_0, self.model.betabin_M_1)):
                expected = 0.
                for w in range(2):
                    k = self.x1[n, w]
                    bb = (G(allelic + 1) - G(k + 1) - G(allelic - k + 1)
                        + G(k + M * ratio) + G(allelic - k + M * (1 - ratio)) - G(allelic + M)
                        - G(M * ratio) - G(M * (1 - ratio)) + G(M))
                    expected = expected + np.asarray(self.model.p_allele_swap)[n, w] * bb
                terms.append(log_prior[v] + expected)
            ll += np.logaddexp(terms[0], terms[1])

        return ll

    def propose_window_centres(self):
        """ Window centres from the local variational score of candidate states.

        For each windowed segment, clonal candidate states (major, minor) over a
        broad grid are scored by expected log likelihood plus the expected log
        transition potentials to the neighbouring segments, taken with respect
        to the neighbours' current posterior marginals and the breakpoint
        posteriors.  This is the coordinate ascent update for that segment with
        its state space unrestricted, so windows can move to states outside the
        current window, for instance to the level at which a breakpoint explains
        the copy number change on both of its sides.

        Returns:
            dict: segment index to list of one (major, minor) window centre
        """
        cn_states = np.asarray(self.model.cn_states)
        q_all = np.asarray(self.model.posterior_marginals)
        M = cn_states.shape[2]
        normal = cn_states[:, 0, 0, :]

        estimates = self.estimate_allele_cn(np.asarray(self.model.h), allow_short=True)

        centres = {}
        for n in sorted(self._amp_window_centres):
            neighbours = []
            if n > 0 and self.model.is_telomere[n - 1] == 0:
                neighbours.append((n - 1, n - 1, False))
            if n < self.N1 - 1 and self.model.is_telomere[n] == 0:
                neighbours.append((n + 1, n, True))

            # Candidate grid: totals up to above the largest copy number in reach,
            # minor copy numbers near those of the segment and its neighbours
            q_n = q_all[n]
            relevant = [cn_states[n][q_n > 1e-3]]
            for k, _, _ in neighbours:
                relevant.append(cn_states[k][q_all[k] > 1e-3])
            relevant = np.concatenate(relevant)[:, 1:, :]
            max_total = int(relevant.sum(axis=-1).max())
            estimate = estimates[n]
            if not np.isnan(estimate[0]):
                max_total = max(max_total, int(np.round(estimate.sum())))
            max_total = int(max_total * 1.2) + 5
            minors = set()
            for b in np.unique(relevant.min(axis=-1)):
                minors.update(range(max(0, b - 1), b + 2))
            minors.update(range(0, 3))
            grid = [(a, b) for b in sorted(minors) for a in range(b, max_total - b + 1)]
            cand = np.zeros((len(grid), M, 2), dtype=int)
            cand[:, 0, :] = normal[n]
            cand[:, 1:, 0] = np.array(grid)[:, np.newaxis, 0]
            cand[:, 1:, 1] = np.array(grid)[:, np.newaxis, 1]

            score = self._candidate_log_likelihood(n, cand)
            for k, j, cand_is_left in neighbours:
                score += self._expected_junction_log_potential(j, cand, cn_states[k], q_all[k], cand_is_left)

            centres[n] = [tuple(int(c) for c in grid[int(np.argmax(score))])]

        return centres

    def optimal_cn(self):
        cn = np.zeros((self.model.num_segments, self.model.num_clones, self.model.num_alleles), dtype=int)

        self.model.infer_cn(cn)

        log_breakpoint_p = np.zeros((self.model.num_breakpoints, self.model.num_brk_states))

        for n in range(0, self.model.num_segments - 1):
            if self.model.breakpoint_idx[n] < 0:
                continue

            for m in range(self.model.num_clones):
                d = cn[n, m, :].sum(axis=-1) - cn[n + 1, m, :].sum(axis=-1)

                for s_b in range(self.model.num_brk_states):
                    log_breakpoint_p[self.model.breakpoint_idx[n], s_b] += (
                        -self.model.transition_penalty * abs(d - self.model.breakpoint_orient[n] * self.model.brk_states[s_b, m]))

        brk_cn = dict()

        for k in range(self.model.num_breakpoints):
            s_b = log_breakpoint_p[k, :].argmax()
            brk_cn[self.breakpoint_ids[k]] = np.asarray(self.model.brk_states)[s_b]

        # Remap cn to original segmentation
        cn = cn[self.seg_fwd_remap]

        return cn, brk_cn

    def breakpoint_prob(self):
        p_breakpoint = np.asarray(self.model.p_breakpoint)
        brk_prob = dict(zip(self.breakpoints, p_breakpoint))

        return brk_prob

    @property
    def h(self):
        return np.asarray(self.model.h)

    @property
    def p_breakpoint(self):
        return np.asarray(self.model.p_breakpoint)

    @property
    def p_outlier_total(self):
        return np.asarray(self.model.p_outlier_total)[self.seg_fwd_remap]

    @property
    def p_outlier_allele(self):
        return np.asarray(self.model.p_outlier_allele)[self.seg_fwd_remap]

    @property
    def total_likelihood_mask(self):
        return np.asarray(self.model.total_likelihood_mask)[self.seg_fwd_remap]

    @property
    def is_windowed(self):
        return self._is_windowed[self.seg_fwd_remap]

    @property
    def allele_likelihood_mask(self):
        return np.asarray(self.model.allele_likelihood_mask)[self.seg_fwd_remap]


def decode_breakpoints_naive(cn, adjacencies, breakpoints):
    """ Naive decoding of breakpoint copy number.  Finds most likely set of copy numbers given h.

    Args:
        cn (numpy.array): copy number matrix
        adjacencies (list of tuple): ordered pairs of segments representing wild type adjacencies
        breakpoints (dict of frozenset of tuple): list of pairs of segment/side pairs representing detected breakpoints

    Returns:
        pandas.DataFrame: table of breakpoint copy number with columns:
            'n_1', 'ell_1', 'side_1', 'n_2', 'ell_2', 'side_2', 'cn_*'

    """
    
    # Calculate breakpoint copy number based on total copy number transitions
    cn = cn.sum(axis=-1)

    breakend_adj = dict()
    for seg_1, seg_2 in adjacencies:
        breakend_1 = (seg_1, 1)
        breakend_2 = (seg_2, 0)
        breakend_adj[breakend_1] = breakend_2
        breakend_adj[breakend_2] = breakend_1

    M = cn.shape[1]

    brk_cn = dict()

    for breakpoint_id, breakpoint in breakpoints.items():

        # Calculate the copy number 'flow' at each breakend
        breakend_cn = dict()

        for breakend in breakpoint:
            n, side = breakend

            cn_self = cn[n,:]

            if breakend in breakend_adj:
                n_adj, side_adj = breakend_adj[breakend]
                cn_adj = cn[n_adj, :]
            else:
                cn_adj = 0

            cn_residual = np.maximum(cn_self - cn_adj, 0)

            breakend_cn[(n, side)] = cn_residual

        ((n_1, side_1), (n_2, side_2)) = breakpoint

        breakpoint_cn = np.minimum(
            breakend_cn[(n_1, side_1)],
            breakend_cn[(n_2, side_2)])

        brk_cn[breakpoint_id] = breakpoint_cn

    return brk_cn
