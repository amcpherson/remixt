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
        self._total_likelihood_mask &= (self.l1 >= self.min_segment_length)
        self._allele_likelihood_mask &= (self.l1 >= self.min_segment_length)

        # Add proportion genotyped mask
        p = self.x1[:,:2].sum(axis=1).astype(float) / (self.x1[:,2].astype(float) + 1e-16)
        self._allele_likelihood_mask &= (p >= self.min_proportion_genotyped)
        
        # Add amplification mask based on max depth
        depth = self.x1[:,2].astype(float) / (self.l1.astype(float) + 1e-16)
        self._total_likelihood_mask &= (depth <= self.max_depth)
        self._allele_likelihood_mask &= (depth <= self.max_depth)

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
            'negbin_r_1': (1., 2000.),
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

        if self.prev_elbo is None:
            self.prev_elbo = self.model.calculate_elbo()

        for i in range(self.num_em_iter):
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

        sample = self._create_sample()

        result = scipy.optimize.minimize(
            calculate_nll,
            self.model.h,
            method='L-BFGS-B',
            jac=calculate_nll_partial_h,
            bounds=[(1e-8, 10.)] * self.model.num_clones,
            args=(self.model, sample),
        )

        if not result.success:

            # Check the gradiant if optimization failed
            if result.message == 'ABNORMAL_TERMINATION_IN_LNSRCH':
                analytic_fprime = calculate_nll_partial_h(result.x, self.model, sample)
                numerical_fprime = statsmodels.tools.numdiff.approx_fprime(result.x, calculate_nll, args=(self.model, sample))

                if not np.allclose(analytic_fprime, numerical_fprime, atol=2.):
                    raise ValueError('gradiant error, analytic: {}, numerical: {}\n'.format(analytic_fprime, numerical_fprime))

            else:
                raise ValueError('optimization failed\n{}'.format(result)) 

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
        weights = self.get_param_sample_weight(name)

        def calculate_nll(value, model, name, bounds, sample):
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

        elbo_after = self.model.calculate_expected_log_likelihood(np.ones((self.model.num_segments,), dtype=int))
        if elbo_after < elbo_before:
            print ('[{}] {} rejected, elbo before: {}, after: {}'.format(_gettime(), name, elbo_before, elbo_after))
            setattr(self.model, name, value_before)

        else:
            setattr(self.model, name, result[0])

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
