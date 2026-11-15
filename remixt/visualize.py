""" Standalone HTML report of remixt results for one or more tumours.

The report is a single HTML file with the results of each tumour embedded as
gzipped JSON, and plotting done in the browser without external dependencies,
so the file can be moved or shared on its own.

Every solution is embedded, not only the optimal one, so solutions can be
compared in the report.  To keep the file small, segment depths are stored
once per tumour and raw copy number is computed in the browser from the
haploid depths of each solution.
"""

import base64
import gzip
import html
import importlib.resources
import json
import logging
import re

import numpy as np
import pandas as pd
import scipy.stats

import remixt
import remixt.utils


TEMPLATE_NAME = 'visualize_report.html'

# Number of points of the read depth density curves
NUM_DENSITY_POINTS = 400


def chromosome_sort_key(chromosome):
    """ Sort chromosomes numerically, then X, Y, M, then any others by name.
    """
    name = re.sub('^chr', '', str(chromosome))
    if name.isdigit():
        return (0, int(name), '')
    order = {'X': 1, 'Y': 2, 'M': 3, 'MT': 3}
    return (order.get(name, 4), 0, name)


def _rounded(values, decimals):
    """ List of rounded floats with nan and inf as None for JSON.
    """
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), np.round(values, decimals), np.nan)
    return [None if np.isnan(v) else float(v) for v in values]


def _ints(values):
    return [int(v) for v in np.asarray(values)]


def _weighted_density(xs, data, weights, bw_method=0.02):
    """ Weighted gaussian kernel density, zero where it cannot be estimated.
    """
    finite = np.isfinite(data) & np.isfinite(weights) & (weights > 0)
    if finite.sum() < 2 or np.ptp(data[finite]) == 0:
        return np.zeros(len(xs))
    density = scipy.stats.gaussian_kde(data[finite], weights=weights[finite], bw_method=bw_method)
    return density(xs)


def read_depth_densities(read_depth):
    """ Length weighted densities of the major, minor and total read depth of segments.
    """
    read_depth = read_depth.replace([np.inf, -np.inf], np.nan).dropna(subset=['major', 'minor', 'total', 'length'])
    high_quality = read_depth[read_depth['high_quality']]
    if len(high_quality) >= 2:
        read_depth = high_quality

    if len(read_depth) < 2:
        return None

    depth_max = remixt.utils.weighted_percentile(read_depth['total'].values, read_depth['length'].values, 95)
    depths = np.linspace(0., depth_max * 1.1, NUM_DENSITY_POINTS)

    densities = {'depth': _rounded(depths, 6)}
    for allele in ('minor', 'major', 'total'):
        density = _weighted_density(depths, read_depth[allele].values, read_depth['length'].values)
        densities[allele] = _rounded(density, 4)

    return densities


MASK_NONE, MASK_SHORT, MASK_HIGH_DEPTH, MASK_LOW_GENOTYPED = 0, 1, 2, 3


def segment_mask_reasons(segments):
    """ Reason each segment is masked from the likelihood.

    The model masks the total and allele likelihood of segments shorter than a
    minimum length or with depth above a max depth, and additionally the allele
    likelihood of segments with too few genotyped reads, see remixt.cn_model.
    Each reason is stored in the results as a column of the copy number table.
    A segment masked for several reasons is given the first in the order
    short, high depth, low genotyped.

    Args:
        segments (pandas.DataFrame): copy number table of a solution

    Returns:
        numpy.array: MASK_* reason for each segment
    """
    reasons = np.full(len(segments.index), MASK_NONE)
    reasons[segments['is_low_genotyped'].astype(bool).values] = MASK_LOW_GENOTYPED
    reasons[segments['is_high_depth'].astype(bool).values] = MASK_HIGH_DEPTH
    reasons[segments['is_short'].astype(bool).values] = MASK_SHORT

    return reasons


def _solution_ids(store):
    """ Ids of solutions in a results store, in order of init id.
    """
    stats = store['stats']
    return sorted(stats['init_id'].astype(int).values)


def _breakpoint_type(row):
    if row['chromosome_1'] != row['chromosome_2']:
        return 'translocation'
    if row['strand_1'] == row['strand_2']:
        return 'inversion'
    ends = sorted([(row['position_1'], row['strand_1']), (row['position_2'], row['strand_2'])])
    return 'deletion' if ends[0][1] == '+' else 'duplication'


def load_sample(results_filename):
    """ Load the results of one tumour into a JSON serializable dict.

    The optimal solution, the max depth and the max proportion divergent are
    read from the params table of the results.

    Args:
        results_filename (str): results HDF5 from remixt collate

    Returns:
        dict: segments, breakpoints, solutions and read depth density
    """
    with pd.HDFStore(results_filename, 'r') as store:
        solution_ids = _solution_ids(store)
        stats = store['stats'].set_index('init_id')
        failed_fits = store['failed_fits']
        read_depth = store['read_depth']
        minor_modes = store['minor_modes']
        optimal_mix = store['mix'].values
        params = store['params'].to_dict()

        solutions_cn = {}
        solutions_brk_cn = {}
        solutions_h = {}
        for solution_id in solution_ids:
            prefix = f'/solutions/solution_{solution_id}'
            solutions_cn[solution_id] = store[prefix + '/cn']
            solutions_brk_cn[solution_id] = store[prefix + '/brk_cn']
            solutions_h[solution_id] = store[prefix + '/h'].values

    max_prop_diverge = float(params['max_prop_diverge'])
    max_depth = float(params['max_depth'])
    optimal_id = int(params['optimal_init_id'])

    # Segments are shared by all solutions, taken from the optimal solution
    segments = solutions_cn[optimal_id].copy()
    segments['chromosome'] = segments['chromosome'].astype(str)
    num_clones = len(solutions_h[optimal_id])

    # Breakpoints are shared by all solutions, with copy number per solution
    breakpoints = solutions_brk_cn[optimal_id]
    if len(breakpoints) > 0 and 'chromosome_1' in breakpoints:
        breakpoints = breakpoints.drop_duplicates('prediction_id').copy()
        breakpoints['type'] = breakpoints.apply(_breakpoint_type, axis=1)
    else:
        breakpoints = pd.DataFrame(columns=['prediction_id'])
    breakpoint_ids = breakpoints['prediction_id'].values

    solutions = []
    for solution_id in solution_ids:
        cn = solutions_cn[solution_id]
        h = solutions_h[solution_id]
        solution_stats = stats.loc[solution_id]
        solution = {
            'id': int(solution_id),
            'h': _rounded(h, 8),
            'elbo': float(solution_stats['elbo']),
            'ploidy': float(solution_stats['ploidy']),
            'proportion_divergent': float(solution_stats['proportion_divergent']),
            'divergence_weight': float(solution_stats['divergence_weight']),
            'mode_idx': int(solution_stats['mode_idx']),
            'major': [_ints(cn[f'major_{m}']) for m in range(num_clones)],
            'minor': [_ints(cn[f'minor_{m}']) for m in range(num_clones)],
        }

        brk_cn = solutions_brk_cn[solution_id]
        if len(breakpoint_ids) > 0:
            brk_cn = brk_cn.drop_duplicates('prediction_id').set_index('prediction_id').reindex(breakpoint_ids)
            solution['brk_cn'] = [_ints(brk_cn[f'cn_{m}']) for m in range(1, num_clones)]
        else:
            solution['brk_cn'] = [[] for m in range(1, num_clones)]

        solutions.append(solution)

    sample = {
        'num_clones': num_clones,
        'optimal_id': optimal_id,
        'optimal_mix': _rounded(optimal_mix, 6),
        'max_prop_diverge': max_prop_diverge,
        'num_failed_fits': int(len(failed_fits)),
        'segments': {
            'chromosome': list(segments['chromosome'].values),
            'start': _ints(segments['start']),
            'end': _ints(segments['end']),
            'length': _rounded(segments['length'], 1),
            'readcount': _ints(segments['readcount']),
            'allele_readcount': _ints(segments['major_readcount'] + segments['minor_readcount']),
            'major_depth': _rounded(segments['major_depth'], 6),
            'minor_depth': _rounded(segments['minor_depth'], 6),
            'total_depth': _rounded(segments['total_depth'], 6),
            'allele_readcount_a': _ints(segments['allele_a_readcount']),
            'allele_readcount_b': _ints(segments['allele_b_readcount']),
            'num_hap_blocks': _ints(segments['num_hap_blocks']),
            'total_mask': _ints(segments['total_likelihood_mask']),
            'allele_mask': _ints(segments['allele_likelihood_mask']),
            'mask_reason': _ints(segment_mask_reasons(segments)),
        },
        'max_depth': max_depth,
        'breakpoints': {
            'prediction_id': [str(a) for a in breakpoint_ids],
        },
        'solutions': solutions,
        'read_depth': read_depth_densities(read_depth),
        'minor_modes': _rounded(minor_modes.values, 6),
    }

    for col in ('chromosome_1', 'chromosome_2', 'strand_1', 'strand_2', 'type'):
        if col in breakpoints:
            sample['breakpoints'][col] = [str(a) for a in breakpoints[col].values]
    for col in ('position_1', 'position_2'):
        if col in breakpoints:
            sample['breakpoints'][col] = _ints(breakpoints[col])

    return sample


def chromosome_info(samples):
    """ Chromosome order and lengths, from the largest segment end over all samples.
    """
    lengths = {}
    for sample in samples.values():
        segments = sample['segments']
        for chromosome, end in zip(segments['chromosome'], segments['end']):
            lengths[chromosome] = max(lengths.get(chromosome, 0), end)
        breakpoints = sample['breakpoints']
        for side in ('1', '2'):
            for chromosome, position in zip(
                    breakpoints.get('chromosome_' + side, []), breakpoints.get('position_' + side, [])):
                if chromosome in lengths:
                    lengths[chromosome] = max(lengths[chromosome], position)

    chromosomes = sorted(lengths, key=chromosome_sort_key)
    return {
        'names': chromosomes,
        'lengths': [int(lengths[c]) for c in chromosomes],
    }


def _encode(data):
    """ Gzipped, base64 encoded JSON.
    """
    raw = json.dumps(data, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return base64.b64encode(gzip.compress(raw, compresslevel=9)).decode('ascii')


def create_report(results_filenames, html_filename, title='ReMixT results'):
    """ Create a standalone HTML report of remixt results.

    Args:
        results_filenames (dict): results HDF5 filename keyed by tumour id
        html_filename (str): output HTML filename

    KwArgs:
        title (str): title of the report
    """
    samples = {}
    for sample_id, results_filename in results_filenames.items():
        logging.info(f'loading results of {sample_id} from {results_filename}')
        samples[str(sample_id)] = load_sample(
            results_filename)

    report = {
        'title': title,
        'version': getattr(remixt, '__version__', ''),
        'chromosomes': chromosome_info(samples),
        'sample_ids': list(samples.keys()),
    }

    payload = {
        'report': report,
        'samples': {sample_id: _encode(sample) for sample_id, sample in samples.items()},
    }

    template = importlib.resources.files('remixt').joinpath(TEMPLATE_NAME).read_text(encoding='utf-8')

    # JSON in a script tag must not contain a closing script tag
    payload_json = json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')

    page = (
        template
        .replace('{{TITLE}}', html.escape(title))
        .replace('{{PAYLOAD}}', payload_json))

    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(page)

    logging.info(f'wrote report of {len(samples)} samples to {html_filename}')
