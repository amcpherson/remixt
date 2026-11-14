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
import remixt.defaults
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
    if 'high_quality' in read_depth:
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


def segment_mask_reasons(length, depth, total_mask, allele_mask, min_segment_length):
    """ Reason each segment is masked from the likelihood, and the inferred max depth.

    The model masks the total and allele likelihood of segments shorter than a
    minimum length or with depth above a max depth, and additionally the allele
    likelihood of segments with too few genotyped reads, see remixt.cn_model.
    The max depth is not stored in the results, so segments with total
    likelihood masked are attributed to length if shorter than the min segment
    length, and otherwise to depth.  Short segments are checked first as their
    depth is noisy and often also above the max depth.

    Args:
        length (numpy.array): effective length of each segment, as used by the model
        depth (numpy.array): total depth of each segment
        total_mask (numpy.array): total likelihood used for each segment
        allele_mask (numpy.array): allele likelihood used for each segment
        min_segment_length (float): likelihood_min_segment_length of the model fit

    Returns:
        numpy.array: MASK_* reason for each segment
        float: smallest depth masked as too high, or None if none
    """
    length = np.asarray(length, dtype=float)
    depth = np.asarray(depth, dtype=float)
    total_mask = np.asarray(total_mask, dtype=bool)
    allele_mask = np.asarray(allele_mask, dtype=bool)

    short = ~total_mask & (length < min_segment_length)
    high_depth = ~total_mask & ~short
    reasons = np.full(len(depth), MASK_NONE)
    reasons[short] = MASK_SHORT
    reasons[high_depth] = MASK_HIGH_DEPTH
    reasons[total_mask & ~allele_mask] = MASK_LOW_GENOTYPED

    finite_high_depth = depth[high_depth & np.isfinite(depth)]
    max_depth = float(finite_high_depth.min()) if len(finite_high_depth) > 0 else None

    return reasons, max_depth


def _solution_ids(store):
    """ Ids of solutions in a results store, in order of init id.
    """
    stats = store['stats']
    return sorted(stats['init_id'].astype(int).values)


def _segment_key(cn):
    return list(zip(cn['chromosome'].astype(str), cn['start'], cn['end']))


def _breakpoint_type(row):
    if row['chromosome_1'] != row['chromosome_2']:
        return 'translocation'
    if row['strand_1'] == row['strand_2']:
        return 'inversion'
    ends = sorted([(row['position_1'], row['strand_1']), (row['position_2'], row['strand_2'])])
    return 'deletion' if ends[0][1] == '+' else 'duplication'


def load_sample(results_filename, max_prop_diverge=remixt.defaults.max_prop_diverge,
                min_segment_length=remixt.defaults.likelihood_min_segment_length):
    """ Load the results of one tumour into a JSON serializable dict.

    Args:
        results_filename (str): results HDF5 from remixt collate

    KwArgs:
        max_prop_diverge (float): solutions with a larger proportion divergent
            are only selected as optimal if no solution is below it
        min_segment_length (float): likelihood_min_segment_length of the model
            fit, to attribute masked segments to length or depth

    Returns:
        dict: segments, breakpoints, solutions and read depth density
    """
    with pd.HDFStore(results_filename, 'r') as store:
        keys = set(store.keys())
        solution_ids = _solution_ids(store)
        stats = store['stats'].set_index('init_id')
        failed_fits = store['failed_fits'] if '/failed_fits' in keys else pd.DataFrame()
        read_depth = store['read_depth'] if '/read_depth' in keys else None
        minor_modes = store['minor_modes'] if '/minor_modes' in keys else None
        optimal_mix = store['mix'].values if '/mix' in keys else None

        solutions_cn = {}
        solutions_brk_cn = {}
        solutions_h = {}
        for solution_id in solution_ids:
            prefix = f'/solutions/solution_{solution_id}'
            solutions_cn[solution_id] = store[prefix + '/cn']
            solutions_brk_cn[solution_id] = store[prefix + '/brk_cn']
            solutions_h[solution_id] = store[prefix + '/h'].values

    # The optimal solution, chosen as in remixt.analysis.pipeline.store_optimal_solution
    eligible = stats[stats['proportion_divergent'] < max_prop_diverge]
    if len(eligible) == 0:
        eligible = stats
    optimal_id = int(eligible['elbo'].idxmax())

    # Segments are shared by all solutions, checked against the optimal solution
    segments = solutions_cn[optimal_id].copy()
    segments['chromosome'] = segments['chromosome'].astype(str)
    segment_key = _segment_key(segments)
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
        if _segment_key(cn) != segment_key:
            cn = (
                cn.assign(chromosome=cn['chromosome'].astype(str))
                .set_index(['chromosome', 'start', 'end'])
                .reindex(segment_key)
                .reset_index())

        h = solutions_h[solution_id]
        if len(h) != num_clones:
            logging.warning(f'skipping solution {solution_id} of {results_filename} with {len(h)} clones')
            continue

        solution_stats = stats.loc[solution_id]
        solution = {
            'id': int(solution_id),
            'h': _rounded(h, 8),
            'elbo': float(solution_stats['elbo']),
            'ploidy': float(solution_stats['ploidy']),
            'proportion_divergent': float(solution_stats['proportion_divergent']),
            'divergence_weight': float(solution_stats.get('divergence_weight', np.nan)),
            'mode_idx': int(solution_stats.get('mode_idx', -1)),
            'major': [_ints(cn[f'major_{m}'].fillna(0)) for m in range(num_clones)],
            'minor': [_ints(cn[f'minor_{m}'].fillna(0)) for m in range(num_clones)],
        }

        brk_cn = solutions_brk_cn[solution_id]
        if len(breakpoint_ids) > 0:
            brk_cn = brk_cn.drop_duplicates('prediction_id').set_index('prediction_id').reindex(breakpoint_ids)
            solution['brk_cn'] = [
                _ints(brk_cn.get(f'cn_{m}', pd.Series(0, index=brk_cn.index)).fillna(0))
                for m in range(1, num_clones)]
        else:
            solution['brk_cn'] = [[] for m in range(1, num_clones)]

        solutions.append(solution)

    def column_or(name, default):
        return segments[name] if name in segments else pd.Series(default, index=segments.index)

    total_mask = column_or('total_likelihood_mask', 1).fillna(0).astype(bool).values
    allele_mask = column_or('allele_likelihood_mask', 1).fillna(0).astype(bool).values
    mask_reason, max_depth = segment_mask_reasons(
        segments['length'].values, segments['total_depth'].values, total_mask, allele_mask,
        min_segment_length)

    sample = {
        'num_clones': num_clones,
        'optimal_id': optimal_id,
        'optimal_mix': _rounded(optimal_mix, 6) if optimal_mix is not None else None,
        'max_prop_diverge': max_prop_diverge,
        'num_failed_fits': int(len(failed_fits)),
        'segments': {
            'chromosome': list(segments['chromosome'].values),
            'start': _ints(segments['start']),
            'end': _ints(segments['end']),
            'length': _rounded(segments['length'], 1),
            'readcount': _ints(column_or('readcount', 0).fillna(0)),
            'allele_readcount': _ints(
                (column_or('major_readcount', 0) + column_or('minor_readcount', 0)).fillna(0)),
            'major_depth': _rounded(segments['major_depth'], 6),
            'minor_depth': _rounded(segments['minor_depth'], 6),
            'total_depth': _rounded(segments['total_depth'], 6),
            'total_mask': _ints(total_mask),
            'allele_mask': _ints(allele_mask),
            'mask_reason': _ints(mask_reason),
        },
        'max_depth': max_depth,
        'breakpoints': {
            'prediction_id': [str(a) for a in breakpoint_ids],
        },
        'solutions': solutions,
        'read_depth': read_depth_densities(read_depth) if read_depth is not None else None,
        'minor_modes': _rounded(minor_modes.values, 6) if minor_modes is not None else None,
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


def create_report(results_filenames, html_filename, title='ReMixT results',
                  max_prop_diverge=remixt.defaults.max_prop_diverge,
                  min_segment_length=remixt.defaults.likelihood_min_segment_length):
    """ Create a standalone HTML report of remixt results.

    Args:
        results_filenames (dict): results HDF5 filename keyed by tumour id
        html_filename (str): output HTML filename

    KwArgs:
        title (str): title of the report
        max_prop_diverge (float): used to mark solutions eligible to be optimal
        min_segment_length (float): likelihood_min_segment_length of the model fit
    """
    samples = {}
    for sample_id, results_filename in results_filenames.items():
        logging.info(f'loading results of {sample_id} from {results_filename}')
        samples[str(sample_id)] = load_sample(
            results_filename, max_prop_diverge=max_prop_diverge, min_segment_length=min_segment_length)

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
