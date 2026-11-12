""" Bin level GC and mappability correction of read depth.

Read depth is modelled per genomic bin, from the count of fragment midpoints in the bin:

    log E[x_b] = s + f(gc_b) + m_b

f is a smooth function of the bin's reference GC, fitted per sample by robust regression. The
mappability term m_b is either

  - 'panel': a per bin factor learned from a panel of normals, the median over the normals of
    their residual log depth after their own GC correction (build_panel). The panel also
    provides a mask of hypervariable bins, which are dropped from read counts and lengths.
  - 'curve': a smooth function of the bin's 100-mer mappable fraction, fitted per sample
    together with f.

Segment lengths are the sum of bin expectations over the part of each segment where a
contained fragment can have its midpoint, scaled to the total segment length.
"""

import numpy as np
import pandas as pd
import patsy

import remixt.segalg
import remixt.seqdataio
import remixt.utils


SPLINE_UPPER = 1.001  # patsy B-spline bases vanish at exactly the upper bound
GC_SPLINE = 'bs(gc, df={df}, lower_bound=0, upper_bound={ub})'
MAP_SPLINE = 'bs(map_frac, df={df}, lower_bound=0, upper_bound={ub})'
HIGH_MAPPABILITY = 0.95


def make_bins(chromosome_lengths, bin_size):
    """ Regular bins tiling each chromosome, the last bin of each chromosome may be shorter.

    Args:
        chromosome_lengths (dict): chromosome name to length
        bin_size (int): bin length

    Returns:
        pandas.DataFrame: columns chromosome, start, end
    """
    bins = []
    for chromosome, length in chromosome_lengths.items():
        start = np.arange(0, length, bin_size)
        bins.append(pd.DataFrame({
            'chromosome': chromosome,
            'start': start,
            'end': np.minimum(start + bin_size, length),
        }))
    return pd.concat(bins, ignore_index=True)


def read_mappability_indicator(mappability_filename, chromosome, max_chromosome_length, map_qual_threshold):
    """ Indicator per position of the k-mer starting there aligning back with at least map_qual_threshold.

    Args:
        mappability_filename (str): mappability HDF5 file, see remixt.mappability
        chromosome (str): chromosome
        max_chromosome_length (int): length of the returned indicator
        map_qual_threshold (int): mapping quality threshold

    Returns:
        numpy.array: uint8 indicator per position
    """
    with pd.HDFStore(mappability_filename, 'r') as store:
        mappability_table = store.select('chromosome_' + chromosome, 'quality >= map_qual_threshold')

    mappability = np.zeros(max_chromosome_length, dtype=np.uint8)

    for start, end in mappability_table[['start', 'end']].values:
        end = min(end, max_chromosome_length)
        mappability[start:end] = 1

    return mappability


def annotate_bins(bins, genome_fasta_filename, mappability_filename, map_qual_threshold=1):
    """ GC and mappable fraction of each bin.

    Args:
        bins (pandas.DataFrame): columns chromosome, start, end, sorted by start within chromosome
        genome_fasta_filename (str): reference genome fasta
        mappability_filename (str): mappability HDF5 file

    KwArgs:
        map_qual_threshold (int): k-mers aligning with at least this quality are mappable

    Returns:
        pandas.DataFrame: bins with additional columns gc (GC of non N bases, nan if all N) and
            map_frac (fraction of positions whose k-mer is mappable)
    """
    bins = bins.copy()
    bins['gc'] = np.nan
    bins['map_frac'] = np.nan

    chromosomes = set(bins['chromosome'])
    for chromosome, sequence in remixt.utils.read_sequences(genome_fasta_filename):
        if chromosome not in chromosomes:
            continue

        idx = np.where(bins['chromosome'].values == chromosome)[0]
        start = bins['start'].values[idx]
        end = np.minimum(bins['end'].values[idx], len(sequence))

        seq = np.frombuffer(sequence.upper().encode(), dtype='S1')
        gc_cumsum = np.concatenate([[0], np.cumsum((seq == b'G') | (seq == b'C'), dtype=np.int64)])
        base_cumsum = np.concatenate([[0], np.cumsum(seq != b'N', dtype=np.int64)])
        del seq

        num_bases = base_cumsum[end] - base_cumsum[start]
        bins.loc[bins.index[idx], 'gc'] = np.where(
            num_bases > 0, (gc_cumsum[end] - gc_cumsum[start]) / np.maximum(num_bases, 1), np.nan)
        del gc_cumsum, base_cumsum

        mappable = read_mappability_indicator(mappability_filename, chromosome, len(sequence), map_qual_threshold)
        map_cumsum = np.concatenate([[0], np.cumsum(mappable, dtype=np.int64)])
        del mappable
        bins.loc[bins.index[idx], 'map_frac'] = (map_cumsum[end] - map_cumsum[start]) / (end - start)

    return bins


def read_fragment_midpoints(seqdata_filename, chromosome, filter_duplicates=False, map_qual_threshold=1):
    """ Fragment start, end and midpoint for one chromosome, sorted by start. """
    reads = remixt.seqdataio.read_fragment_data(
        seqdata_filename, chromosome,
        filter_duplicates=filter_duplicates,
        map_qual_threshold=map_qual_threshold,
    )
    reads = reads.sort_values('start')
    reads['midpoint'] = (reads['start'] + reads['end']) // 2
    return reads


def count_bins(seqdata_filename, bins, filter_duplicates=False, map_qual_threshold=1):
    """ Count fragment midpoints in each bin.

    Args:
        seqdata_filename (str): seqdata file
        bins (pandas.DataFrame): columns chromosome, start, end, non overlapping

    KwArgs:
        filter_duplicates (bool): filter fragments marked as duplicate
        map_qual_threshold (int): filter fragments with less than this mapping quality

    Returns:
        numpy.array: count per bin, in the order of bins
    """
    counts = np.zeros(len(bins), dtype=np.int64)
    seqdata_chromosomes = set(remixt.seqdataio.read_chromosomes(seqdata_filename))
    for chromosome in bins['chromosome'].unique():
        if chromosome not in seqdata_chromosomes:
            continue
        idx = np.where(bins['chromosome'].values == chromosome)[0]
        order = np.argsort(bins['start'].values[idx])
        idx = idx[order]
        start = bins['start'].values[idx]
        end = bins['end'].values[idx]
        reads = read_fragment_midpoints(seqdata_filename, chromosome, filter_duplicates, map_qual_threshold)
        midpoint = reads['midpoint'].values
        b = np.searchsorted(start, midpoint, side='right') - 1
        inside = (b >= 0) & (midpoint < end[np.maximum(b, 0)])
        counts[idx] += np.bincount(b[inside], minlength=len(idx))
    return counts


def huber_regression(X, y, t=1.345, max_iter=50, tol=1e-8):
    """ Huber M-estimate of linear regression coefficients by iteratively reweighted least squares.

    The scale is re-estimated each iteration as the normalised median absolute deviation of the
    residuals, as statsmodels RLM with HuberT does by default, with a fraction of its memory.

    Args:
        X (numpy.array): n x p design
        y (numpy.array): n responses

    KwArgs:
        t (float): Huber threshold in units of scale

    Returns:
        numpy.array: p coefficients
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        resid = y - X @ beta
        scale = np.median(np.abs(resid - np.median(resid))) / 0.6745
        if scale <= 0:
            break
        u = np.abs(resid) / scale
        sqrt_w = np.sqrt(np.where(u <= t, 1., t / np.maximum(u, 1e-12)))
        beta_new = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
        converged = np.max(np.abs(beta_new - beta)) <= tol * (1. + np.max(np.abs(beta)))
        beta = beta_new
        if converged:
            break
    return beta


def _robust_fit(formula, data, y):
    """ Huber regression of y on a patsy design, returns a predictor on new data. """
    X = patsy.dmatrix(formula, data)
    params = huber_regression(np.asarray(X), y)
    info = X.design_info

    def predict(new_data):
        Xn = patsy.build_design_matrices([info], new_data)[0]
        return np.asarray(Xn) @ params

    return predict


def _fittable(bins, readcount):
    return (
        bins['gc'].notnull().values &
        (bins['map_frac'].values > 0) &
        (readcount > 0)
    )


def fit_gc_curve(bins, readcount, fit_bins=None, offset=None, gc_df=8, gc_column='gc'):
    """ Fit log readcount ~ s + f(gc) and predict for all bins.

    Args:
        bins (pandas.DataFrame): annotated bins, columns gc, map_frac
        readcount (numpy.array): count per bin

    KwArgs:
        fit_bins (numpy.array): boolean, bins to fit on, default bins with map_frac >= 0.95
        offset (numpy.array): log offset per bin, subtracted before fitting and added to the prediction
        gc_df (int): spline degrees of freedom
        gc_column (str): bins column holding GC

    Returns:
        numpy.array: log expected count per bin, nan where GC is unknown
    """
    if offset is None:
        offset = np.zeros(len(bins))
    if fit_bins is None:
        fit_bins = bins['map_frac'].values >= HIGH_MAPPABILITY
    data = bins[[gc_column]].rename(columns={gc_column: 'gc'})
    fit = _fittable(bins, readcount) & fit_bins & np.isfinite(offset) & data['gc'].notnull().values
    y = np.log(readcount[fit]) - offset[fit]
    predict = _robust_fit(GC_SPLINE.format(df=gc_df, ub=SPLINE_UPPER), data[fit], y)
    log_expected = np.full(len(bins), np.nan)
    ok = data['gc'].notnull().values
    log_expected[ok] = predict(data[ok]) + offset[ok]
    return log_expected


def fit_gc_map_curve(bins, readcount, fit_bins=None, gc_df=8, map_df=6):
    """ Fit log readcount ~ s + f(gc) + g(map_frac) and predict for all bins.

    Returns:
        numpy.array: log expected count per bin, nan where GC is unknown or map_frac is 0
    """
    fit = _fittable(bins, readcount)
    if fit_bins is not None:
        fit &= fit_bins
    formula = GC_SPLINE.format(df=gc_df, ub=SPLINE_UPPER) + ' + ' + MAP_SPLINE.format(df=map_df, ub=SPLINE_UPPER)
    predict = _robust_fit(formula, bins[fit], np.log(readcount[fit]))
    log_expected = np.full(len(bins), np.nan)
    ok = bins['gc'].notnull().values & (bins['map_frac'].values > 0)
    log_expected[ok] = predict(bins[ok])
    return log_expected


def gc_residuals(bins, readcount, pseudocount=0.5, **kwargs):
    """ Residual log depth after a GC curve fitted on highly mappable bins. """
    log_expected = fit_gc_curve(bins, readcount, **kwargs)
    return np.log(readcount + pseudocount) - log_expected, log_expected


def aggregate_bins(bins, bin_size, values=None, sum_columns=None):
    """ Map regular bins onto coarser regular bins of bin_size (a multiple of the bin size).

    Returns:
        numpy.array: index of the coarse bin of each bin
        pandas.DataFrame: coarse bins, columns chromosome, start, end
    """
    key = bins['chromosome'].astype(str) + ':' + (bins['start'] // bin_size).astype(str)
    coarse_index, uniques = pd.factorize(key)
    coarse = bins.groupby(coarse_index).agg(chromosome=('chromosome', 'first'), start=('start', 'min'), end=('end', 'max'))
    return coarse_index, coarse.reset_index(drop=True)


def build_panel(
        bins,
        readcounts,
        is_female,
        panel_bin_size,
        mask_bin_size=None,
        mask_excess_cv=0.3,
        mask_min_normals=3,
        gc_df=8,
):
    """ Build a panel of normals depth correction.

    Args:
        bins (pandas.DataFrame): annotated regular bins (annotate_bins) at the base resolution,
            columns chromosome, start, end, gc, map_frac
        readcounts (numpy.array): normals x bins fragment midpoint counts
        is_female (list of bool): per normal, male normals have their X depth doubled
        panel_bin_size (int): resolution of the factor, a multiple of the base bin size

    KwArgs:
        mask_bin_size (int): resolution of the hypervariable mask, a multiple of the base bin
            size, default the base bin size
        mask_excess_cv (float): mask bins whose spread across normals, beyond Poisson, exceeds
            this coefficient of variation
        mask_min_normals (int): need at least this many normals to estimate the spread
        gc_df (int): spline degrees of freedom of the GC curves

    Returns:
        pandas.DataFrame: panel bins, columns chromosome, start, end, gc, map_frac, factor
            (median residual log depth over normals), factor_spread (robust sd over normals),
            num_normals (normals with data), masked_frac (fraction of the bin masked)
        pandas.DataFrame: mask bins, columns chromosome, start, end, excess_cv, num_normals,
            including only masked bins
    """
    readcounts = np.asarray(readcounts, dtype=float)
    num_normals = readcounts.shape[0]
    base_size = int((bins['end'] - bins['start']).max())
    if mask_bin_size is None:
        mask_bin_size = base_size

    # Male normals: double X so all normals have the same expected X depth as autosomes
    readcounts = readcounts.copy()
    is_x = bins['chromosome'].isin(['X', 'chrX']).values
    for i, female in enumerate(is_female):
        if not female:
            readcounts[i, is_x] *= 2.

    def residuals_at(bin_size):
        """ Per normal GC residuals at a coarser resolution. """
        index, coarse = aggregate_bins(bins, bin_size)
        coarse = annotate_coarse(bins, index, coarse)
        counts = np.vstack([np.bincount(index, weights=r, minlength=len(coarse)) for r in readcounts])
        resid = np.full(counts.shape, np.nan)
        expected = np.full(counts.shape, np.nan)
        for i in range(num_normals):
            resid[i], log_expected = gc_residuals(coarse, counts[i], gc_df=gc_df)
            expected[i] = np.exp(log_expected)
        resid[:, ~(coarse['map_frac'].values > 0)] = np.nan
        return index, coarse, counts, resid, expected

    # Hypervariable mask at the mask resolution
    mask_index, mask_bins, mask_counts, mask_resid, mask_expected = residuals_at(mask_bin_size)
    # Robust spread over normals of residual log depth, centred per bin by the median
    centred = mask_resid - np.nanmedian(mask_resid, axis=0)
    spread = 1.4826 * np.nanmedian(np.abs(centred), axis=0)
    # Poisson part of the spread, for counts of exp(median residual) * expected
    expected_counts = np.exp(np.nanmedian(mask_resid, axis=0)) * np.nanmedian(mask_expected, axis=0)
    poisson_var = 1. / np.maximum(expected_counts, 1e-3)
    excess_cv = np.sqrt(np.maximum(spread ** 2 - poisson_var, 0.))
    num_with_data = np.isfinite(mask_resid).sum(axis=0)
    mask_bins['excess_cv'] = excess_cv
    mask_bins['num_normals'] = num_with_data
    mask_bins['masked'] = (num_with_data >= mask_min_normals) & (excess_cv > mask_excess_cv)

    # Factor at the panel resolution, from base bins not masked
    base_masked = mask_bins['masked'].values[mask_index]
    unmasked_counts = readcounts * ~base_masked
    panel_index, panel = aggregate_bins(bins, panel_bin_size)
    panel_annot = bins.assign(base_len=(bins['end'] - bins['start']) * ~base_masked)
    panel = annotate_coarse(bins[~base_masked], panel_index[~base_masked], panel)
    unmasked_len = np.bincount(panel_index, weights=panel_annot['base_len'], minlength=len(panel))
    panel['masked_frac'] = 1. - unmasked_len / (panel['end'] - panel['start']).values
    counts = np.vstack([np.bincount(panel_index, weights=r, minlength=len(panel)) for r in unmasked_counts])
    resid = np.full(counts.shape, np.nan)
    for i in range(num_normals):
        resid[i], _ = gc_residuals(panel, counts[i], gc_df=gc_df)
    usable = (panel['map_frac'].values > 0) & (panel['masked_frac'].values < 1)
    resid[:, ~usable] = np.nan
    panel['factor'] = np.nanmedian(resid, axis=0) if num_normals > 0 else np.nan
    panel['factor_spread'] = 1.4826 * np.nanmedian(np.abs(resid - panel['factor'].values), axis=0)
    panel['num_normals'] = np.isfinite(resid).sum(axis=0)

    mask = mask_bins.loc[mask_bins['masked'], ['chromosome', 'start', 'end', 'excess_cv', 'num_normals']]
    return panel, mask.reset_index(drop=True)


def annotate_coarse(bins, index, coarse):
    """ GC and mappable fraction of coarse bins from base bins (bins and index may be a subset). """
    coarse = coarse.copy()
    length = (bins['end'] - bins['start']).values.astype(float)
    gc = bins['gc'].values
    has_gc = np.isfinite(gc)
    # Weight GC by base bin length; base bins with all N have no GC
    gc_num = np.bincount(index[has_gc], weights=gc[has_gc] * length[has_gc], minlength=len(coarse))
    gc_den = np.bincount(index[has_gc], weights=length[has_gc], minlength=len(coarse))
    map_num = np.bincount(index, weights=bins['map_frac'].values * length, minlength=len(coarse))
    map_den = np.bincount(index, weights=length, minlength=len(coarse))
    coarse['gc'] = np.where(gc_den > 0, gc_num / np.maximum(gc_den, 1), np.nan)
    coarse['map_frac'] = np.where(map_den > 0, map_num / np.maximum(map_den, 1), 0.)
    return coarse


def read_panel(panel_filename):
    return pd.read_csv(panel_filename, sep='\t', converters={'chromosome': str})


def read_mask(mask_filename):
    return pd.read_csv(mask_filename, sep='\t', converters={'chromosome': str})


def fit_sample_bins(bins, readcount, method, panel=None, gc_df=8, map_df=6):
    """ Expected count per bin for one sample.

    Args:
        bins (pandas.DataFrame): bins, columns chromosome, start, end, gc, map_frac; for the panel
            method these are the panel bins, with columns factor and masked_frac
        readcount (numpy.array): fragment midpoint count per bin, masked regions excluded
        method (str): 'panel' or 'curve'

    Returns:
        numpy.array: expected count per bin, 0 where there is no information (unknown GC, unmappable,
            or fully masked)
    """
    if method == 'panel':
        factor = bins['factor'].values
        # Expected count scales with the unmasked part of the bin
        unmasked = 1. - bins['masked_frac'].values
        offset = factor + np.log(np.where(unmasked > 0, unmasked, np.nan))
        fit_bins = np.isfinite(offset)
        log_expected = fit_gc_curve(bins, readcount, fit_bins=fit_bins, offset=offset, gc_df=gc_df)
    elif method == 'curve':
        log_expected = fit_gc_map_curve(bins, readcount, gc_df=gc_df, map_df=map_df)
    else:
        raise ValueError('unknown method {}'.format(method))
    expected = np.exp(log_expected)
    expected[~np.isfinite(expected)] = 0.
    return expected


def _density_cumsum(starts, ends, values, lengths, mask_starts, mask_ends):
    """ Piecewise linear cumulative sum of a density, uniform within each interval and zero in the mask.

    values[i] is spread over lengths[i], the unmasked length of interval i.

    Returns positions and cumulative values at those positions, for np.interp.
    """
    pos = np.unique(np.concatenate([starts, ends, mask_starts, mask_ends]))
    # Density on each elementary interval [pos[i], pos[i+1])
    lengths = np.maximum(np.asarray(lengths, dtype=float), 1e-9)
    density = np.zeros(len(pos) - 1)
    k = np.searchsorted(starts, pos[:-1], side='right') - 1
    inside = (k >= 0) & (pos[:-1] < ends[np.maximum(k, 0)])
    density[inside] = values[k[inside]] / lengths[k[inside]]
    if len(mask_starts) > 0:
        m = np.searchsorted(mask_starts, pos[:-1], side='right') - 1
        in_mask = (m >= 0) & (pos[:-1] < mask_ends[np.maximum(m, 0)])
        density[in_mask] = 0.
    cumulative = np.concatenate([[0.], np.cumsum(density * np.diff(pos))])
    return pos, cumulative


def segment_expected(segments, bins, expected, fragment_mean, mask=None):
    """ Expected count of fragments contained in each segment.

    A fragment of length f is contained in segment [s, e) if its midpoint is in
    [s + f/2, e - f/2), approximated with the mean fragment length.

    Args:
        segments (pandas.DataFrame): columns chromosome, start, end
        bins (pandas.DataFrame): columns chromosome, start, end, non overlapping
        expected (numpy.array): expected fragment midpoint count per bin, excluding masked regions,
            spread uniformly over the unmasked part of the bin (column masked_frac if present)
        fragment_mean (float): mean fragment length

    KwArgs:
        mask (pandas.DataFrame): masked regions, columns chromosome, start, end, non overlapping

    Returns:
        numpy.array: expected count per segment
    """
    result = np.zeros(len(segments))
    half = fragment_mean / 2.
    for chromosome in segments['chromosome'].unique():
        seg_idx = np.where(segments['chromosome'].values == chromosome)[0]
        b = bins[bins['chromosome'] == chromosome]
        order = np.argsort(b['start'].values)
        b_start = b['start'].values[order]
        b_end = b['end'].values[order]
        chrom_idx = np.where(bins['chromosome'].values == chromosome)[0][order]
        b_exp = np.asarray(expected)[chrom_idx]
        b_len = (b_end - b_start).astype(float)
        if mask is not None and 'masked_frac' in bins:
            b_len = b_len * (1. - bins['masked_frac'].values[chrom_idx])
        if mask is not None:
            m = mask[mask['chromosome'] == chromosome].sort_values('start')
            m_start, m_end = m['start'].values, m['end'].values
        else:
            m_start = m_end = np.array([], dtype=np.int64)
        if len(b_start) == 0:
            continue
        pos, cumulative = _density_cumsum(b_start, b_end, b_exp, b_len, m_start, m_end)
        s = segments['start'].values[seg_idx] + half
        e = segments['end'].values[seg_idx] - half
        e = np.maximum(e, s)
        result[seg_idx] = np.interp(e, pos, cumulative) - np.interp(s, pos, cumulative)
    return result


def masked_contained_counts(seqdata_filename, segments, mask, filter_duplicates=False, map_qual_threshold=1):
    """ Count fragments contained in each segment whose midpoint is masked.

    Subtract from the segment readcount so that counts and lengths both exclude the mask.
    """
    counts = np.zeros(len(segments), dtype=np.int64)
    for chromosome in segments['chromosome'].unique():
        m = mask[mask['chromosome'] == chromosome].sort_values('start')
        if len(m) == 0:
            continue
        seg_idx = np.where(segments['chromosome'].values == chromosome)[0]
        reads = read_fragment_midpoints(seqdata_filename, chromosome, filter_duplicates, map_qual_threshold)
        k = np.searchsorted(m['start'].values, reads['midpoint'].values, side='right') - 1
        in_mask = (k >= 0) & (reads['midpoint'].values < m['end'].values[np.maximum(k, 0)])
        reads = reads[in_mask]
        segs = segments.iloc[seg_idx]
        order = np.argsort(segs['start'].values)
        c = remixt.segalg.contained_counts(
            segs[['start', 'end']].values[order],
            reads[['start', 'end']].values,
        )
        counts[seg_idx[order]] = c
    return counts


def bin_segment_lengths(
        segments,
        seqdata_filename,
        fragment_mean,
        method,
        bins=None,
        panel=None,
        mask=None,
        filter_duplicates=False,
        map_qual_threshold=1,
        gc_df=8,
        map_df=6,
):
    """ Bias corrected segment lengths with the bin level model.

    Args:
        segments (pandas.DataFrame): segment counts, columns chromosome, start, end, readcount
        seqdata_filename (str): tumour seqdata
        fragment_mean (float): mean fragment length of the tumour
        method (str): 'panel' or 'curve'

    KwArgs:
        bins (pandas.DataFrame): annotated bins for the curve method
        panel (pandas.DataFrame): panel bins for the panel method
        mask (pandas.DataFrame): masked regions for the panel method

    Returns:
        pandas.DataFrame: segments with readcount reduced by masked fragments, and columns
            bias (expected count) and length (bias scaled to the total segment length)
        pandas.DataFrame: bins with readcount and expected columns, for diagnostics
    """
    segments = segments.copy()
    if method == 'panel':
        bins = panel
    bins = bins.reset_index(drop=True)

    readcount = count_bins(seqdata_filename, bins, filter_duplicates, map_qual_threshold).astype(float)
    if method == 'panel' and mask is not None and len(mask) > 0:
        # Remove masked fragment midpoints from the bin counts
        masked_bins = mask[['chromosome', 'start', 'end']].reset_index(drop=True)
        masked = count_bins(seqdata_filename, masked_bins, filter_duplicates, map_qual_threshold)
        masked_bins['readcount'] = masked
        # Each mask bin lies within one panel bin when the panel resolution is a multiple of the mask resolution
        key = pd.MultiIndex.from_frame(bins[['chromosome', 'start']])
        panel_bin_size = int((bins['end'] - bins['start']).max())
        mask_key = pd.MultiIndex.from_arrays([masked_bins['chromosome'], masked_bins['start'] // panel_bin_size * panel_bin_size])
        pos = key.get_indexer(mask_key)
        assert (pos >= 0).all()
        np.subtract.at(readcount, pos, masked)

    expected = fit_sample_bins(bins, readcount, method, gc_df=gc_df, map_df=map_df)
    bins = bins.assign(readcount=readcount, expected=expected)

    segments['bias'] = segment_expected(segments, bins, expected, fragment_mean,
                                        mask=mask if method == 'panel' else None)
    if method == 'panel' and mask is not None and len(mask) > 0:
        segments['readcount'] -= masked_contained_counts(
            seqdata_filename, segments, mask, filter_duplicates, map_qual_threshold)

    segments = calculate_biased_length(segments)
    return segments, bins


def calculate_biased_length(segments):
    """ Scale segment biases (expected counts) to lengths summing to the total segment length. """
    segments = segments.copy()
    segments['bias'] = segments['bias'] / segments['bias'].sum()
    segments['length'] = segments['bias'] * float((segments['end'] - segments['start']).sum())
    return segments
