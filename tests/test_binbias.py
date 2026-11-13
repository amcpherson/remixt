import numpy as np
import pandas as pd
import pytest

import remixt.analysis.binbias as bb


def test_make_bins():
    bins = bb.make_bins({'1': 25, '2': 10}, 10)
    assert bins['chromosome'].tolist() == ['1', '1', '1', '2']
    assert bins['start'].tolist() == [0, 10, 20, 0]
    assert bins['end'].tolist() == [10, 20, 25, 10]


def test_segment_expected_uniform():
    # Density 1 per base, fragments of length 4 lose 2 bases at each end of a segment
    bins = bb.make_bins({'1': 100}, 10)
    expected = np.full(len(bins), 10.)
    segments = pd.DataFrame({'chromosome': ['1', '1'], 'start': [0, 35], 'end': [30, 100]})
    result = bb.segment_expected(segments, bins, expected, fragment_mean=4)
    np.testing.assert_allclose(result, [26., 61.])


def test_segment_expected_mask():
    # Bin [10, 20) is half masked, its 5 expected midpoints lie in [15, 20)
    bins = bb.make_bins({'1': 40}, 10)
    bins['masked_frac'] = [0., .5, 0., 0.]
    expected = np.array([10., 5., 10., 10.])
    mask = pd.DataFrame({'chromosome': ['1'], 'start': [10], 'end': [15]})
    segments = pd.DataFrame({'chromosome': ['1', '1'], 'start': [0, 12], 'end': [12, 40]})
    result = bb.segment_expected(segments, bins, expected, fragment_mean=0, mask=mask)
    np.testing.assert_allclose(result, [10., 25.])


def test_annotate_bins(tmp_path):
    seq = 'GGGGAAAANN' + 'CCCCCCCCCC'
    fasta = tmp_path / 'g.fa'
    fasta.write_text('>1\n' + seq + '\n')
    mappability = tmp_path / 'm.h5'
    with pd.HDFStore(mappability, 'w') as store:
        store.put('chromosome_1', pd.DataFrame({'start': [0, 12], 'end': [5, 20], 'quality': [60, 0]}),
                  format='table', data_columns=True)
    bins = bb.make_bins({'1': 20}, 10)
    annotated = bb.annotate_bins(bins, str(fasta), str(mappability), map_qual_threshold=1)
    np.testing.assert_allclose(annotated['gc'], [4 / 8, 1.])
    np.testing.assert_allclose(annotated['map_frac'], [.5, 0.])


def test_build_panel_recovers_factor():
    rng = np.random.default_rng(1)
    n = 4000
    bins = bb.make_bins({'1': n * 1000}, 1000)
    bins['gc'] = rng.uniform(.35, .6, n)
    bins['map_frac'] = np.where(np.arange(n) % 10 == 0, .5, 1.)
    true_factor = np.where(bins['map_frac'] < 1, np.log(.7), 0.)
    # Bins 105-111 are hypervariable, straddling two 10 kb panel bins
    hyper = (np.arange(n) >= 105) & (np.arange(n) < 112)
    counts = []
    for i in range(6):
        gc_effect = -2. * (i + 1) / 6 * (bins['gc'].values - .45)
        noise = np.where(hyper, rng.normal(0, 1., n), 0.)
        counts.append(rng.poisson(2000 * np.exp(gc_effect + true_factor + noise)))
    counts = np.array(counts)
    panel, mask = bb.build_panel(bins, counts, [True] * 6, panel_bin_size=1000, mask_excess_cv=.3)
    assert set(mask["start"] // 1000) <= set(range(105, 112))
    assert len(mask) >= 6
    ok = panel['masked_frac'] == 0
    resid = panel['factor'][ok] - true_factor[ok]
    assert np.abs(np.median(resid)) < .01
    assert np.abs(resid).max() < .1

    # At a coarser panel resolution, partly masked bins keep a factor per unmasked base
    panel, mask = bb.build_panel(bins, counts, [True] * 6, panel_bin_size=10000,
                                 mask_bin_size=1000, mask_excess_cv=.3)
    partial = (panel['masked_frac'] > 0) & (panel['masked_frac'] < 1)
    assert partial.any()
    assert np.abs(panel.loc[partial, 'factor']).max() < .05

def test_in_mask():
    mask = pd.DataFrame({'chromosome': ['1', '1', '2'], 'start': [30, 10, 0], 'end': [40, 20, 100]})
    positions = np.array([5, 10, 19, 20, 29, 30, 39, 40])
    np.testing.assert_array_equal(bb.in_mask(positions, mask, '1'),
                                  [False, True, True, False, False, True, True, False])
    assert not bb.in_mask(positions, mask, '3').any()


def _fake_fragments(monkeypatch, fragments):
    """ Serve fragments, a dict of chromosome to (start, end) arrays, in place of a seqdata file. """
    import remixt.seqdataio

    def read_fragment_data(seqdata_filename, chromosome, **kwargs):
        start, end = fragments[chromosome]
        return pd.DataFrame({'fragment_id': np.arange(len(start)), 'start': start, 'end': end})

    monkeypatch.setattr(remixt.seqdataio, 'read_fragment_data', read_fragment_data)
    monkeypatch.setattr(remixt.seqdataio, 'read_chromosomes', lambda f: list(fragments))


def test_mask_applied_alike_to_segment_and_bin_counts(monkeypatch):
    import remixt.analysis.segment

    # Fragments of length 10 starting at 1, 3, ..., 181, so none start on a segment boundary
    start = np.arange(1, 182, 2)
    _fake_fragments(monkeypatch, {'1': (start, start + 10)})
    mask = pd.DataFrame({'chromosome': ['1'], 'start': [50], 'end': [70]})
    segments = pd.DataFrame({'chromosome': ['1', '1'], 'start': [0, 100], 'end': [100, 200]})
    bins = bb.make_bins({'1': 200}, 100)

    unmasked = remixt.analysis.segment.create_segment_counts(segments.copy(), 'unused')['readcount'].values
    masked = remixt.analysis.segment.create_segment_counts(segments.copy(), 'unused', mask=mask)['readcount'].values
    # Segment [0, 100) contains the 45 fragments starting 1-89, of which the 10 starting 45-63
    # have midpoints in [50, 70)
    np.testing.assert_array_equal(unmasked, [45, 41])
    np.testing.assert_array_equal(masked, [35, 41])

    binned = bb.count_bins('unused', bins, mask=mask)
    # Midpoints 6-98 (47 fragments) in bin [0, 100), less the same 10, and 100-186 (44) in [100, 200)
    np.testing.assert_array_equal(binned, [37, 44])
