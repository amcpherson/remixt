import numpy as np
import pandas as pd
import pytest

import remixt.utils
import remixt.analysis.experiment


HEADER = '\t'.join(remixt.utils.breakpoint_columns) + '\n'


def _count_data(num_segments=10):
    l = 100000
    data = pd.DataFrame({
        'chromosome': '1',
        'start': np.arange(num_segments) * l,
        'end': (np.arange(num_segments) + 1) * l,
        'length': l,
        'readcount': 1000,
        'major_readcount': 30,
        'minor_readcount': 20,
        'major_is_allele_a': 1,
        'allele_a_readcount': 30,
        'allele_b_readcount': 20,
        'num_hap_blocks': 1,
    })
    return data


@pytest.mark.parametrize('contents', [None, '', HEADER])
def test_read_breakpoints_empty(tmp_path, contents):
    filename = None
    if contents is not None:
        filename = tmp_path / 'breakpoints.tsv'
        filename.write_text(contents)

    breakpoints = remixt.utils.read_breakpoints(filename)

    assert len(breakpoints) == 0
    assert list(breakpoints.columns) == remixt.utils.breakpoint_columns


def test_read_breakpoints(tmp_path):
    filename = tmp_path / 'breakpoints.tsv'
    filename.write_text(HEADER + '7\tX\t+\t100\tX\t-\t200\n')

    breakpoints = remixt.utils.read_breakpoints(filename)

    assert breakpoints['chromosome_1'].tolist() == ['X']
    assert breakpoints['position_2'].tolist() == [200]


def test_read_breakpoints_missing_columns(tmp_path):
    filename = tmp_path / 'breakpoints.tsv'
    filename.write_text('prediction_id\tchromosome_1\n')

    with pytest.raises(ValueError, match='missing columns'):
        remixt.utils.read_breakpoints(filename)


def test_experiment_no_breakpoints():
    breakpoint_data = remixt.utils.read_breakpoints(None)
    experiment = remixt.analysis.experiment.Experiment(_count_data(), breakpoint_data)

    assert experiment.breakpoints == {}

    brk_cn_table = remixt.analysis.experiment.create_brk_cn_table(
        {}, experiment.breakpoint_segment_data, 3)

    assert len(brk_cn_table) == 0
    for column in ['prediction_id', 'cn_0', 'cn_1', 'cn_2', 'n_1', 'side_1', 'chromosome_1', 'position_2']:
        assert column in brk_cn_table.columns


def test_model_no_breakpoints():
    cn_model = pytest.importorskip('remixt.cn_model')

    experiment = remixt.analysis.experiment.Experiment(_count_data(), None)
    model = cn_model.BreakpointModel(
        experiment.x, experiment.l, experiment.adjacencies, experiment.breakpoints,
        max_depth=1.)

    assert model.num_breakpoints == 0
    assert model.N1 == model.N
    assert np.all(model.breakpoint_idx == -1)
