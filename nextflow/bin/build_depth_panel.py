#!/usr/bin/env python
"""Build a depth panel of normals: a per bin depth factor and a mask of hypervariable bins.

Normals are given as id:counts.tsv arguments, counts from count_bins.py on the annotated bins.
"""

import click
import numpy as np
import pandas as pd
import remixt.analysis.binbias
import remixt.defaults


@click.command()
@click.option('--bins', required=True, help='Annotated bins TSV file at the base resolution')
@click.option('--female_ids', default='', help='Comma separated ids of female normals, others are male')
@click.option('--output_panel', required=True, help='Output panel TSV file')
@click.option('--output_mask', required=True, help='Output mask TSV file')
@click.option('--panel_bin_size', type=int, default=remixt.defaults.panel_bin_size, show_default=True)
@click.option('--panel_mask_bin_size', type=int, default=remixt.defaults.panel_mask_bin_size, show_default=True)
@click.option('--panel_mask_excess_cv', type=float, default=remixt.defaults.panel_mask_excess_cv, show_default=True)
@click.argument('counts', nargs=-1, required=True)
def main(bins, female_ids, output_panel, output_mask, panel_bin_size, panel_mask_bin_size, panel_mask_excess_cv, counts):
    bins = pd.read_csv(bins, sep='\t', converters={'chromosome': str})
    female_ids = set(female_ids.split(',')) if female_ids else set()

    sample_ids = []
    readcounts = []
    for arg in sorted(counts):
        sample_id, filename = arg.split(':', 1)
        c = pd.read_csv(filename, sep='\t', converters={'chromosome': str})
        assert (c['chromosome'].values == bins['chromosome'].values).all()
        assert (c['start'].values == bins['start'].values).all()
        sample_ids.append(sample_id)
        readcounts.append(c['readcount'].values)

    panel, mask = remixt.analysis.binbias.build_panel(
        bins, np.array(readcounts), [s in female_ids for s in sample_ids],
        panel_bin_size=panel_bin_size,
        mask_bin_size=panel_mask_bin_size,
        mask_excess_cv=panel_mask_excess_cv,
    )

    panel.to_csv(output_panel, sep='\t', index=False)
    mask.to_csv(output_mask, sep='\t', index=False)
    print('normals', len(sample_ids), 'female', sum(s in female_ids for s in sample_ids))
    print('panel bins', len(panel), 'with factor', np.isfinite(panel['factor']).sum())
    print('masked bins', len(mask), 'masked bases', (mask['end'] - mask['start']).sum())


if __name__ == '__main__':
    main()
