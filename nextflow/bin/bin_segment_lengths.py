#!/usr/bin/env python
"""Bias corrected segment lengths with the bin level depth model.

With --panel and --mask, depth is GC corrected per bin with the panel's depth factor and
masked regions are removed from counts and lengths. Otherwise --bins gives annotated bins
and depth is fitted as a smooth function of GC and mappable fraction.
"""

import json
import click
import pandas as pd
import remixt.analysis.binbias
import remixt.defaults


@click.command()
@click.option('--counts', required=True, help='Input segment counts TSV file')
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--fragstats', required=True, help='Input fragment stats JSON file')
@click.option('--output', required=True, help='Output segment lengths TSV file')
@click.option('--output_bins', required=True, help='Output per bin counts and expected counts')
@click.option('--bins', default=None, help='Annotated bins TSV file, without a panel')
@click.option('--panel', default=None, help='Depth panel TSV file')
@click.option('--mask', default=None, help='Depth panel mask TSV file')
@click.option('--filter_duplicates', type=bool, default=remixt.defaults.filter_duplicates, show_default=True)
@click.option('--map_qual_threshold', type=int, default=remixt.defaults.map_qual_threshold, show_default=True)
def main(counts, seqdata, fragstats, output, output_bins, bins, panel, mask, **kwargs):
    with open(fragstats) as f:
        fragment_mean = json.load(f)['fragment_mean']

    segments = pd.read_csv(counts, sep='\t', converters={'chromosome': str})

    if panel is not None:
        method = 'panel'
        panel = remixt.analysis.binbias.read_panel(panel)
        mask = remixt.analysis.binbias.read_mask(mask) if mask is not None else None
    elif bins is not None:
        method = 'curve'
        bins = pd.read_csv(bins, sep='\t', converters={'chromosome': str})
    else:
        raise click.UsageError('Provide --panel or --bins')

    segments, bin_data = remixt.analysis.binbias.bin_segment_lengths(
        segments, seqdata, fragment_mean, method,
        bins=bins, panel=panel, mask=mask, **kwargs,
    )

    segments.to_csv(output, sep='\t', index=False)
    bin_data.to_csv(output_bins, sep='\t', index=False)


if __name__ == '__main__':
    main()
