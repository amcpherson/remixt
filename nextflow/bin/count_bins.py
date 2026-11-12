#!/usr/bin/env python
"""Count fragment midpoints per bin."""

import click
import pandas as pd
import remixt.analysis.binbias
import remixt.defaults


@click.command()
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--bins', required=True, help='Input bins TSV file, columns chromosome, start, end')
@click.option('--output', required=True, help='Output TSV file, the bins with a readcount column')
@click.option('--filter_duplicates', type=bool, default=remixt.defaults.filter_duplicates, show_default=True)
@click.option('--map_qual_threshold', type=int, default=remixt.defaults.map_qual_threshold, show_default=True)
def main(seqdata, bins, output, **kwargs):
    bins = pd.read_csv(bins, sep='\t', converters={'chromosome': str}, usecols=['chromosome', 'start', 'end'])
    bins['readcount'] = remixt.analysis.binbias.count_bins(seqdata, bins, **kwargs)
    bins.to_csv(output, sep='\t', index=False)


if __name__ == '__main__':
    main()
