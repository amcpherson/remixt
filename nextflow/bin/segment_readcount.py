#!/usr/bin/env python
"""Count reads per segment from seqdata."""

import click
import remixt.analysis.readcount


@click.command()
@click.option('--segments', required=True, help='Input segment TSV file')
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--output', required=True, help='Output segment counts TSV file')
@click.option('--filter_duplicates/--no_filter_duplicates', default=False, help='Filter duplicate reads')
@click.option('--map_qual_threshold', type=int, default=1, help='Mapping quality threshold')
def main(segments, seqdata, output, filter_duplicates, map_qual_threshold):
    cfg = {
        'filter_duplicates': filter_duplicates,
        'map_qual_threshold': map_qual_threshold,
    }

    remixt.analysis.readcount.segment_readcount(
        output, segments, seqdata, cfg,
    )


if __name__ == '__main__':
    main()
