#!/usr/bin/env python
"""Count reads per segment from seqdata."""

import click
import remixt.analysis.readcount
import remixt.defaults


@click.command()
@click.option('--segments', required=True, help='Input segment TSV file')
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--output', required=True, help='Output segment counts TSV file')
@click.option('--mask', default=None, help='Depth panel mask TSV file, fragments with a masked midpoint are not counted')
@click.option('--filter_duplicates', type=bool, default=remixt.defaults.filter_duplicates, show_default=True)
@click.option('--map_qual_threshold', type=int, default=remixt.defaults.map_qual_threshold, show_default=True)
def main(segments, seqdata, output, mask, **kwargs):
    remixt.analysis.readcount.segment_readcount(
        output, segments, seqdata, mask_filename=mask, **kwargs,
    )


if __name__ == '__main__':
    main()
