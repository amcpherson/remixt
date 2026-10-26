#!/usr/bin/env python
"""Count reads per segment from seqdata."""

import json
import click
import remixt.analysis.readcount


@click.command()
@click.option('--segments', required=True, help='Input segment TSV file')
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--output', required=True, help='Output segment counts TSV file')
@click.option('--config', type=click.File(), required=True, help='remixt config JSON file')
def main(segments, seqdata, output, config):
    cfg = json.load(config)

    remixt.analysis.readcount.segment_readcount(
        output, segments, seqdata, cfg,
    )


if __name__ == '__main__':
    main()
