#!/usr/bin/env python
"""Count reads per segment from seqdata."""

import click
import yaml
import remixt.analysis.readcount


@click.command()
@click.option('--segments', required=True, help='Input segment TSV file')
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--config', required=True, help='YAML config file')
@click.option('--output', required=True, help='Output segment counts TSV file')
def main(segments, seqdata, config, output):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    remixt.analysis.readcount.segment_readcount(
        output, segments, seqdata, cfg,
    )


if __name__ == '__main__':
    main()
