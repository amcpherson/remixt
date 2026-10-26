#!/usr/bin/env python
"""Create genomic segments from reference data and breakpoints."""

import json
import click
import remixt.analysis.segment


@click.command()
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--breakpoints', default=None, help='Input breakpoints TSV file')
@click.option('--output', required=True, help='Output segments TSV file')
@click.option('--config', type=click.File(), required=True, help='remixt config JSON file')
def main(ref_data_dir, breakpoints, output, config):
    cfg = json.load(config)

    remixt.analysis.segment.create_segments(
        output, cfg, ref_data_dir,
        breakpoint_filename=breakpoints,
    )


if __name__ == '__main__':
    main()
