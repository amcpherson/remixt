#!/usr/bin/env python
"""Create genomic segments from reference data and breakpoints."""

import click
import yaml
import remixt.analysis.segment


@click.command()
@click.option('--config', required=True, help='YAML config file')
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--breakpoints', default=None, help='Input breakpoints TSV file')
@click.option('--output', required=True, help='Output segments TSV file')
def main(config, ref_data_dir, breakpoints, output):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    remixt.analysis.segment.create_segments(
        output, cfg, ref_data_dir,
        breakpoint_filename=breakpoints,
    )


if __name__ == '__main__':
    main()
