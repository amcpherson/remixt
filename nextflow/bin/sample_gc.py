#!/usr/bin/env python
"""Sample GC content and read counts at random genomic positions."""

import click
import yaml
import remixt.analysis.gcbias


@click.command()
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--fragment_mean', required=True, type=float, help='Mean fragment length')
@click.option('--config', required=True, help='YAML config file')
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--output', required=True, help='Output GC samples TSV file')
def main(seqdata, fragment_mean, config, ref_data_dir, output):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    remixt.analysis.gcbias.sample_gc(
        output, seqdata, fragment_mean, cfg, ref_data_dir,
    )


if __name__ == '__main__':
    main()
