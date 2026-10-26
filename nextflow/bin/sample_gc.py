#!/usr/bin/env python
"""Sample GC content and read counts at random genomic positions."""

import json
import click
import remixt.analysis.gcbias


@click.command()
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--fragment_mean', required=True, type=float, help='Mean fragment length')
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--output', required=True, help='Output GC samples TSV file')
@click.option('--config', type=click.File(), required=True, help='remixt config JSON file')
def main(seqdata, fragment_mean, ref_data_dir, output, config):
    cfg = json.load(config)

    remixt.analysis.gcbias.sample_gc(
        output, seqdata, fragment_mean, cfg, ref_data_dir,
    )


if __name__ == '__main__':
    main()
