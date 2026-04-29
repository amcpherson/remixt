#!/usr/bin/env python
"""Resolve chromosome list from reference data and config."""

import click
import yaml
import remixt.config


@click.command()
@click.option('--config', required=True, help='YAML config file')
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--output', required=True, help='Output file with one chromosome per line')
def main(config, ref_data_dir, output):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    chromosomes = remixt.config.get_chromosomes(cfg, ref_data_dir)

    with open(output, 'w') as f:
        for chrom in chromosomes:
            f.write(chrom + '\n')


if __name__ == '__main__':
    main()
