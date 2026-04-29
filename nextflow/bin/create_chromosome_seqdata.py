#!/usr/bin/env python
"""Extract sequence data from a BAM file for a single chromosome."""

import click
import yaml
import remixt.config
import remixt.seqdataio


@click.command()
@click.option('--bam', required=True, help='Input BAM file')
@click.option('--snp_positions', required=True, help='SNP positions file')
@click.option('--chromosome', required=True, help='Chromosome to extract')
@click.option('--config', required=True, help='YAML config file')
@click.option('--output', required=True, help='Output seqdata HDF5 file')
def main(bam, snp_positions, chromosome, config, output):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    max_fragment_length = remixt.config.get_param(cfg, 'bam_max_fragment_length')
    max_soft_clipped = remixt.config.get_param(cfg, 'bam_max_soft_clipped')
    check_proper_pair = remixt.config.get_param(cfg, 'bam_check_proper_pair')

    remixt.seqdataio.create_chromosome_seqdata(
        output,
        bam,
        snp_positions,
        chromosome,
        max_fragment_length,
        max_soft_clipped,
        check_proper_pair,
    )


if __name__ == '__main__':
    main()
