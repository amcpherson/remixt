#!/usr/bin/env python
"""Extract sequence data from a BAM file for a single chromosome."""

import json
import click
import remixt.config
import remixt.seqdataio


@click.command()
@click.option('--bam', required=True, help='Input BAM file')
@click.option('--snp_positions', required=True, help='SNP positions file')
@click.option('--chromosome', required=True, help='Chromosome to extract')
@click.option('--output', required=True, help='Output seqdata HDF5 file')
@click.option('--config', type=click.File(), required=True, help='remixt config JSON file')
def main(bam, snp_positions, chromosome, output, config):
    cfg = json.load(config)

    remixt.seqdataio.create_chromosome_seqdata(
        output,
        bam,
        snp_positions,
        chromosome,
        remixt.config.get_param(cfg, 'bam_max_fragment_length'),
        remixt.config.get_param(cfg, 'bam_max_soft_clipped'),
        remixt.config.get_param(cfg, 'bam_check_proper_pair'),
    )


if __name__ == '__main__':
    main()
