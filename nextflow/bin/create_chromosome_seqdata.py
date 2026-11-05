#!/usr/bin/env python
"""Extract sequence data from a BAM file for a single chromosome."""

import click
import remixt.defaults
import remixt.seqdataio


@click.command()
@click.option('--bam', required=True, help='Input BAM file')
@click.option('--snp_positions', required=True, help='SNP positions file')
@click.option('--chromosome', required=True, help='Chromosome to extract')
@click.option('--output', required=True, help='Output seqdata HDF5 file')
@click.option('--bam_max_fragment_length', type=int, default=remixt.defaults.bam_max_fragment_length, show_default=True)
@click.option('--bam_max_soft_clipped', type=int, default=remixt.defaults.bam_max_soft_clipped, show_default=True)
@click.option('--bam_check_proper_pair', type=bool, default=remixt.defaults.bam_check_proper_pair, show_default=True)
def main(bam, snp_positions, chromosome, output, bam_max_fragment_length, bam_max_soft_clipped, bam_check_proper_pair):
    remixt.seqdataio.create_chromosome_seqdata(
        output,
        bam,
        snp_positions,
        chromosome,
        bam_max_fragment_length,
        bam_max_soft_clipped,
        bam_check_proper_pair,
    )


if __name__ == '__main__':
    main()
