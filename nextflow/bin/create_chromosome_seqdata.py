#!/usr/bin/env python
"""Extract sequence data from a BAM file for a single chromosome."""

import click
import remixt.seqdataio


@click.command()
@click.option('--bam', required=True, help='Input BAM file')
@click.option('--snp_positions', required=True, help='SNP positions file')
@click.option('--chromosome', required=True, help='Chromosome to extract')
@click.option('--output', required=True, help='Output seqdata HDF5 file')
@click.option('--max_fragment_length', type=int, default=1000, help='Max fragment length')
@click.option('--max_soft_clipped', type=int, default=8, help='Max soft clipped bases')
@click.option('--check_proper_pair/--no_check_proper_pair', default=True, help='Check proper pair flag')
def main(bam, snp_positions, chromosome, output, max_fragment_length, max_soft_clipped, check_proper_pair):
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
