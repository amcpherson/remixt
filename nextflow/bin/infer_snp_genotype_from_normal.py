#!/usr/bin/env python
"""Infer SNP genotype from a normal sample for a single chromosome."""

import json
import click
import remixt.analysis.haplotype


@click.command()
@click.option('--seqdata', required=True, help='Input normal seqdata HDF5 file')
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--output', required=True, help='Output SNP genotype TSV file')
@click.option('--config', type=click.File(), required=True, help='remixt config JSON file')
def main(seqdata, chromosome, output, config):
    cfg = json.load(config)

    remixt.analysis.haplotype.infer_snp_genotype_from_normal(
        output, seqdata, chromosome, cfg,
    )


if __name__ == '__main__':
    main()
