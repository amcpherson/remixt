#!/usr/bin/env python
"""Infer SNP genotype from a normal sample for a single chromosome."""

import click
import yaml
import remixt.analysis.haplotype


@click.command()
@click.option('--seqdata', required=True, help='Input normal seqdata HDF5 file')
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--config', required=True, help='YAML config file')
@click.option('--output', required=True, help='Output SNP genotype TSV file')
def main(seqdata, chromosome, config, output):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    remixt.analysis.haplotype.infer_snp_genotype_from_normal(
        output, seqdata, chromosome, cfg,
    )


if __name__ == '__main__':
    main()
