#!/usr/bin/env python
"""Infer SNP genotype from tumour samples for a single chromosome.

Accepts multiple tumour seqdata files as positional arguments in the form
  tumour_id:path tumour_id:path ...
"""

import click
import yaml
import remixt.analysis.haplotype


@click.command()
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--config', required=True, help='YAML config file')
@click.option('--output', required=True, help='Output SNP genotype TSV file')
@click.argument('tumour_seqdata_args', nargs=-1, required=True)
def main(chromosome, config, output, tumour_seqdata_args):
    """TUMOUR_SEQDATA_ARGS: tumour_id:path pairs"""
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    # Parse tumour_id:path pairs into a dict
    seqdata_filenames = {}
    for arg in tumour_seqdata_args:
        tumour_id, path = arg.split(':', 1)
        seqdata_filenames[tumour_id] = path

    remixt.analysis.haplotype.infer_snp_genotype_from_tumour(
        output, seqdata_filenames, chromosome, cfg,
    )


if __name__ == '__main__':
    main()
