#!/usr/bin/env python
"""Infer SNP genotype from a normal sample for a single chromosome."""

import click
import remixt.analysis.haplotype
import remixt.defaults


@click.command()
@click.option('--seqdata', required=True, help='Input normal seqdata HDF5 file')
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--output', required=True, help='Output SNP genotype TSV file')
@click.option('--sequencing_base_call_error', type=float, default=remixt.defaults.sequencing_base_call_error, show_default=True)
@click.option('--het_snp_call_threshold', type=float, default=remixt.defaults.het_snp_call_threshold, show_default=True)
def main(seqdata, chromosome, output, **kwargs):
    remixt.analysis.haplotype.infer_snp_genotype_from_normal(
        output, seqdata, chromosome, **kwargs,
    )


if __name__ == '__main__':
    main()
