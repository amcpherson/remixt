#!/usr/bin/env python
"""Infer SNP genotype from a normal sample for a single chromosome."""

import click
import remixt.analysis.haplotype


@click.command()
@click.option('--seqdata', required=True, help='Input normal seqdata HDF5 file')
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--output', required=True, help='Output SNP genotype TSV file')
@click.option('--sequencing_base_call_error', type=float, default=0.01, help='Base call error rate')
@click.option('--het_snp_call_threshold', type=float, default=0.9, help='Het SNP call threshold')
def main(seqdata, chromosome, output, sequencing_base_call_error, het_snp_call_threshold):
    cfg = {
        'sequencing_base_call_error': sequencing_base_call_error,
        'het_snp_call_threshold': het_snp_call_threshold,
    }

    remixt.analysis.haplotype.infer_snp_genotype_from_normal(
        output, seqdata, chromosome, cfg,
    )


if __name__ == '__main__':
    main()
