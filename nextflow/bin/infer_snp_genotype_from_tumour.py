#!/usr/bin/env python
"""Infer SNP genotype from tumour samples for a single chromosome.

Accepts multiple tumour seqdata files as positional arguments in the form
  tumour_id:path tumour_id:path ...
"""

import click
import remixt.analysis.haplotype


@click.command()
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--output', required=True, help='Output SNP genotype TSV file')
@click.option('--sequencing_base_call_error', type=float, default=0.01, help='Base call error rate')
@click.option('--homozygous_p_value_threshold', type=float, default=1e-16, help='Homozygous p-value threshold')
@click.argument('tumour_seqdata_args', nargs=-1, required=True)
def main(chromosome, output, sequencing_base_call_error, homozygous_p_value_threshold, tumour_seqdata_args):
    """TUMOUR_SEQDATA_ARGS: tumour_id:path pairs"""
    cfg = {
        'sequencing_base_call_error': sequencing_base_call_error,
        'homozygous_p_value_threshold': homozygous_p_value_threshold,
    }

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
