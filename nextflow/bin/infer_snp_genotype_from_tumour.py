#!/usr/bin/env python
"""Infer SNP genotype from tumour samples for a single chromosome.

Accepts multiple tumour seqdata files as positional arguments in the form
  tumour_id:path tumour_id:path ...
"""

import click
import remixt.analysis.haplotype
import remixt.defaults


@click.command()
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--output', required=True, help='Output SNP genotype TSV file')
@click.option('--sequencing_base_call_error', type=float, default=remixt.defaults.sequencing_base_call_error, show_default=True)
@click.option('--homozygous_p_value_threshold', type=float, default=remixt.defaults.homozygous_p_value_threshold, show_default=True)
@click.argument('tumour_seqdata_args', nargs=-1, required=True)
def main(chromosome, output, tumour_seqdata_args, **kwargs):
    """TUMOUR_SEQDATA_ARGS: tumour_id:path pairs"""

    # Parse tumour_id:path pairs into a dict
    seqdata_filenames = {}
    for arg in tumour_seqdata_args:
        tumour_id, path = arg.split(':', 1)
        seqdata_filenames[tumour_id] = path

    remixt.analysis.haplotype.infer_snp_genotype_from_tumour(
        output, seqdata_filenames, chromosome, **kwargs,
    )


if __name__ == '__main__':
    main()
