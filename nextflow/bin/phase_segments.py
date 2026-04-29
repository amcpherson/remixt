#!/usr/bin/env python
"""Phase allele counts across multiple tumour samples.

Accepts allele count files as tumour_id:path pairs.
Outputs phased allele count files named {tumour_id}.phased_allele_counts.tsv
"""

import click
import yaml
import remixt.analysis.readcount


@click.command()
@click.option('--output_prefix', required=True, help='Output prefix; files named {prefix}.{tumour_id}.tsv')
@click.argument('tumour_allele_args', nargs=-1, required=True)
def main(output_prefix, tumour_allele_args):
    """TUMOUR_ALLELE_ARGS: tumour_id:path pairs for allele count files"""
    allele_counts_filenames = {}
    phased_allele_counts_filenames = {}

    for arg in tumour_allele_args:
        tumour_id, path = arg.split(':', 1)
        allele_counts_filenames[tumour_id] = path
        phased_allele_counts_filenames[tumour_id] = f'{output_prefix}.{tumour_id}.tsv'

    remixt.analysis.readcount.phase_segments(
        allele_counts_filenames,
        phased_allele_counts_filenames,
    )


if __name__ == '__main__':
    main()
