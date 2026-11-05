#!/usr/bin/env python
"""Combine segment and allele counts into a single count table."""

import click
import remixt.analysis.readcount


@click.command()
@click.option('--segment_counts', required=True, help='Input segment counts TSV file')
@click.option('--allele_counts', required=True, help='Input phased allele counts TSV file')
@click.option('--output', required=True, help='Output combined count TSV file')
def main(segment_counts, allele_counts, output):
    remixt.analysis.readcount.prepare_readcount_table(
        segment_counts, allele_counts, output,
    )


if __name__ == '__main__':
    main()
