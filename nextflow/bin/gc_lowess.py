#!/usr/bin/env python
"""Fit LOWESS regression to GC content vs read count."""

import click
import remixt.analysis.gcbias


@click.command()
@click.option('--gc_samples', required=True, help='Input GC samples TSV file')
@click.option('--output_dist', required=True, help='Output GC distribution TSV file')
@click.option('--output_table', required=True, help='Output GC table TSV file')
def main(gc_samples, output_dist, output_table):
    remixt.analysis.gcbias.gc_lowess(
        gc_samples, output_dist, output_table,
    )


if __name__ == '__main__':
    main()
