#!/usr/bin/env python
"""Create genomic segments from reference data and breakpoints."""

import click
import remixt.analysis.segment
import remixt.defaults


@click.command()
@click.option('--genome_fai', required=True, help='Reference genome .fai file')
@click.option('--gap_table', required=True, help='UCSC gap table')
@click.option('--breakpoints', default=None, help='Input breakpoints TSV file')
@click.option('--output', required=True, help='Output segments TSV file')
@click.option('--chromosomes', default=','.join(remixt.defaults.chromosomes), show_default=True,
              callback=lambda ctx, param, value: value.split(','), help='Comma separated chromosomes')
@click.option('--chr_name_prefix', default=remixt.defaults.chr_name_prefix, show_default=True)
@click.option('--segment_length', type=int, default=remixt.defaults.segment_length, show_default=True)
def main(genome_fai, gap_table, breakpoints, output, **kwargs):
    remixt.analysis.segment.create_segments(
        output, genome_fai, gap_table,
        breakpoint_filename=breakpoints,
        **kwargs,
    )


if __name__ == '__main__':
    main()
