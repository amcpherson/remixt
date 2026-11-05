#!/usr/bin/env python
"""Count reads per haplotype allele within segments."""

import click
import remixt.analysis.readcount
import remixt.defaults


@click.command()
@click.option('--segments', required=True, help='Input segment TSV file')
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--haplotypes', required=True, help='Input haplotypes TSV file')
@click.option('--output', required=True, help='Output allele counts TSV file')
@click.option('--filter_duplicates', type=bool, default=remixt.defaults.filter_duplicates, show_default=True)
@click.option('--map_qual_threshold', type=int, default=remixt.defaults.map_qual_threshold, show_default=True)
def main(segments, seqdata, haplotypes, output, **kwargs):
    remixt.analysis.readcount.haplotype_allele_readcount(
        output, segments, seqdata, haplotypes, **kwargs,
    )


if __name__ == '__main__':
    main()
