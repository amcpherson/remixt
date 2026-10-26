#!/usr/bin/env python
"""Count reads per haplotype allele within segments."""

import json
import click
import remixt.analysis.readcount


@click.command()
@click.option('--segments', required=True, help='Input segment TSV file')
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--haplotypes', required=True, help='Input haplotypes TSV file')
@click.option('--output', required=True, help='Output allele counts TSV file')
@click.option('--config', type=click.File(), required=True, help='remixt config JSON file')
def main(segments, seqdata, haplotypes, output, config):
    cfg = json.load(config)

    remixt.analysis.readcount.haplotype_allele_readcount(
        output, segments, seqdata, haplotypes, cfg,
    )


if __name__ == '__main__':
    main()
