#!/usr/bin/env python
"""Count reads per haplotype allele within segments."""

import click
import yaml
import remixt.analysis.readcount


@click.command()
@click.option('--segments', required=True, help='Input segment TSV file')
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--haplotypes', required=True, help='Input haplotypes TSV file')
@click.option('--config', required=True, help='YAML config file')
@click.option('--output', required=True, help='Output allele counts TSV file')
def main(segments, seqdata, haplotypes, config, output):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    remixt.analysis.readcount.haplotype_allele_readcount(
        output, segments, seqdata, haplotypes, cfg,
    )


if __name__ == '__main__':
    main()
