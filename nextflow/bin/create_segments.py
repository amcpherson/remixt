#!/usr/bin/env python
"""Create genomic segments from reference data and breakpoints."""

import click
from genome_version import genome_version_config
import remixt.analysis.segment


@click.command()
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--breakpoints', default=None, help='Input breakpoints TSV file')
@click.option('--output', required=True, help='Output segments TSV file')
@click.option('--segment_length', type=int, default=500000, help='Segment length')
@click.option('--chromosomes', default=None, help='Comma-separated chromosome list')
@click.option('--ensembl_genome_version', default='GRCh38', help='Genome version (GRCh38 or GRCh37)')
@click.option('--chr_name_prefix', default='', help='Chromosome name prefix')
def main(ref_data_dir, breakpoints, output, segment_length, chromosomes, chr_name_prefix, ensembl_genome_version):
    cfg = {
        'segment_length': segment_length,
    }
    if chromosomes:
        cfg['chromosomes'] = chromosomes.split(',')
    if chr_name_prefix:
        cfg['chr_name_prefix'] = chr_name_prefix
    cfg.update(genome_version_config(ensembl_genome_version))

    remixt.analysis.segment.create_segments(
        output, cfg, ref_data_dir,
        breakpoint_filename=breakpoints,
    )


if __name__ == '__main__':
    main()
