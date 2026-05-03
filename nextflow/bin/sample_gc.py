#!/usr/bin/env python
"""Sample GC content and read counts at random genomic positions."""

import click
import remixt.analysis.gcbias


@click.command()
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--fragment_mean', required=True, type=float, help='Mean fragment length')
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--output', required=True, help='Output GC samples TSV file')
@click.option('--sample_gc_num_positions', type=int, default=10000000, help='Number of positions to sample')
@click.option('--gc_position_offset', type=int, default=4, help='GC position offset')
@click.option('--filter_duplicates/--no_filter_duplicates', default=False, help='Filter duplicate reads')
@click.option('--map_qual_threshold', type=int, default=1, help='Mapping quality threshold')
@click.option('--chromosomes', default=None, help='Comma-separated chromosome list')
@click.option('--chr_name_prefix', default='', help='Chromosome name prefix')
def main(seqdata, fragment_mean, ref_data_dir, output,
         sample_gc_num_positions, gc_position_offset,
         filter_duplicates, map_qual_threshold,
         chromosomes, chr_name_prefix):
    cfg = {
        'sample_gc_num_positions': sample_gc_num_positions,
        'gc_position_offset': gc_position_offset,
        'filter_duplicates': filter_duplicates,
        'map_qual_threshold': map_qual_threshold,
    }
    if chromosomes:
        cfg['chromosomes'] = chromosomes.split(',')
    if chr_name_prefix:
        cfg['chr_name_prefix'] = chr_name_prefix

    remixt.analysis.gcbias.sample_gc(
        output, seqdata, fragment_mean, cfg, ref_data_dir,
    )


if __name__ == '__main__':
    main()
