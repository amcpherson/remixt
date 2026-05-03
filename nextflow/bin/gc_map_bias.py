#!/usr/bin/env python
"""Calculate per-segment GC and mappability biases."""

import json
import click
import remixt.analysis.gcbias


@click.command()
@click.option('--segments', required=True, help='Input segment TSV file (chunk)')
@click.option('--fragstats', required=True, help='Input fragment stats JSON file')
@click.option('--gc_dist', required=True, help='Input GC distribution TSV file')
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--output', required=True, help='Output biases TSV file')
@click.option('--do_gc_correction/--no_gc_correction', default=True, help='Enable GC correction')
@click.option('--do_mappability_correction/--no_mappability_correction', default=True, help='Enable mappability correction')
@click.option('--gc_position_offset', type=int, default=4, help='GC position offset')
@click.option('--map_qual_threshold', type=int, default=1, help='Mapping quality threshold')
@click.option('--mappability_length', type=int, default=100, help='Mappability read length')
@click.option('--chromosomes', default=None, help='Comma-separated chromosome list')
@click.option('--chr_name_prefix', default='', help='Chromosome name prefix')
def main(segments, fragstats, gc_dist, ref_data_dir, output,
         do_gc_correction, do_mappability_correction,
         gc_position_offset, map_qual_threshold, mappability_length,
         chromosomes, chr_name_prefix):
    cfg = {
        'do_gc_correction': do_gc_correction,
        'do_mappability_correction': do_mappability_correction,
        'gc_position_offset': gc_position_offset,
        'map_qual_threshold': map_qual_threshold,
        'mappability_length': mappability_length,
    }
    if chromosomes:
        cfg['chromosomes'] = chromosomes.split(',')
    if chr_name_prefix:
        cfg['chr_name_prefix'] = chr_name_prefix

    with open(fragstats) as f:
        stats = json.load(f)

    remixt.analysis.gcbias.gc_map_bias(
        segments,
        stats['fragment_mean'],
        stats['fragment_stddev'],
        gc_dist,
        output,
        cfg,
        ref_data_dir,
    )


if __name__ == '__main__':
    main()
