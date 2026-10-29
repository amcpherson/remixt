#!/usr/bin/env python
"""Calculate per-segment GC and mappability biases."""

import json
import click
import remixt.analysis.gcbias
import remixt.defaults


@click.command()
@click.option('--segments', required=True, help='Input segment TSV file (chunk)')
@click.option('--fragstats', required=True, help='Input fragment stats JSON file')
@click.option('--gc_dist', required=True, help='Input GC distribution TSV file')
@click.option('--genome_fasta', required=True, help='Reference genome fasta file')
@click.option('--mappability', required=True, help='Mappability HDF5 file')
@click.option('--output', required=True, help='Output biases TSV file')
@click.option('--do_gc_correction', type=bool, default=remixt.defaults.do_gc_correction, show_default=True)
@click.option('--do_mappability_correction', type=bool, default=remixt.defaults.do_mappability_correction, show_default=True)
@click.option('--gc_position_offset', type=int, default=remixt.defaults.gc_position_offset, show_default=True)
@click.option('--map_qual_threshold', type=int, default=remixt.defaults.map_qual_threshold, show_default=True)
@click.option('--mappability_length', type=int, default=remixt.defaults.mappability_length, show_default=True)
def main(segments, fragstats, gc_dist, genome_fasta, mappability, output, **kwargs):
    with open(fragstats) as f:
        stats = json.load(f)

    remixt.analysis.gcbias.gc_map_bias(
        segments,
        stats['fragment_mean'],
        stats['fragment_stddev'],
        gc_dist,
        output,
        genome_fasta,
        mappability,
        **kwargs,
    )


if __name__ == '__main__':
    main()
