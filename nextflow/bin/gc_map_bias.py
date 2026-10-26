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
@click.option('--config', type=click.File(), required=True, help='remixt config JSON file')
def main(segments, fragstats, gc_dist, ref_data_dir, output, config):
    cfg = json.load(config)

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
