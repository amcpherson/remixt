#!/usr/bin/env python
"""Calculate per-segment GC and mappability biases."""

import json
import click
import yaml
import remixt.analysis.gcbias


@click.command()
@click.option('--segments', required=True, help='Input segment TSV file (chunk)')
@click.option('--fragstats', required=True, help='Input fragment stats JSON file')
@click.option('--gc_dist', required=True, help='Input GC distribution TSV file')
@click.option('--config', required=True, help='YAML config file')
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--output', required=True, help='Output biases TSV file')
def main(segments, fragstats, gc_dist, config, ref_data_dir, output):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

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
