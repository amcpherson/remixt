#!/usr/bin/env python
"""Calculate fragment length statistics from seqdata.

Outputs a JSON file with fragment_mean and fragment_stddev.
"""

import json
import click
import remixt.analysis.stats


@click.command()
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--output', required=True, help='Output JSON file with fragment stats')
@click.option('--filter_duplicates/--no_filter_duplicates', default=False, help='Filter duplicate reads')
@click.option('--map_qual_threshold', type=int, default=1, help='Mapping quality threshold')
def main(seqdata, output, filter_duplicates, map_qual_threshold):
    cfg = {
        'filter_duplicates': filter_duplicates,
        'map_qual_threshold': map_qual_threshold,
    }

    fragstats = remixt.analysis.stats.calculate_fragment_stats(seqdata, cfg)

    with open(output, 'w') as f:
        json.dump({
            'fragment_mean': fragstats.fragment_mean,
            'fragment_stddev': fragstats.fragment_stddev,
        }, f)


if __name__ == '__main__':
    main()
