#!/usr/bin/env python
"""Calculate fragment length statistics from seqdata.

Outputs a JSON file with fragment_mean and fragment_stddev.
"""

import json
import click
import remixt.analysis.stats
import remixt.defaults


@click.command()
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--output', required=True, help='Output JSON file with fragment stats')
@click.option('--filter_duplicates', type=bool, default=remixt.defaults.filter_duplicates, show_default=True)
@click.option('--map_qual_threshold', type=int, default=remixt.defaults.map_qual_threshold, show_default=True)
def main(seqdata, output, **kwargs):
    fragstats = remixt.analysis.stats.calculate_fragment_stats(seqdata, **kwargs)

    with open(output, 'w') as f:
        json.dump({
            'fragment_mean': fragstats.fragment_mean,
            'fragment_stddev': fragstats.fragment_stddev,
        }, f)


if __name__ == '__main__':
    main()
