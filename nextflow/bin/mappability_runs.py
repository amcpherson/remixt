#!/usr/bin/env python
"""Calculate runs of mappable positions from bwa alignments of k-mers read from stdin."""

import sys
import click
import remixt.mappability


@click.command()
@click.option('--output', required=True, help='Output runs TSV')
def main(output):
    mapping_quality = remixt.mappability.read_origin_mapping_quality(sys.stdin)
    runs = remixt.mappability.calculate_mappability_runs(mapping_quality)
    runs.to_csv(output, sep='\t', index=False)


if __name__ == '__main__':
    main()
