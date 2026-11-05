#!/usr/bin/env python
"""Merge runs of mappable positions into a remixt mappability HDF5 file."""

import click
import remixt.mappability


@click.command()
@click.option('--output', required=True, help='Output mappability HDF5 file')
@click.argument('runs_filenames', nargs=-1, required=True)
def main(output, runs_filenames):
    """RUNS_FILENAMES: runs TSV files in genome order"""
    remixt.mappability.merge_mappability(runs_filenames, output)


if __name__ == '__main__':
    main()
