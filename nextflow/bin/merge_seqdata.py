#!/usr/bin/env python
"""Merge per-chromosome seqdata HDF5 files into a single file."""

import click
import glob
import remixt.seqdataio


@click.command()
@click.option('--output', required=True, help='Output merged seqdata HDF5 file')
@click.argument('input_files', nargs=-1, required=True)
def main(output, input_files):
    # merge_seqdata expects a dict; keys are arbitrary (chromosome names not needed)
    in_filenames = {str(i): f for i, f in enumerate(input_files)}
    remixt.seqdataio.merge_seqdata(output, in_filenames)


if __name__ == '__main__':
    main()
