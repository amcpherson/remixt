#!/usr/bin/env python
"""Merge multiple TSV tables into a single file.

Reused for merge_haps, merge_biases, and any other table merge step.
"""

import click
import remixt.utils


@click.command()
@click.option('--output', required=True, help='Output merged TSV file')
@click.argument('input_files', nargs=-1, required=True)
def main(output, input_files):
    remixt.utils.merge_tables(output, *input_files)


if __name__ == '__main__':
    main()
