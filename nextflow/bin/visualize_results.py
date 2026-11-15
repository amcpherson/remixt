#!/usr/bin/env python
"""Create a standalone HTML report of the results of all tumours.

Accepts results files as tumour_id:path pairs.
"""

import logging

import click
import remixt.visualize


@click.command()
@click.option('--output', required=True, help='Output HTML report')
@click.option('--title', default='ReMixT results', show_default=True, help='Title of the report')
@click.argument('results_args', nargs=-1, required=True)
def main(output, title, results_args):
    """RESULTS_ARGS: tumour_id:path pairs for results HDF5 files"""
    logging.basicConfig(level=logging.INFO)

    results_filenames = {}
    for arg in results_args:
        tumour_id, path = arg.split(':', 1)
        results_filenames[tumour_id] = path

    remixt.visualize.create_report(
        results_filenames, output, title=title)


if __name__ == '__main__':
    main()
