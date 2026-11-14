#!/usr/bin/env python
"""Create a standalone HTML report of the results of all tumours.

Accepts results files as tumour_id:path pairs.
"""

import logging

import click
import remixt.defaults
import remixt.visualize


@click.command()
@click.option('--output', required=True, help='Output HTML report')
@click.option('--title', default='ReMixT results', show_default=True, help='Title of the report')
@click.option('--max_prop_diverge', type=float, default=remixt.defaults.max_prop_diverge, show_default=True)
@click.argument('results_args', nargs=-1, required=True)
def main(output, title, max_prop_diverge, results_args):
    """RESULTS_ARGS: tumour_id:path pairs for results HDF5 files"""
    logging.basicConfig(level=logging.INFO)

    results_filenames = {}
    for arg in results_args:
        tumour_id, path = arg.split(':', 1)
        results_filenames[tumour_id] = path

    remixt.visualize.create_report(
        results_filenames, output, title=title, max_prop_diverge=max_prop_diverge)


if __name__ == '__main__':
    main()
