#!/usr/bin/env python
"""Collate fit results from multiple initializations.

Accepts fit result files as init_id:path pairs.
"""

import click
import remixt.analysis.pipeline
import remixt.defaults


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--init_results', required=True, help='Input init results HDF5 file')
@click.option('--output', required=True, help='Output collated HDF5 file')
@click.option('--max_prop_diverge', type=float, default=remixt.defaults.max_prop_diverge, show_default=True)
@click.argument('fit_results_args', nargs=-1, required=True)
def main(experiment, init_results, output, fit_results_args, **kwargs):
    """FIT_RESULTS_ARGS: init_id:path pairs for fit result files"""

    # Parse init_id:path pairs
    fit_results_filenames = {}
    for arg in fit_results_args:
        init_id, path = arg.split(':', 1)
        fit_results_filenames[int(init_id)] = path

    remixt.analysis.pipeline.collate(
        output, experiment, init_results, fit_results_filenames, **kwargs,
    )


if __name__ == '__main__':
    main()
