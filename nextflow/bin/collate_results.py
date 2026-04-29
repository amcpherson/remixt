#!/usr/bin/env python
"""Collate fit results from multiple initializations.

Accepts fit result files as init_id:path pairs.
"""

import click
import yaml
import remixt.config
import remixt.analysis.pipeline


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--init_results', required=True, help='Input init results HDF5 file')
@click.option('--config', required=True, help='YAML config file')
@click.option('--tumour_id', default=None, help='Tumour sample id for sample-specific config')
@click.option('--output', required=True, help='Output collated HDF5 file')
@click.argument('fit_results_args', nargs=-1, required=True)
def main(experiment, init_results, config, tumour_id, output, fit_results_args):
    """FIT_RESULTS_ARGS: init_id:path pairs for fit result files"""
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    cfg = remixt.config.get_sample_config(cfg, tumour_id)

    # Parse init_id:path pairs
    fit_results_filenames = {}
    for arg in fit_results_args:
        init_id, path = arg.split(':', 1)
        fit_results_filenames[int(init_id)] = path

    remixt.analysis.pipeline.collate(
        output, experiment, init_results, fit_results_filenames, cfg,
    )


if __name__ == '__main__':
    main()
