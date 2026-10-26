#!/usr/bin/env python
"""Fit the model for a single initialization."""

import json
import click
import remixt.analysis.pipeline


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--init_params', required=True, help='Input init params JSON file (single init)')
@click.option('--output', required=True, help='Output fit results pickle file')
@click.option('--config', type=click.File(), required=True, help='remixt config JSON file')
def main(experiment, init_params, output, config):
    cfg = json.load(config)

    with open(init_params) as f:
        params = json.load(f)

    remixt.analysis.pipeline.fit_task(
        output, experiment, params, cfg,
    )


if __name__ == '__main__':
    main()
