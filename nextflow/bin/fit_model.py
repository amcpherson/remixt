#!/usr/bin/env python
"""Fit the model for a single initialization."""

import json
import click
import yaml
import remixt.config
import remixt.analysis.pipeline


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--init_params', required=True, help='Input init params JSON file (single init)')
@click.option('--config', required=True, help='YAML config file')
@click.option('--tumour_id', default=None, help='Tumour sample id for sample-specific config')
@click.option('--output', required=True, help='Output fit results pickle file')
def main(experiment, init_params, config, tumour_id, output):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    cfg = remixt.config.get_sample_config(cfg, tumour_id)

    with open(init_params) as f:
        params = json.load(f)

    remixt.analysis.pipeline.fit_task(
        output, experiment, params, cfg,
    )


if __name__ == '__main__':
    main()
