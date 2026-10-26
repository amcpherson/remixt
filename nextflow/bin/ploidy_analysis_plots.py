#!/usr/bin/env python
"""Generate ploidy analysis PDF plots from an experiment."""

import json
import click
import remixt.config
import remixt.cn_plot


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--output', required=True, help='Output PDF file')
@click.option('--config', type=click.File(), required=True, help='remixt config JSON file')
def main(experiment, output, config):
    cfg = json.load(config)

    remixt.cn_plot.ploidy_analysis_plots(
        experiment, output, chromosomes=remixt.config.get_param(cfg, 'chromosomes'),
    )


if __name__ == '__main__':
    main()
