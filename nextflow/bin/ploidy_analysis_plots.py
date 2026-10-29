#!/usr/bin/env python
"""Generate ploidy analysis PDF plots from an experiment."""

import click
import remixt.cn_plot
import remixt.defaults


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--output', required=True, help='Output PDF file')
@click.option('--chromosomes', default=','.join(remixt.defaults.chromosomes), show_default=True,
              callback=lambda ctx, param, value: value.split(','), help='Comma separated chromosomes')
def main(experiment, output, chromosomes):
    remixt.cn_plot.ploidy_analysis_plots(
        experiment, output, chromosomes=chromosomes,
    )


if __name__ == '__main__':
    main()
