#!/usr/bin/env python
"""Generate ploidy analysis PDF plots from an experiment."""

import click
import yaml
import remixt.config
import remixt.cn_plot


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--output', required=True, help='Output PDF file')
@click.option('--chromosomes', default=None, help='Comma-separated chromosome list (optional)')
def main(experiment, output, chromosomes):
    chrom_list = None
    if chromosomes:
        chrom_list = chromosomes.split(',')

    remixt.cn_plot.ploidy_analysis_plots(
        experiment, output, chromosomes=chrom_list,
    )


if __name__ == '__main__':
    main()
