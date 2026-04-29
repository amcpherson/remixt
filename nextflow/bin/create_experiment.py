#!/usr/bin/env python
"""Create experiment object from count data and breakpoints."""

import click
import remixt.analysis.experiment


@click.command()
@click.option('--counts', required=True, help='Input count data TSV file')
@click.option('--breakpoints', required=True, help='Input breakpoints TSV file')
@click.option('--output', required=True, help='Output experiment pickle file')
def main(counts, breakpoints, output):
    remixt.analysis.experiment.create_experiment(
        counts, breakpoints, output,
    )


if __name__ == '__main__':
    main()
