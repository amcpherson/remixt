#!/usr/bin/env python
"""Convert bias scores to bias-corrected segment lengths."""

import click
import remixt.analysis.gcbias


@click.command()
@click.option('--biases', required=True, help='Input merged biases TSV file')
@click.option('--output', required=True, help='Output segment lengths TSV file')
def main(biases, output):
    remixt.analysis.gcbias.biased_length(output, biases)


if __name__ == '__main__':
    main()
