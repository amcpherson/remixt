#!/usr/bin/env python
"""Fit the model for a single initialization."""

import json
import click
import remixt.analysis.pipeline
import remixt.defaults


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--init_params', required=True, help='Input init params JSON file (single init)')
@click.option('--output', required=True, help='Output fit results pickle file')
@click.option('--is_female', type=bool, default=remixt.defaults.is_female, show_default=True)
@click.option('--normal_contamination', type=bool, default=remixt.defaults.normal_contamination, show_default=True)
@click.option('--max_copy_number', type=int, default=remixt.defaults.max_copy_number, show_default=True)
@click.option('--likelihood_min_segment_length', type=int, default=remixt.defaults.likelihood_min_segment_length, show_default=True)
@click.option('--likelihood_min_proportion_genotyped', type=float, default=remixt.defaults.likelihood_min_proportion_genotyped, show_default=True)
@click.option('--num_em_iter', type=int, default=remixt.defaults.num_em_iter, show_default=True)
@click.option('--num_update_iter', type=int, default=remixt.defaults.num_update_iter, show_default=True)
@click.option('--disable_breakpoints', type=bool, default=remixt.defaults.disable_breakpoints, show_default=True)
@click.option('--do_h_update', type=bool, default=remixt.defaults.do_h_update, show_default=True)
@click.option('--random_seed', type=int, default=remixt.defaults.random_seed, show_default=True)
def main(experiment, init_params, output, **kwargs):
    with open(init_params) as f:
        params = json.load(f)

    remixt.analysis.pipeline.fit_task(
        output, experiment, params, **kwargs,
    )


if __name__ == '__main__':
    main()
