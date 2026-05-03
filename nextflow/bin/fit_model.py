#!/usr/bin/env python
"""Fit the model for a single initialization."""

import json
import click
import remixt.analysis.pipeline


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--init_params', required=True, help='Input init params JSON file (single init)')
@click.option('--output', required=True, help='Output fit results pickle file')
@click.option('--normal_contamination/--no_normal_contamination', default=True, help='Model normal contamination')
@click.option('--max_copy_number', type=int, default=12, help='Maximum copy number')
@click.option('--likelihood_min_segment_length', type=int, default=10000, help='Min segment length for likelihood')
@click.option('--likelihood_min_proportion_genotyped', type=float, default=0.01, help='Min proportion genotyped')
@click.option('--num_em_iter', type=int, default=5, help='Number of EM iterations')
@click.option('--num_update_iter', type=int, default=5, help='Number of update iterations per EM')
@click.option('--disable_breakpoints/--no_disable_breakpoints', default=False, help='Disable breakpoints')
@click.option('--is_female/--is_male', default=True, help='Sample is female')
@click.option('--do_h_update/--no_h_update', default=True, help='Update h parameter')
def main(experiment, init_params, output,
         normal_contamination, max_copy_number,
         likelihood_min_segment_length, likelihood_min_proportion_genotyped,
         num_em_iter, num_update_iter, disable_breakpoints,
         is_female, do_h_update):
    cfg = {
        'normal_contamination': normal_contamination,
        'max_copy_number': max_copy_number,
        'likelihood_min_segment_length': likelihood_min_segment_length,
        'likelihood_min_proportion_genotyped': likelihood_min_proportion_genotyped,
        'num_em_iter': num_em_iter,
        'num_update_iter': num_update_iter,
        'disable_breakpoints': disable_breakpoints,
        'is_female': is_female,
        'do_h_update': do_h_update,
    }

    with open(init_params) as f:
        params = json.load(f)

    remixt.analysis.pipeline.fit_task(
        output, experiment, params, cfg,
    )


if __name__ == '__main__':
    main()
