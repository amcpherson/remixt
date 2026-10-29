#!/usr/bin/env python
"""Initialize model fitting parameters.

Outputs:
  - init_results HDF5 file (read depth, minor modes)
  - init_params JSON file (dict of init_id -> parameter dict)
"""

import json
import click
import remixt.analysis.pipeline
import remixt.defaults


def float_list(ctx, param, value):
    return [float(a) for a in value.split(',')]


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--output_results', required=True, help='Output init results HDF5 file')
@click.option('--output_params', required=True, help='Output init params JSON file')
@click.option('--min_ploidy', type=float, default=remixt.defaults.min_ploidy, show_default=True)
@click.option('--max_ploidy', type=float, default=remixt.defaults.max_ploidy, show_default=True)
@click.option('--h_normal', type=float, default=remixt.defaults.h_normal, show_default=True)
@click.option('--h_tumour', type=float, default=remixt.defaults.h_tumour, show_default=True)
@click.option('--tumour_mix_fractions', default=','.join(str(a) for a in remixt.defaults.tumour_mix_fractions),
              show_default=True, callback=float_list, help='Comma separated fractions')
@click.option('--divergence_weights', default=','.join(str(a) for a in remixt.defaults.divergence_weights),
              show_default=True, callback=float_list, help='Comma separated weights')
@click.option('--max_copy_number', type=int, default=remixt.defaults.max_copy_number, show_default=True)
@click.option('--random_seed', type=int, default=remixt.defaults.random_seed, show_default=True)
def main(experiment, output_results, output_params, **kwargs):
    init_params = remixt.analysis.pipeline.init(
        output_results, experiment, **kwargs,
    )

    # Serialize init_params dict (keys are ints, values are param dicts)
    # Convert keys to strings for JSON
    serializable = {str(k): v for k, v in init_params.items()}
    with open(output_params, 'w') as f:
        json.dump(serializable, f)


if __name__ == '__main__':
    main()
