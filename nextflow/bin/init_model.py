#!/usr/bin/env python
"""Initialize model fitting parameters.

Outputs:
  - init_results HDF5 file (read depth, minor modes)
  - init_params JSON file (dict of init_id -> parameter dict)
"""

import json
import click
import remixt.analysis.pipeline


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--output_results', required=True, help='Output init results HDF5 file')
@click.option('--output_params', required=True, help='Output init params JSON file')
@click.option('--min_ploidy', type=float, default=1.5, help='Minimum ploidy')
@click.option('--max_ploidy', type=float, default=6.0, help='Maximum ploidy')
@click.option('--h_normal', type=float, default=None, help='Forced haploid normal depth')
@click.option('--h_tumour', type=float, default=None, help='Forced haploid tumour depth')
@click.option('--tumour_mix_fractions', default='0.45,0.3,0.2,0.1', help='Comma-separated tumour mix fractions')
@click.option('--divergence_weights', default='1e-6,1e-7,1e-8', help='Comma-separated divergence weights')
@click.option('--max_copy_number', type=int, default=12, help='Maximum copy number')
@click.option('--random_seed', type=int, default=1234, help='Random seed')
def main(experiment, output_results, output_params,
         min_ploidy, max_ploidy, h_normal, h_tumour,
         tumour_mix_fractions, divergence_weights, max_copy_number, random_seed):
    cfg = {
        'min_ploidy': min_ploidy,
        'max_ploidy': max_ploidy,
        'h_normal': h_normal,
        'h_tumour': h_tumour,
        'tumour_mix_fractions': [float(x) for x in tumour_mix_fractions.split(',')],
        'divergence_weights': [float(x) for x in divergence_weights.split(',')],
        'max_copy_number': max_copy_number,
        'random_seed': random_seed,
    }

    init_params = remixt.analysis.pipeline.init(
        output_results, experiment, cfg,
    )

    # Serialize init_params dict (keys are ints, values are param dicts)
    # Convert keys to strings for JSON
    serializable = {str(k): v for k, v in init_params.items()}
    with open(output_params, 'w') as f:
        json.dump(serializable, f)


if __name__ == '__main__':
    main()
