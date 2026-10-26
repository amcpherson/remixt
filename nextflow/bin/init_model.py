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
@click.option('--config', type=click.File(), required=True, help='remixt config JSON file')
def main(experiment, output_results, output_params, config):
    cfg = json.load(config)

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
