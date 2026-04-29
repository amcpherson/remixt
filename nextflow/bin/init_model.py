#!/usr/bin/env python
"""Initialize model fitting parameters.

Outputs:
  - init_results HDF5 file (read depth, minor modes)
  - init_params JSON file (dict of init_id -> parameter dict)
"""

import json
import click
import yaml
import remixt.config
import remixt.analysis.pipeline


@click.command()
@click.option('--experiment', required=True, help='Input experiment pickle file')
@click.option('--config', required=True, help='YAML config file')
@click.option('--tumour_id', default=None, help='Tumour sample id for sample-specific config')
@click.option('--output_results', required=True, help='Output init results HDF5 file')
@click.option('--output_params', required=True, help='Output init params JSON file')
def main(experiment, config, tumour_id, output_results, output_params):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    # Apply sample-specific config overrides
    cfg = remixt.config.get_sample_config(cfg, tumour_id)

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
