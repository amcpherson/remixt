#!/usr/bin/env nextflow

/*
 * Processes for model fitting.
 * Mirrors create_fit_model_workflow from pypeliner.
 */


/*
 * Initialize model fitting: produces init_results + init_params JSON.
 * The init_params JSON is a dict of init_id -> param dict, which will
 * be split into individual files for parallel fitting.
 */
process init_model {
    label 'mem_medium'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(experiment)
    path config_yaml

    output:
    tuple val(tumour_id), path("${tumour_id}.init_results.h5"), path("${tumour_id}.init_params.json")

    script:
    """
    init_model.py \
        --experiment ${experiment} \
        --config ${config_yaml} \
        --tumour_id ${tumour_id} \
        --output_results ${tumour_id}.init_results.h5 \
        --output_params ${tumour_id}.init_params.json
    """
}


/*
 * Split the init_params JSON into individual per-init files.
 * This is a pure Nextflow process to enable dynamic parallelism.
 */
process split_init_params {
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(init_params_json)

    output:
    tuple val(tumour_id), path("init_param.*.json")

    script:
    """
    #!/usr/bin/env python3
    import json
    with open('${init_params_json}') as f:
        params = json.load(f)
    for init_id, p in params.items():
        with open(f'init_param.{init_id}.json', 'w') as f:
            json.dump(p, f)
    """
}


/*
 * Fit model for a single initialization.
 */
process fit_model {
    label 'mem_very_high'
    tag "${tumour_id}_${init_id}"

    input:
    tuple val(tumour_id), val(init_id), path(experiment), path(init_params)
    path config_yaml

    output:
    tuple val(tumour_id), val(init_id), path("${tumour_id}.fit.${init_id}.pickle")

    script:
    """
    fit_model.py \
        --experiment ${experiment} \
        --init_params ${init_params} \
        --config ${config_yaml} \
        --tumour_id ${tumour_id} \
        --output ${tumour_id}.fit.${init_id}.pickle
    """
}


/*
 * Collate results from all initializations.
 */
process collate_results {
    label 'mem_medium'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(experiment), path(init_results), val(fit_args)
    path config_yaml

    output:
    tuple val(tumour_id), path("${tumour_id}.results.h5")

    script:
    def args_str = fit_args.join(' ')
    """
    collate_results.py \
        --experiment ${experiment} \
        --init_results ${init_results} \
        --config ${config_yaml} \
        --tumour_id ${tumour_id} \
        --output ${tumour_id}.results.h5 \
        ${args_str}
    """
}
