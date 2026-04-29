#!/usr/bin/env nextflow

/*
 * Sub-workflow: fit_model
 *
 * Mirrors create_fit_model_workflow from pypeliner.
 * Initializes model parameters, fits in parallel across init_ids,
 * then collates results.
 *
 * Input:
 *   ch_experiments   - [tumour_id, experiment.pickle] tuples
 *   config_yaml      - path to config YAML
 *
 * Output:
 *   results          - [tumour_id, results.h5] tuples
 */

include { init_model }        from '../modules/fit_model'
include { split_init_params } from '../modules/fit_model'
include { fit_model as fit }  from '../modules/fit_model'
include { collate_results }   from '../modules/fit_model'


workflow fit_model {

    take:
    ch_experiments    // channel: [tumour_id, experiment.pickle]
    config_yaml       // path

    main:

    // 1. Initialize model: produces init_results + init_params JSON
    init_model(ch_experiments, config_yaml)
    // out: [tumour_id, init_results.h5, init_params.json]

    // 2. Split init_params JSON into per-init files for dynamic parallelism
    ch_init_params_input = init_model.out
        .map { tumour_id, init_results, init_params -> tuple(tumour_id, init_params) }

    split_init_params(ch_init_params_input)
    // out: [tumour_id, [init_param.0.json, init_param.1.json, ...]]

    // Flatten to [tumour_id, init_id, init_param_file]
    ch_per_init = split_init_params.out
        .flatMap { tumour_id, param_files ->
            def files = param_files instanceof List ? param_files : [param_files]
            files.collect { f ->
                def init_id = (f.name =~ /init_param\.(\d+)\.json/)[0][1]
                tuple(tumour_id, init_id, f)
            }
        }

    // Join with experiment file: [tumour_id, init_id, experiment.pickle, init_param.json]
    ch_experiments_map = ch_experiments.map { tid, exp -> tuple(tid, exp) }
    ch_fit_input = ch_per_init
        .combine(ch_experiments_map, by: 0)
        .map { tumour_id, init_id, init_params, experiment ->
            tuple(tumour_id, init_id, experiment, init_params)
        }

    // 3. Fit per initialization
    fit(ch_fit_input, config_yaml)
    // out: [tumour_id, init_id, fit_results.pickle]

    // 4. Collate: group fit results per tumour, build "init_id:path" args
    ch_fit_args = fit.out
        .map { tumour_id, init_id, fit_pickle -> tuple(tumour_id, "${init_id}:${fit_pickle}") }
        .groupTuple()
        // [tumour_id, [list of "init_id:path" strings]]

    // Join with experiment and init_results
    ch_init_results = init_model.out
        .map { tumour_id, init_results, init_params -> tuple(tumour_id, init_results) }

    ch_collate_input = ch_experiments
        .join(ch_init_results)
        .join(ch_fit_args)
        // [tumour_id, experiment, init_results, [fit_args]]

    collate_results(ch_collate_input, config_yaml)
    // out: [tumour_id, results.h5]

    emit:
    results = collate_results.out   // [tumour_id, results.h5]
}
