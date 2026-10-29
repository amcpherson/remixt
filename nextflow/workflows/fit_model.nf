#!/usr/bin/env nextflow

/*
 * Sub-workflow: fit_model
 *
 * Initializes model parameters, fits in parallel across init_ids,
 * then collates results.
 *
 * Input:
 *   ch_experiments   - [tumour_id, experiment.pickle] tuples
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

    main:

    // 1. Initialize model: produces init_results + init_params JSON
    init_model(ch_experiments)
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
    fit(ch_fit_input)
    // out: [tumour_id, init_id, fit_results.pickle]

    // 4. Collate: group fit results per tumour
    ch_fit_results = fit.out
        .map { tumour_id, init_id, fit_pickle -> tuple(tumour_id, init_id as int, fit_pickle) }
        .groupTuple()
        .map { tumour_id, init_ids, fit_pickles ->
            // Sort by init_id for a stable task hash
            def pairs = [init_ids, fit_pickles].transpose().sort { pair -> pair[0] }
            tuple(tumour_id, pairs.collect { pair -> pair[0] }, pairs.collect { pair -> pair[1] })
        }
        // [tumour_id, [init_ids], [fit_results.pickle]]

    // Join with experiment and init_results
    ch_init_results = init_model.out
        .map { tumour_id, init_results, init_params -> tuple(tumour_id, init_results) }

    ch_collate_input = ch_experiments
        .join(ch_init_results)
        .join(ch_fit_results)
        // [tumour_id, experiment, init_results, [init_ids], [fit_results.pickle]]

    collate_results(ch_collate_input)
    // out: [tumour_id, results.h5]

    emit:
    results = collate_results.out   // [tumour_id, results.h5]
}
