#!/usr/bin/env nextflow

/*
 * Core processes: segments, experiment creation, ploidy plots.
 */


/*
 * Write the remixt config, built from params, that is passed to each remixt step.
 */
process write_config {

    input:
    val config

    output:
    path "remixt_config.json"

    exec:
    task.workDir.resolve('remixt_config.json').text = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(config)
    )
}


process create_segments {
    label 'mem_medium'

    input:
    val ref_data_dir
    path breakpoints
    path config

    output:
    path "segments.tsv"

    script:
    """
    create_segments.py \
        --ref_data_dir ${ref_data_dir} \
        --breakpoints ${breakpoints} \
        --output segments.tsv \
        --config ${config}
    """
}


process create_experiment {
    label 'mem_medium'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(counts)
    path breakpoints

    output:
    tuple val(tumour_id), path("${tumour_id}.experiment.pickle")

    script:
    """
    create_experiment.py \
        --counts ${counts} \
        --breakpoints ${breakpoints} \
        --output ${tumour_id}.experiment.pickle
    """
}


process ploidy_analysis_plots {
    label 'mem_medium'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(experiment)
    path config

    output:
    tuple val(tumour_id), path("${tumour_id}.ploidy_plots.pdf")

    script:
    """
    ploidy_analysis_plots.py \
        --experiment ${experiment} \
        --output ${tumour_id}.ploidy_plots.pdf \
        --config ${config}
    """
}
