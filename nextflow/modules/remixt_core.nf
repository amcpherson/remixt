#!/usr/bin/env nextflow

/*
 * Core processes: segments, experiment creation, ploidy plots.
 */

include { remixt_args } from './utils'


process create_segments {
    label 'mem_medium'

    input:
    path genome_fai
    path gap_table
    path breakpoints

    output:
    path "segments.tsv"

    script:
    """
    create_segments.py \
        --genome_fai ${genome_fai} \
        --gap_table ${gap_table} \
        --breakpoints ${breakpoints} \
        --output segments.tsv \
        ${remixt_args('create_segments')}
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

    output:
    tuple val(tumour_id), path("${tumour_id}.ploidy_plots.pdf")

    script:
    """
    ploidy_analysis_plots.py \
        --experiment ${experiment} \
        --output ${tumour_id}.ploidy_plots.pdf \
        ${remixt_args('ploidy_analysis_plots')}
    """
}
