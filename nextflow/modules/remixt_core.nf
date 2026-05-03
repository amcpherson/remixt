#!/usr/bin/env nextflow

/*
 * Core processes: segments, experiment creation, ploidy plots.
 */


process create_segments {
    label 'mem_medium'

    input:
    val ref_data_dir
    path breakpoints

    output:
    path "segments.tsv"

    script:
    def args = task.ext.args ?: ''
    """
    create_segments.py \
        --ref_data_dir ${ref_data_dir} \
        --breakpoints ${breakpoints} \
        --output segments.tsv \
        ${args}
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
    val chromosomes_csv

    output:
    tuple val(tumour_id), path("${tumour_id}.ploidy_plots.pdf")

    script:
    def chrom_arg = chromosomes_csv ? "--chromosomes ${chromosomes_csv}" : ""
    """
    ploidy_analysis_plots.py \
        --experiment ${experiment} \
        --output ${tumour_id}.ploidy_plots.pdf \
        ${chrom_arg}
    """
}
