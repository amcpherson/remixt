#!/usr/bin/env nextflow

/*
 * Standalone HTML report of the results of all tumours.
 */

include { remixt_args; id_path_args } from './utils'


process visualize_results {
    label 'mem_medium'

    input:
    tuple val(tumour_ids), path(results)

    output:
    path "remixt_report.html"

    script:
    def args_str = id_path_args(tumour_ids, results)
    """
    visualize_results.py \
        --output remixt_report.html \
        ${remixt_args('visualize_results')} \
        ${args_str}
    """
}
