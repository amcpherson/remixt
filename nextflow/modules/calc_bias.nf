#!/usr/bin/env nextflow

/*
 * Fragment length statistics for the depth bias model, see also modules/binbias.nf.
 */

include { remixt_args } from './utils'


process calc_fragment_stats {
    label 'mem_medium'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(seqdata)

    output:
    tuple val(tumour_id), path("${tumour_id}.fragstats.json")

    script:
    """
    calc_fragment_stats.py \
        --seqdata ${seqdata} \
        --output ${tumour_id}.fragstats.json \
        ${remixt_args('calc_fragment_stats')}
    """
}
