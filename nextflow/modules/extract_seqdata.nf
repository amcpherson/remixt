#!/usr/bin/env nextflow

/*
 * Processes for extracting sequence data from BAM files.
 */

include { remixt_args } from './utils'


/*
 * Extract seqdata from a BAM file for a single chromosome.
 * Runs per (sample_id, chromosome) pair.
 */
process create_chromosome_seqdata {
    label 'mem_medium'
    tag "${sample_id}_${chromosome}"

    input:
    tuple val(sample_id), path(bam), path(bai)
    val chromosome
    path snp_positions

    output:
    tuple val(sample_id), val(chromosome), path("${sample_id}.${chromosome}.seqdata.h5")

    script:
    """
    create_chromosome_seqdata.py \
        --bam ${bam} \
        --snp_positions ${snp_positions} \
        --chromosome ${chromosome} \
        --output ${sample_id}.${chromosome}.seqdata.h5 \
        ${remixt_args('create_chromosome_seqdata')}
    """
}


/*
 * Merge per-chromosome seqdata files into a single sample-level file.
 * Collects all chromosome outputs for a given sample_id.
 */
process merge_seqdata {
    label 'mem_medium'
    tag "${sample_id}"

    input:
    tuple val(sample_id), path(seqdata_files)

    output:
    tuple val(sample_id), path("${sample_id}.seqdata.h5")

    script:
    """
    merge_seqdata.py \
        --output ${sample_id}.seqdata.h5 \
        ${seqdata_files}
    """
}
