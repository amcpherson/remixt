#!/usr/bin/env nextflow

/*
 * Processes for read count preparation.
 * Mirrors create_prepare_counts_workflow from pypeliner.
 */


process segment_readcount {
    label 'mem_high'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(seqdata)
    path segments

    output:
    tuple val(tumour_id), path("${tumour_id}.segment_counts.tsv")

    script:
    def args = task.ext.args ?: ''
    """
    segment_readcount.py \
        --segments ${segments} \
        --seqdata ${seqdata} \
        --output ${tumour_id}.segment_counts.tsv \
        ${args}
    """
}


process haplotype_allele_readcount {
    label 'mem_high'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(seqdata)
    path segments
    path haplotypes

    output:
    tuple val(tumour_id), path("${tumour_id}.allele_counts.tsv")

    script:
    def args = task.ext.args ?: ''
    """
    haplotype_allele_readcount.py \
        --segments ${segments} \
        --seqdata ${seqdata} \
        --haplotypes ${haplotypes} \
        --output ${tumour_id}.allele_counts.tsv \
        ${args}
    """
}


/*
 * Phase allele counts across all tumour samples.
 * This is a merge/split step: takes all tumours as input, outputs per-tumour phased files.
 */
process phase_segments {
    label 'mem_medium'

    input:
    val tumour_allele_args   // list of "tumour_id:path" strings

    output:
    path "phased.*.tsv"

    script:
    def args_str = tumour_allele_args.join(' ')
    """
    phase_segments.py \
        --output_prefix phased \
        ${args_str}
    """
}


process prepare_readcount_table {
    label 'mem_medium'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(segment_counts), path(phased_allele_counts)

    output:
    tuple val(tumour_id), path("${tumour_id}.counts.tsv")

    script:
    """
    prepare_readcount_table.py \
        --segment_counts ${segment_counts} \
        --allele_counts ${phased_allele_counts} \
        --output ${tumour_id}.counts.tsv
    """
}
