#!/usr/bin/env nextflow

/*
 * Processes for calculating genome mappability with bwa.
 */

include { remixt_args } from './utils'


/*
 * Build the bwa index of the genome, used if the fasta is not already indexed.
 */
process bwa_index {
    label 'mem_medium'
    tag "${genome_fasta.name}"

    input:
    path genome_fasta

    output:
    path "${genome_fasta}.{amb,ann,bwt,pac,sa}"

    script:
    """
    bwa index ${genome_fasta}
    """
}


/*
 * Align the k-mers starting in a region of the genome back to the genome,
 * and calculate runs of positions whose k-mer aligns back to that position.
 */
process mappability_region {
    label 'mem_medium'
    cpus 8
    tag "${chromosome}:${start}-${end}"

    input:
    tuple val(chromosome), val(start), val(end)
    path genome_fasta
    path genome_fai
    path bwa_index

    output:
    tuple val(chromosome), val(start), path("${chromosome}.${start}.mappability.tsv")

    script:
    """
    set -o pipefail
    mappability_kmers.py \
        --genome_fasta ${genome_fasta} \
        --chromosome ${chromosome} \
        --start ${start} \
        --end ${end} \
        ${remixt_args('mappability_kmers')} \
    | bwa mem -M -t ${task.cpus} ${genome_fasta} /dev/stdin \
    | mappability_runs.py \
        --output ${chromosome}.${start}.mappability.tsv
    """
}


/*
 * Merge the per region runs, in genome order, into the mappability HDF5 file.
 */
process merge_mappability {
    label 'mem_medium'

    input:
    path runs
    val output_name

    output:
    path output_name

    script:
    def runs_list = runs instanceof List ? runs : [runs]
    """
    merge_mappability.py \
        --output ${output_name} \
        ${runs_list.join(' ')}
    """
}
