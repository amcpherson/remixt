#!/usr/bin/env nextflow

/*
 * Processes for haplotype inference.
 */

include { remixt_args; id_path_args } from './utils'


/*
 * Infer SNP genotype from a normal sample, per chromosome.
 */
process infer_snp_genotype_from_normal {
    label 'mem_medium'
    tag "${chromosome}"

    input:
    path normal_seqdata
    val chromosome

    output:
    tuple val(chromosome), path("snp_genotype.${chromosome}.tsv")

    script:
    """
    infer_snp_genotype_from_normal.py \
        --seqdata ${normal_seqdata} \
        --chromosome ${chromosome} \
        --output snp_genotype.${chromosome}.tsv \
        ${remixt_args('infer_snp_genotype_from_normal')}
    """
}


/*
 * Infer SNP genotype from tumour samples, per chromosome.
 * Receives all tumour seqdata files as a collected list.
 */
process infer_snp_genotype_from_tumour {
    label 'mem_medium'
    tag "${chromosome}"

    input:
    tuple val(tumour_ids), path(tumour_seqdata)
    val chromosome

    output:
    tuple val(chromosome), path("snp_genotype.${chromosome}.tsv")

    script:
    def args_str = id_path_args(tumour_ids, tumour_seqdata)
    """
    infer_snp_genotype_from_tumour.py \
        --chromosome ${chromosome} \
        --output snp_genotype.${chromosome}.tsv \
        ${remixt_args('infer_snp_genotype_from_tumour')} \
        ${args_str}
    """
}


/*
 * Infer haplotype phasing for a single GRCh38 chromosome with SHAPEIT4.
 */
process infer_haps_shapeit4 {
    label 'mem_medium'
    tag "${chromosome}"

    input:
    tuple val(chromosome), path(snp_genotype), path(bcf_reference), path(bcf_index), path(genetic_map)
    path snp_positions

    output:
    tuple val(chromosome), path("haps.${chromosome}.tsv")

    script:
    """
    infer_haps_shapeit4.py \
        --snp_genotype ${snp_genotype} \
        --chromosome ${chromosome} \
        --snp_positions ${snp_positions} \
        --bcf_reference ${bcf_reference} \
        --genetic_map ${genetic_map} \
        --output haps.${chromosome}.tsv \
        --temp_directory "\${PWD}/shapeit_temp" \
        ${remixt_args('infer_haps_shapeit4')}
    """
}


/*
 * Infer haplotype phasing for a single GRCh37 chromosome with SHAPEIT2.
 */
process infer_haps_shapeit2 {
    label 'mem_medium'
    tag "${chromosome}"

    input:
    tuple val(chromosome), path(snp_genotype), path(genetic_map), path(haplotypes), path(legend), path(sample)

    output:
    tuple val(chromosome), path("haps.${chromosome}.tsv")

    script:
    """
    infer_haps_shapeit2.py \
        --snp_genotype ${snp_genotype} \
        --chromosome ${chromosome} \
        --genetic_map ${genetic_map} \
        --haplotypes ${haplotypes} \
        --legend ${legend} \
        --sample ${sample} \
        --output haps.${chromosome}.tsv \
        --temp_directory "\${PWD}/shapeit_temp" \
        ${remixt_args('infer_haps_shapeit2')}
    """
}


/*
 * Merge per-chromosome haplotype tables into one file.
 */
process merge_haps {
    label 'mem_medium'

    input:
    path hap_files

    output:
    path "haplotypes.tsv"

    script:
    """
    merge_tables.py --output haplotypes.tsv ${hap_files}
    """
}
