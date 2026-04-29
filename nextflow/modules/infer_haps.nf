#!/usr/bin/env nextflow

/*
 * Processes for haplotype inference.
 * Mirrors create_infer_haps_workflow from pypeliner.
 */


/*
 * Infer SNP genotype from a normal sample, per chromosome.
 */
process infer_snp_genotype_from_normal {
    label 'mem_medium'
    tag "${chromosome}"

    input:
    path normal_seqdata
    val chromosome
    path config_yaml

    output:
    tuple val(chromosome), path("snp_genotype.${chromosome}.tsv")

    script:
    """
    infer_snp_genotype_from_normal.py \
        --seqdata ${normal_seqdata} \
        --chromosome ${chromosome} \
        --config ${config_yaml} \
        --output snp_genotype.${chromosome}.tsv
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
    val tumour_seqdata_args   // list of "tumour_id:path" strings
    val chromosome
    path config_yaml

    output:
    tuple val(chromosome), path("snp_genotype.${chromosome}.tsv")

    script:
    def args_str = tumour_seqdata_args.join(' ')
    """
    infer_snp_genotype_from_tumour.py \
        --chromosome ${chromosome} \
        --config ${config_yaml} \
        --output snp_genotype.${chromosome}.tsv \
        ${args_str}
    """
}


/*
 * Infer haplotype phasing for a single chromosome.
 */
process infer_haps {
    label 'mem_medium'
    tag "${chromosome}"

    input:
    tuple val(chromosome), path(snp_genotype)
    path config_yaml
    val ref_data_dir

    output:
    tuple val(chromosome), path("haps.${chromosome}.tsv")

    script:
    """
    infer_haps.py \
        --snp_genotype ${snp_genotype} \
        --chromosome ${chromosome} \
        --config ${config_yaml} \
        --ref_data_dir ${ref_data_dir} \
        --output haps.${chromosome}.tsv
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
