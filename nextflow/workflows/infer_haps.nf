#!/usr/bin/env nextflow

/*
 * Sub-workflow: infer_haps
 *
 * Mirrors create_infer_haps_workflow from pypeliner.
 * Infers SNP genotypes (from normal or tumour), then phases haplotypes
 * per chromosome, then merges into a single haplotypes file.
 *
 * Input:
 *   ch_seqdata       - [sample_id, seqdata.h5] tuples
 *   config_yaml      - path to config YAML
 *   ref_data_dir     - path to reference data directory
 *   ch_chromosomes   - channel of chromosome names
 *   normal_id        - val: normal sample id or null
 *
 * Output:
 *   haplotypes       - path to merged haplotypes.tsv
 */

include { infer_snp_genotype_from_normal } from '../modules/infer_haps'
include { infer_snp_genotype_from_tumour } from '../modules/infer_haps'
include { infer_haps as infer_haps_proc } from '../modules/infer_haps'
include { merge_haps }                     from '../modules/infer_haps'


workflow infer_haps {

    take:
    ch_seqdata         // channel: [sample_id, seqdata.h5]
    config_yaml        // path
    ref_data_dir       // val
    ch_chromosomes     // channel: chromosome names
    normal_id          // val: string or 'none'

    main:

    // Collect chromosomes into a reusable list-channel
    ch_chroms = ch_chromosomes.collect().flatMap()

    if (normal_id != 'none') {
        // --- Normal-based genotyping ---
        // Extract the normal seqdata file
        ch_normal = ch_seqdata
            .filter { sample_id, seqdata -> sample_id == normal_id }
            .map { sample_id, seqdata -> seqdata }

        // Run per chromosome
        infer_snp_genotype_from_normal(
            ch_normal.first(),
            ch_chroms,
            config_yaml,
        )
        ch_snp_genotype = infer_snp_genotype_from_normal.out  // [chromosome, snp_genotype.tsv]

    } else {
        // --- Tumour-based genotyping ---
        // Collect all tumour seqdata as "tumour_id:path" strings
        ch_tumour_args = ch_seqdata
            .map { sample_id, seqdata -> "${sample_id}:${seqdata}" }
            .collect()

        // Run per chromosome
        infer_snp_genotype_from_tumour(
            ch_tumour_args,
            ch_chroms,
            config_yaml,
        )
        ch_snp_genotype = infer_snp_genotype_from_tumour.out  // [chromosome, snp_genotype.tsv]
    }

    // Phase haplotypes per chromosome
    infer_haps_proc(
        ch_snp_genotype,   // [chromosome, snp_genotype.tsv]
        config_yaml,
        ref_data_dir,
    )

    // Merge all chromosome haplotypes
    ch_all_haps = infer_haps_proc.out
        .map { chrom, haps -> haps }
        .collect()

    merge_haps(ch_all_haps)

    emit:
    haplotypes = merge_haps.out   // path: haplotypes.tsv
}
