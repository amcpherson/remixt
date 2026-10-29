#!/usr/bin/env nextflow

/*
 * Sub-workflow: infer_haps
 *
 * Infers SNP genotypes (from normal or tumour), then phases haplotypes
 * per chromosome, then merges into a single haplotypes file.
 *
 * Input:
 *   ch_seqdata       - [sample_id, seqdata.h5] tuples
 *   ch_chromosomes   - channel of chromosome names
 *   normal_id        - val: normal sample id or null
 *   ref              - map of reference files
 *   ch_phasing       - [chromosome, map of phasing reference files] tuples
 *   phasing_method   - val: shapeit4 (GRCh38) or shapeit2 (GRCh37)
 *
 * Output:
 *   haplotypes       - path to merged haplotypes.tsv
 */

include { infer_snp_genotype_from_normal } from '../modules/infer_haps'
include { infer_snp_genotype_from_tumour } from '../modules/infer_haps'
include { infer_haps_shapeit4 }            from '../modules/infer_haps'
include { infer_haps_shapeit2 }            from '../modules/infer_haps'
include { merge_haps }                     from '../modules/infer_haps'


workflow infer_haps {

    take:
    ch_seqdata         // channel: [sample_id, seqdata.h5]
    ch_chromosomes     // channel: chromosome names
    normal_id          // val: string or 'none'
    ref                // map: reference files
    ch_phasing         // channel: [chromosome, map of phasing reference files]
    phasing_method     // val: shapeit4 or shapeit2

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
        )
        ch_snp_genotype = infer_snp_genotype_from_normal.out  // [chromosome, snp_genotype.tsv]

    } else {
        // --- Tumour-based genotyping ---
        // Collect all tumour seqdata, sorted by sample id for a stable task hash
        ch_tumour_args = ch_seqdata
            .toSortedList { a, b -> a[0] <=> b[0] }
            .map { pairs -> tuple(pairs.collect { pair -> pair[0] }, pairs.collect { pair -> pair[1] }) }

        // Run per chromosome
        infer_snp_genotype_from_tumour(
            ch_tumour_args,
            ch_chroms,
        )
        ch_snp_genotype = infer_snp_genotype_from_tumour.out  // [chromosome, snp_genotype.tsv]
    }

    // Phase haplotypes per chromosome
    // [chromosome, snp_genotype.tsv, map of phasing reference files]
    ch_phasing_input = ch_snp_genotype.join(ch_phasing)

    if (phasing_method == 'shapeit4') {
        infer_haps_shapeit4(
            ch_phasing_input.map { chrom, snp_genotype, files ->
                tuple(chrom, snp_genotype, files.bcf_reference, files.bcf_index, files.genetic_map)
            },
            ref.snp_positions,
        )
        ch_haps = infer_haps_shapeit4.out
    } else if (phasing_method == 'shapeit2') {
        infer_haps_shapeit2(
            ch_phasing_input.map { chrom, snp_genotype, files ->
                tuple(chrom, snp_genotype, files.genetic_map, files.haplotypes, files.legend, files.sample)
            },
        )
        ch_haps = infer_haps_shapeit2.out
    } else {
        error "Unknown phasing method ${phasing_method}"
    }

    // Merge all chromosome haplotypes
    ch_all_haps = ch_haps
        .map { chrom, haps -> haps }
        .collect()

    merge_haps(ch_all_haps)

    emit:
    haplotypes = merge_haps.out   // path: haplotypes.tsv
}
