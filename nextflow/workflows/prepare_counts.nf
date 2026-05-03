#!/usr/bin/env nextflow

/*
 * Sub-workflow: prepare_counts
 *
 * Mirrors create_prepare_counts_workflow from pypeliner.
 * Computes segment and allele read counts, phases across tumours,
 * and produces combined count tables per tumour.
 *
 * Input:
 *   ch_tumour_seqdata  - [tumour_id, seqdata.h5] tuples
 *   segments           - path to segments.tsv
 *   haplotypes         - path to haplotypes.tsv
 *
 * Output:
 *   rawcounts          - [tumour_id, counts.tsv] tuples
 */

include { segment_readcount }         from '../modules/prepare_counts'
include { haplotype_allele_readcount } from '../modules/prepare_counts'
include { phase_segments }             from '../modules/prepare_counts'
include { prepare_readcount_table }    from '../modules/prepare_counts'


workflow prepare_counts {

    take:
    ch_tumour_seqdata    // channel: [tumour_id, seqdata.h5]
    segments             // path
    haplotypes           // path

    main:

    // 1. Count reads per segment, per tumour
    segment_readcount(ch_tumour_seqdata, segments)
    // out: [tumour_id, segment_counts.tsv]

    // 2. Count allele reads per segment per tumour
    haplotype_allele_readcount(ch_tumour_seqdata, segments, haplotypes)
    // out: [tumour_id, allele_counts.tsv]

    // 3. Phase segments across all tumours (merge then split)
    // Collect all allele counts as "tumour_id:path" args
    ch_tumour_allele_args = haplotype_allele_readcount.out
        .map { tumour_id, allele_counts -> "${tumour_id}:${allele_counts}" }
        .collect()

    phase_segments(ch_tumour_allele_args)
    // out: list of phased.{tumour_id}.tsv files

    // Re-associate phased files with tumour_ids
    ch_phased = phase_segments.out
        .flatten()
        .map { f ->
            def tumour_id = f.name.replaceAll(/^phased\./, '').replaceAll(/\.tsv$/, '')
            tuple(tumour_id, f)
        }

    // 4. Combine segment counts + phased allele counts per tumour
    ch_combined = segment_readcount.out
        .join(ch_phased)
        // [tumour_id, segment_counts, phased_allele_counts]

    prepare_readcount_table(ch_combined)
    // out: [tumour_id, counts.tsv]

    emit:
    rawcounts = prepare_readcount_table.out   // [tumour_id, counts.tsv]
}
