#!/usr/bin/env nextflow

/*
 * Sub-workflow: extract_seqdata
 *
 * For each sample, extracts seqdata per chromosome in parallel,
 * then merges into a single HDF5 file.
 *
 * Input channels:
 *   ch_bams          - [sample_id, bam_file] tuples
 *   ch_chromosomes   - channel of chromosome name strings
 *   snp_positions    - thousand genomes SNP positions file
 *
 * Output channels:
 *   seqdata          - [sample_id, seqdata.h5] tuples
 */

include { create_chromosome_seqdata } from '../modules/extract_seqdata'
include { merge_seqdata }             from '../modules/extract_seqdata'


workflow extract_seqdata {

    take:
    ch_bams            // channel: [sample_id, bam_file]
    ch_chromosomes     // channel: chromosome names
    snp_positions      // path: thousand genomes SNP positions

    main:

    // Prepare BAM inputs with index files.
    // Expect .bai alongside .bam (standard convention).
    ch_bams_with_index = ch_bams.map { sample_id, bam ->
        def bai = file("${bam}.bai")
        tuple(sample_id, bam, bai)
    }

    // Cross each sample with each chromosome to create
    // [sample_id, bam, bai] x chromosome combinations.
    ch_inputs = ch_bams_with_index.combine(ch_chromosomes.collect().flatMap())

    create_chromosome_seqdata(
        ch_inputs.map { sid, bam, bai, chrom -> tuple(sid, bam, bai) },
        ch_inputs.map { sid, bam, bai, chrom -> chrom },
        snp_positions,
    )

    // Group per-chromosome seqdata by sample_id for merging.
    // create_chromosome_seqdata.out: [sample_id, chromosome, seqdata.h5]
    ch_grouped = create_chromosome_seqdata.out
        .map { sample_id, chrom, seqdata -> tuple(sample_id, seqdata) }
        .groupTuple()

    // Merge all chromosome seqdata into one file per sample.
    merge_seqdata(ch_grouped)

    emit:
    seqdata = merge_seqdata.out   // [sample_id, seqdata.h5]
}
