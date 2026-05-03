#!/usr/bin/env nextflow

/*
 * Sub-workflow: calc_bias
 *
 * Mirrors create_calc_bias_workflow from pypeliner.
 * Calculates GC and mappability bias correction per tumour sample.
 *
 * Input:
 *   ch_tumour_seqdata  - [tumour_id, seqdata.h5] tuples
 *   ch_rawcounts       - [tumour_id, rawcounts.tsv] tuples (segment file per tumour)
 *   ref_data_dir       - path to reference data directory
 *
 * Output:
 *   segment_lengths    - [tumour_id, segment_lengths.tsv] tuples
 */

include { calc_fragment_stats } from '../modules/calc_bias'
include { sample_gc }           from '../modules/calc_bias'
include { gc_lowess }           from '../modules/calc_bias'
include { split_segments }      from '../modules/calc_bias'
include { gc_map_bias }         from '../modules/calc_bias'
include { merge_biases }        from '../modules/calc_bias'
include { biased_length }       from '../modules/calc_bias'


workflow calc_bias {

    take:
    ch_tumour_seqdata    // channel: [tumour_id, seqdata.h5]
    ch_rawcounts         // channel: [tumour_id, rawcounts.tsv]
    ref_data_dir         // val

    main:

    // 1. Calculate fragment stats per tumour
    calc_fragment_stats(ch_tumour_seqdata)
    // out: [tumour_id, fragstats.json]

    // 2. Sample GC - needs seqdata + fragstats
    ch_sample_gc_input = ch_tumour_seqdata
        .join(calc_fragment_stats.out)
        // [tumour_id, seqdata, fragstats.json]

    sample_gc(ch_sample_gc_input, ref_data_dir)
    // out: [tumour_id, gcsamples.tsv]

    // 3. GC lowess
    gc_lowess(sample_gc.out)
    // out: [tumour_id, gcloess.tsv, gctable.tsv]

    // 4. Split segments into chunks
    split_segments(ch_rawcounts)
    // out: [tumour_id, [chunk files]]

    // 5. Flatten chunks for parallel gc_map_bias
    // Combine with fragstats and gc_dist per tumour
    ch_chunks = split_segments.out
        .flatMap { tumour_id, chunk_files ->
            def files = chunk_files instanceof List ? chunk_files : [chunk_files]
            files.withIndex().collect { f, idx -> tuple(tumour_id, idx, f) }
        }
        // [tumour_id, chunk_idx, chunk_file]

    // Join fragstats and gc_dist per tumour_id
    ch_fragstats = calc_fragment_stats.out   // [tumour_id, fragstats.json]
    ch_gc_dist = gc_lowess.out.map { tid, loess, table -> tuple(tid, loess) }  // [tumour_id, gcloess.tsv]

    ch_bias_input = ch_chunks
        .combine(ch_fragstats, by: 0)
        .combine(ch_gc_dist, by: 0)
        // [tumour_id, chunk_idx, chunk_file, fragstats.json, gcloess.tsv]

    gc_map_bias(ch_bias_input, ref_data_dir)
    // out: [tumour_id, bias.chunk_idx.tsv]

    // 6. Merge biases per tumour
    ch_biases_grouped = gc_map_bias.out
        .map { tumour_id, bias_file -> tuple(tumour_id, bias_file) }
        .groupTuple()

    merge_biases(ch_biases_grouped)
    // out: [tumour_id, biases.tsv]

    // 7. Compute biased lengths
    biased_length(merge_biases.out)
    // out: [tumour_id, segment_lengths.tsv]

    emit:
    segment_lengths = biased_length.out   // [tumour_id, segment_lengths.tsv]
}
