#!/usr/bin/env nextflow

/*
 * Sub-workflow: calc_bias
 *
 * Calculates GC and mappability bias correction per tumour sample, with the
 * position level model (params.bias_method = 'remixt') or the bin level model
 * (params.bias_method = 'bins', see remixt/analysis/binbias.py).
 *
 * Input:
 *   ch_tumour_seqdata  - [tumour_id, seqdata.h5] tuples
 *   ch_rawcounts       - [tumour_id, rawcounts.tsv] tuples (segment file per tumour)
 *   ref                - map of reference files
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
include { annotate_bins }       from '../modules/binbias'
include { bin_segment_lengths } from '../modules/binbias'


workflow calc_bias {

    take:
    ch_tumour_seqdata    // channel: [tumour_id, seqdata.h5]
    ch_rawcounts         // channel: [tumour_id, rawcounts.tsv]
    ref                  // map: reference files

    main:

    // 1. Calculate fragment stats per tumour
    calc_fragment_stats(ch_tumour_seqdata)
    // out: [tumour_id, fragstats.json]

    if (params.bias_method == 'bins') {
        bin_lengths(ch_tumour_seqdata, ch_rawcounts, calc_fragment_stats.out, ref)
        segment_lengths = bin_lengths.out.segment_lengths
    } else if (params.bias_method == 'remixt') {
        remixt_lengths(ch_tumour_seqdata, ch_rawcounts, calc_fragment_stats.out, ref)
        segment_lengths = remixt_lengths.out.segment_lengths
    } else {
        error "Unknown bias_method ${params.bias_method}, expected remixt or bins"
    }

    emit:
    segment_lengths = segment_lengths   // [tumour_id, segment_lengths.tsv]
}


/*
 * Bin level depth model, with the depth panel of normals in params.depth_panel if
 * given, and otherwise a smooth function of GC and mappable fraction in bins of
 * params.bias_bin_size.
 */
workflow bin_lengths {

    take:
    ch_tumour_seqdata    // channel: [tumour_id, seqdata.h5]
    ch_rawcounts         // channel: [tumour_id, rawcounts.tsv]
    ch_fragstats         // channel: [tumour_id, fragstats.json]
    ref                  // map: reference files

    main:

    no_file = file("${projectDir}/assets/NO_FILE")

    if (params.depth_panel) {
        method = 'panel'
        bins_or_panel = file(params.depth_panel, checkIfExists: true)
        mask = params.depth_panel_mask ? file(params.depth_panel_mask, checkIfExists: true) : no_file
    } else {
        method = 'curve'
        bins_or_panel = annotate_bins(ref.genome_fasta, ref.genome_fai, ref.mappability, params.bias_bin_size)
        mask = no_file
    }

    ch_input = ch_tumour_seqdata
        .join(ch_rawcounts)
        .join(ch_fragstats)
        // [tumour_id, seqdata, rawcounts, fragstats]

    bin_segment_lengths(ch_input, bins_or_panel, mask, method)

    emit:
    segment_lengths = bin_segment_lengths.out.segment_lengths   // [tumour_id, segment_lengths.tsv]
}


/*
 * Position level GC and mappability model.
 */
workflow remixt_lengths {

    take:
    ch_tumour_seqdata    // channel: [tumour_id, seqdata.h5]
    ch_rawcounts         // channel: [tumour_id, rawcounts.tsv]
    ch_fragstats         // channel: [tumour_id, fragstats.json]
    ref                  // map: reference files

    main:

    // 2. Sample GC - needs seqdata + fragstats
    ch_sample_gc_input = ch_tumour_seqdata
        .join(ch_fragstats)
        // [tumour_id, seqdata, fragstats.json]

    sample_gc(ch_sample_gc_input, ref.genome_fasta, ref.genome_fai, ref.mappability)
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
    ch_gc_dist = gc_lowess.out.map { tid, loess, table -> tuple(tid, loess) }  // [tumour_id, gcloess.tsv]

    ch_bias_input = ch_chunks
        .combine(ch_fragstats, by: 0)
        .combine(ch_gc_dist, by: 0)
        // [tumour_id, chunk_idx, chunk_file, fragstats.json, gcloess.tsv]

    gc_map_bias(ch_bias_input, ref.genome_fasta, ref.mappability)
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
