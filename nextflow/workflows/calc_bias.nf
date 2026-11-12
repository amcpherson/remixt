#!/usr/bin/env nextflow

/*
 * Sub-workflow: calc_bias
 *
 * Calculates bias corrected segment lengths per tumour sample with the bin level
 * depth model, see remixt/analysis/binbias.py. Read depth is GC corrected per bin,
 * with the depth panel of normals in params.depth_panel (and its mask in
 * params.depth_panel_mask) if given, and otherwise with a smooth function of GC and
 * mappable fraction in bins of params.bias_bin_size.
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
include { annotate_bins }       from '../modules/binbias'
include { bin_segment_lengths } from '../modules/binbias'


workflow calc_bias {

    take:
    ch_tumour_seqdata    // channel: [tumour_id, seqdata.h5]
    ch_rawcounts         // channel: [tumour_id, rawcounts.tsv]
    ref                  // map: reference files

    main:

    // 1. Fragment stats per tumour, the mean fragment length trims segment ends
    calc_fragment_stats(ch_tumour_seqdata)
    // out: [tumour_id, fragstats.json]

    // 2. Depth panel, or annotated bins for the GC and mappability curve
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

    // 3. Segment lengths from per bin expected depth
    ch_input = ch_tumour_seqdata
        .join(ch_rawcounts)
        .join(calc_fragment_stats.out)
        // [tumour_id, seqdata, rawcounts, fragstats]

    bin_segment_lengths(ch_input, bins_or_panel, mask, method)

    emit:
    segment_lengths = bin_segment_lengths.out.segment_lengths   // [tumour_id, segment_lengths.tsv]
}
