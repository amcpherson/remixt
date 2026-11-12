#!/usr/bin/env nextflow

/*
 * Sub-workflow: remixt_seqdata
 *
 * Orchestrates the full pipeline from seqdata files through to final results:
 *   create_segments -> infer_haps -> prepare_counts -> calc_bias
 *   -> create_experiment -> ploidy_plots -> fit_model
 *
 * Input:
 *   ch_seqdata       - [sample_id, seqdata.h5] tuples (all samples)
 *   breakpoints      - path to breakpoints file
 *   ch_chromosomes   - channel of chromosome names
 *   normal_id        - val: normal sample id or 'none'
 *   ref              - map of reference files
 *   ch_phasing       - [chromosome, map of phasing reference files] tuples
 *   phasing_method   - val: shapeit4 (GRCh38) or shapeit2 (GRCh37)
 *
 * Output:
 *   results          - [tumour_id, results.h5] tuples
 */

include { infer_haps }               from './infer_haps'
include { prepare_counts }           from './prepare_counts'
include { calc_bias }                from './calc_bias'
include { fit_model }                from './fit_model'
include { create_segments }          from '../modules/remixt_core'
include { create_experiment }        from '../modules/remixt_core'
include { ploidy_analysis_plots }    from '../modules/remixt_core'


workflow remixt_seqdata {

    take:
    ch_seqdata         // channel: [sample_id, seqdata.h5]
    breakpoints        // path
    ch_chromosomes     // channel: chromosome names
    normal_id          // val: string or 'none'
    ref                // map: reference files
    ch_phasing         // channel: [chromosome, map of phasing reference files]
    phasing_method     // val: shapeit4 or shapeit2

    main:

    // --- Filter tumour samples ---
    if (normal_id != 'none') {
        ch_tumour_seqdata = ch_seqdata
            .filter { sample_id, seqdata -> sample_id != normal_id }
    } else {
        ch_tumour_seqdata = ch_seqdata
    }

    // --- 1. Create segments ---
    create_segments(ref.genome_fai, ref.gap_table, breakpoints)
    segments = create_segments.out   // path: segments.tsv

    // --- 2. Infer haplotypes, or use params.haplotypes_file ---
    if (params.haplotypes_file) {
        haplotypes = file(params.haplotypes_file, checkIfExists: true)
    } else {
        infer_haps(
            ch_seqdata,
            ch_chromosomes,
            normal_id,
            ref,
            ch_phasing,
            phasing_method,
        )
        haplotypes = infer_haps.out.haplotypes   // path: haplotypes.tsv
    }

    // --- 3. Prepare read counts ---
    prepare_counts(
        ch_tumour_seqdata,
        segments,
        haplotypes,
    )
    // out.rawcounts: [tumour_id, counts.tsv]

    // --- 4. Calculate GC/mappability bias ---
    calc_bias(
        ch_tumour_seqdata,
        prepare_counts.out.rawcounts,
        ref,
    )
    // out.segment_lengths: [tumour_id, segment_lengths.tsv]
    // segment_lengths is the final "counts" input for experiments.

    // --- 5. Create experiment ---
    create_experiment(
        calc_bias.out.segment_lengths,
        breakpoints,
    )
    // out: [tumour_id, experiment.pickle]

    // --- 6. Ploidy analysis plots ---
    ploidy_analysis_plots(
        create_experiment.out,
    )

    // --- 7. Fit model ---
    fit_model(
        create_experiment.out,
    )
    // out.results: [tumour_id, results.h5]

    emit:
    results = fit_model.out.results   // [tumour_id, results.h5]
}
