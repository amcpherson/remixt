#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { extract_seqdata }  from './workflows/extract_seqdata'
include { remixt_seqdata }   from './workflows/remixt_seqdata'


/*
 * Resolve chromosome list from the reference data.
 */
process get_chromosomes {
    label 'mem_medium'

    input:
    path config_yaml
    val ref_data_dir

    output:
    path 'chromosomes.txt'

    script:
    """
    get_chromosomes.py \
        --config ${config_yaml} \
        --ref_data_dir ${ref_data_dir} \
        --output chromosomes.txt
    """
}


/*
 * ========================================================
 *  ENTRY POINT: remixt_bam
 *  Mirrors create_remixt_bam_workflow from pypeliner
 * ========================================================
 */
workflow {

    // --- Validate params ---
    if (!params.ref_data_dir) { error "Please provide --ref_data_dir" }
    if (!params.raw_data_dir) { error "Please provide --raw_data_dir" }

    ref_data_dir = params.ref_data_dir
    breakpoints  = file(params.breakpoint_file)

    // --- Config ---
    config_yaml = params.config
        ? Channel.fromPath(params.config).first()
        : Channel.fromPath("${projectDir}/assets/empty_config.yaml").first()

    // --- Build sample channels ---
    tumour_ids  = params.tumour_sample_ids.tokenize(',')
    tumour_bams = params.tumour_bam_files.tokenize(',')

    // Channel of [sample_id, bam_file] for tumours
    ch_tumour_bams = Channel.fromList(
        [tumour_ids, tumour_bams].transpose()
    ).map { sid, bam -> tuple(sid, file(bam)) }

    // If normal is provided, add to full sample set
    normal_id = params.normal_sample_id ?: 'none'
    if (normal_id != 'none') {
        ch_normal_bam = Channel.of(
            tuple(params.normal_sample_id, file(params.normal_bam_file))
        )
        ch_all_bams = ch_tumour_bams.mix(ch_normal_bam)
    } else {
        ch_all_bams = ch_tumour_bams
    }

    // --- Get chromosomes ---
    ch_chromosomes = get_chromosomes(config_yaml, ref_data_dir)
        .splitText()
        .map { it.trim() }

    // --- Extract seqdata per sample ---
    extract_seqdata(
        ch_all_bams,
        config_yaml,
        ref_data_dir,
        ch_chromosomes,
    )
    // out.seqdata: [sample_id, seqdata.h5]

    // --- Run full remixt pipeline on seqdata ---
    remixt_seqdata(
        extract_seqdata.out.seqdata,
        breakpoints,
        config_yaml,
        ref_data_dir,
        ch_chromosomes,
        normal_id,
    )
    // out.results: [tumour_id, results.h5]
}
