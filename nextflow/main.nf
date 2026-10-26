#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { extract_seqdata }  from './workflows/extract_seqdata'
include { remixt_seqdata }   from './workflows/remixt_seqdata'
include { write_config }     from './modules/remixt_core'


/*
 * Command line params are strings when the param defaults to null, convert
 * booleans and numbers.
 */
def parse_param(value) {
    if (!(value instanceof String)) {
        return value
    }
    if (value in ['true', 'false']) {
        return value == 'true'
    }
    if (value.isLong()) {
        return value as long
    }
    if (value.isDouble()) {
        return value as double
    }
    return value
}


/*
 * Remixt config built from params, params left null use the remixt defaults.
 * Comma separated list params are converted to lists.
 */
def remixt_config() {
    def keys = [
        'chromosomes', 'chr_name_prefix', 'ensembl_genome_version', 'is_female',
        'bam_max_fragment_length', 'bam_max_soft_clipped', 'bam_check_proper_pair',
        'filter_duplicates', 'map_qual_threshold',
        'sequencing_base_call_error', 'het_snp_call_threshold', 'homozygous_p_value_threshold',
        'shapeit_num_samples', 'shapeit_confidence_threshold',
        'segment_length',
        'sample_gc_num_positions', 'gc_position_offset', 'do_gc_correction',
        'do_mappability_correction', 'mappability_length',
        'min_ploidy', 'max_ploidy', 'h_normal', 'h_tumour', 'tumour_mix_fractions',
        'divergence_weights', 'max_copy_number', 'random_seed',
        'normal_contamination', 'likelihood_min_segment_length',
        'likelihood_min_proportion_genotyped', 'num_em_iter', 'num_update_iter',
        'disable_breakpoints', 'do_h_update',
        'max_prop_diverge',
    ]
    def string_list_keys = ['chromosomes']
    def number_list_keys = ['tumour_mix_fractions', 'divergence_weights']

    keys
        .findAll { key -> params[key] != null }
        .collectEntries { key ->
            def value = parse_param(params[key])
            if (key in string_list_keys) {
                value = value.toString().tokenize(',')
            } else if (key in number_list_keys) {
                value = value.toString().tokenize(',').collect { x -> x as double }
            }
            [key, value]
        }
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
    if (!params.out_dir) { error "Please provide --out_dir" }

    ref_data_dir  = params.ref_data_dir
    breakpoints   = file(params.breakpoint_file)

    ch_config = write_config(remixt_config())

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
    ch_chromosomes = Channel.fromList(
        params.chromosomes.toString().tokenize(',')
    )

    // --- Extract seqdata per sample ---
    extract_seqdata(
        ch_all_bams,
        ref_data_dir,
        ch_chromosomes,
        ch_config,
    )
    // out.seqdata: [sample_id, seqdata.h5]

    // --- Run full remixt pipeline on seqdata ---
    remixt_seqdata(
        extract_seqdata.out.seqdata,
        breakpoints,
        ref_data_dir,
        ch_chromosomes,
        normal_id,
        ch_config,
    )
    // out.results: [tumour_id, results.h5]
}
