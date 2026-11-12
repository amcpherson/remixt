#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { extract_seqdata }  from './workflows/extract_seqdata'
include { remixt_seqdata }   from './workflows/remixt_seqdata'
include { remixt_param_names } from './modules/utils'


/*
 * Params that configure the pipeline rather than a remixt step.
 */
def pipeline_param_names() {
    [
        'ref_data_dir', 'out_dir', 'breakpoint_file', 'sample_sheet',
        'tumour_sample_ids', 'tumour_bam_files', 'results_files',
        'normal_sample_id', 'normal_bam_file', 'ensembl_genome_version',
        'genomes', 'remixt_container', 'mappability_chunk_size',
        'tumour_seqdata_files', 'normal_seqdata_file', 'haplotypes_file',
        'bias_method', 'bias_bin_size', 'depth_panel', 'depth_panel_mask',
        'panel_sample_ids', 'panel_bam_files', 'panel_seqdata_files', 'panel_female_ids',
    ]
}


/*
 * Reference file relative to the reference data directory.
 */
def ref_file(String name) {
    file("${params.ref_data_dir}/${name}", checkIfExists: true)
}


/*
 * ========================================================
 *  ENTRY POINT: remixt_bam
 * ========================================================
 */
workflow {

    // --- Validate params ---
    if (!params.ref_data_dir) { error "Please provide --ref_data_dir" }
    if (!params.out_dir) { error "Please provide --out_dir" }

    // Warn about params not forwarded to any remixt step, e.g. typos
    def known_params = pipeline_param_names() + remixt_param_names().values().flatten()
    params.keySet().findAll { name -> !(name in known_params) }.each { name ->
        log.warn "Param ${name} is not used by any remixt step"
    }

    breakpoints   = file(params.breakpoint_file)

    // --- Reference files ---
    genome = params.genomes[params.ensembl_genome_version]
    if (!genome) { error "Unsupported genome ${params.ensembl_genome_version}" }

    ref = [
        genome_fasta:  ref_file(genome.genome_fasta),
        genome_fai:    ref_file(genome.genome_fai),
        gap_table:     ref_file(genome.gap_table),
        mappability:   ref_file(genome.mappability),
        snp_positions: ref_file(genome.snp_positions),
    ]

    // --- Get chromosomes ---
    chromosomes = params.chromosomes.toString().tokenize(',')
    ch_chromosomes = Channel.fromList(chromosomes)

    // Phasing reference files per chromosome: [chromosome, [name: file]]
    ch_phasing = Channel.fromList(chromosomes).map { chrom ->
        def phasing_chrom = chrom.replaceFirst(/^chr/, '')
        phasing_chrom = genome.phasing_chromosomes?.get(phasing_chrom) ?: phasing_chrom
        def files = genome.phasing_files.collectEntries { name, pattern ->
            [name, ref_file(pattern.replace('{chrom}', phasing_chrom))]
        }
        tuple(chrom, files)
    }

    normal_id = params.normal_sample_id ?: 'none'
    tumour_ids = params.tumour_sample_ids.tokenize(',')

    if (params.tumour_seqdata_files) {
        // --- Existing seqdata per sample ---
        tumour_seqdata = params.tumour_seqdata_files.tokenize(',')
        ch_seqdata = Channel.fromList([tumour_ids, tumour_seqdata].transpose())
            .map { sid, f -> tuple(sid, file(f, checkIfExists: true)) }
        if (normal_id != 'none') {
            ch_seqdata = ch_seqdata.mix(Channel.of(
                tuple(normal_id, file(params.normal_seqdata_file, checkIfExists: true))))
        }
    } else {
        // --- Build sample channels ---
        tumour_bams = params.tumour_bam_files.tokenize(',')

        // Channel of [sample_id, bam_file] for tumours
        ch_tumour_bams = Channel.fromList(
            [tumour_ids, tumour_bams].transpose()
        ).map { sid, bam -> tuple(sid, file(bam)) }

        // If normal is provided, add to full sample set
        if (normal_id != 'none') {
            ch_normal_bam = Channel.of(
                tuple(params.normal_sample_id, file(params.normal_bam_file))
            )
            ch_all_bams = ch_tumour_bams.mix(ch_normal_bam)
        } else {
            ch_all_bams = ch_tumour_bams
        }

        // --- Extract seqdata per sample ---
        extract_seqdata(
            ch_all_bams,
            ch_chromosomes,
            ref.snp_positions,
        )
        ch_seqdata = extract_seqdata.out.seqdata
        // [sample_id, seqdata.h5]
    }

    // --- Run full remixt pipeline on seqdata ---
    remixt_seqdata(
        ch_seqdata,
        breakpoints,
        ch_chromosomes,
        normal_id,
        ref,
        ch_phasing,
        genome.phasing_method,
    )
    // out.results: [tumour_id, results.h5]
}
