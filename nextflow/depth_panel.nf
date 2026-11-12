#!/usr/bin/env nextflow

/*
 * Build a depth panel of normals for the depth bias model, see remixt/analysis/binbias.py.
 *
 * Each normal's fragment midpoints are counted in bins of panel_mask_bin_size and GC
 * corrected. The panel holds, per bin of panel_bin_size, the median over normals of the
 * residual log depth (the depth factor, which captures mappability and other reference
 * effects), and a mask of bins of panel_mask_bin_size whose depth varies across normals
 * well beyond Poisson.
 *
 * Usage:
 *   nextflow run nextflow/depth_panel.nf \
 *       --ref_data_dir /path/to/ref_data \
 *       --ensembl_genome_version GRCh37 \
 *       --out_dir /path/to/out \
 *       --panel_sample_ids N1,N2,N3 \
 *       --panel_bam_files n1.bam,n2.bam,n3.bam \
 *       --panel_female_ids N1,N2
 *
 * Use --panel_seqdata_files instead of --panel_bam_files for existing seqdata. Normals
 * not listed in --panel_female_ids are treated as male, use --panel_female_ids all if
 * every normal is female. Outputs depth_panel.tsv.gz and depth_panel_mask.tsv.gz, for the
 * depth_panel and depth_panel_mask params of main.nf.
 *
 * The normals should match the tumours to be corrected in reference, aligner and read
 * length, and duplicate filtering (filter_duplicates) should match the remixt runs.
 */

nextflow.enable.dsl = 2

include { extract_seqdata }   from './workflows/extract_seqdata'
include { annotate_bins }     from './modules/binbias'
include { count_bins }        from './modules/binbias'
include { build_depth_panel } from './modules/binbias'


/*
 * Reference file relative to the reference data directory.
 */
def ref_file(String name) {
    file("${params.ref_data_dir}/${name}", checkIfExists: true)
}


workflow {

    if (!params.ref_data_dir) { error "Please provide --ref_data_dir" }
    if (!params.out_dir) { error "Please provide --out_dir" }
    if (!params.panel_sample_ids) { error "Please provide --panel_sample_ids" }

    genome = params.genomes[params.ensembl_genome_version]
    if (!genome) { error "Unsupported genome ${params.ensembl_genome_version}" }

    genome_fasta = ref_file(genome.genome_fasta)
    genome_fai = ref_file(genome.genome_fai)
    mappability = ref_file(genome.mappability)

    sample_ids = params.panel_sample_ids.tokenize(',')
    female_ids = params.panel_female_ids == 'all' ? sample_ids : (params.panel_female_ids ?: '').tokenize(',')
    female_ids.each { sid -> if (!(sid in sample_ids)) { error "Female id ${sid} not in panel_sample_ids" } }

    // Base resolution of the counts is the mask resolution, the panel resolution must be a multiple
    mask_bin_size = params.panel_mask_bin_size as int
    panel_bin_size = params.panel_bin_size as int
    if (panel_bin_size % mask_bin_size != 0) {
        error "panel_bin_size ${panel_bin_size} must be a multiple of panel_mask_bin_size ${mask_bin_size}"
    }

    if (params.panel_seqdata_files) {
        seqdata_files = params.panel_seqdata_files.tokenize(',')
        if (seqdata_files.size() != sample_ids.size()) { error "Expected one seqdata file per sample id" }
        ch_seqdata = Channel.fromList([sample_ids, seqdata_files].transpose())
            .map { sid, f -> tuple(sid, file(f, checkIfExists: true)) }
    } else {
        bam_files = params.panel_bam_files.tokenize(',')
        if (bam_files.size() != sample_ids.size()) { error "Expected one bam file per sample id" }
        ch_bams = Channel.fromList([sample_ids, bam_files].transpose())
            .map { sid, f -> tuple(sid, file(f, checkIfExists: true)) }
        chromosomes = params.chromosomes.toString().tokenize(',')
        extract_seqdata(ch_bams, Channel.fromList(chromosomes), ref_file(genome.snp_positions))
        ch_seqdata = extract_seqdata.out.seqdata
    }

    bins = annotate_bins(genome_fasta, genome_fai, mappability, mask_bin_size)

    count_bins(ch_seqdata, bins)

    // Sorted by sample id for a stable task hash
    ch_counts = count_bins.out
        .toSortedList { a, b -> a[0] <=> b[0] }
        .map { pairs -> tuple(pairs.collect { p -> p[0] }, pairs.collect { p -> p[1] }) }

    build_depth_panel(
        ch_counts.map { ids, files -> ids },
        ch_counts.map { ids, files -> files },
        bins,
        female_ids,
    )
}
