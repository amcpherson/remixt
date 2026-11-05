#!/usr/bin/env nextflow

/*
 * Calculate the mappability file for a reference genome, as listed in
 * conf/genomes.config, by aligning every k-mer of the genome back to the
 * genome with bwa.
 *
 * Usage:
 *   nextflow run nextflow/mappability.nf \
 *       --ref_data_dir /path/to/ref_data \
 *       --ensembl_genome_version GRCh38 \
 *       --out_dir /path/to/ref_data
 *
 * The genome fasta and fai are read from ref_data_dir, and the bwa index is
 * used if present next to the fasta, otherwise it is built.  K-mers are aligned
 * in regions of mappability_chunk_size k-mers.
 */

nextflow.enable.dsl = 2

include { bwa_index }          from './modules/mappability'
include { mappability_region } from './modules/mappability'
include { merge_mappability }  from './modules/mappability'


workflow {

    if (!params.ref_data_dir) { error "Please provide --ref_data_dir" }
    if (!params.out_dir) { error "Please provide --out_dir" }

    genome = params.genomes[params.ensembl_genome_version]
    if (!genome) { error "Unsupported genome ${params.ensembl_genome_version}" }

    genome_fasta = file("${params.ref_data_dir}/${genome.genome_fasta}", checkIfExists: true)
    genome_fai = file("${params.ref_data_dir}/${genome.genome_fai}", checkIfExists: true)

    // Use an existing bwa index, or build one
    index_files = ['amb', 'ann', 'bwt', 'pac', 'sa'].collect { ext -> file("${genome_fasta}.${ext}") }
    if (index_files.every { f -> f.exists() }) {
        ch_bwa_index = channel.value(index_files)
    } else {
        ch_bwa_index = bwa_index(genome_fasta).collect()
    }

    // Regions of k-mer start positions for every sequence in the fasta, k-mers
    // that would extend past the end of the sequence are skipped when writing k-mers
    chunk_size = params.mappability_chunk_size as long
    chromosomes = []
    regions = []
    genome_fai.eachLine { line ->
        def fields = line.tokenize('\t')
        def chromosome = fields[0]
        def length = fields[1] as long
        chromosomes << chromosome
        def num_chunks = Math.ceilDiv(length, chunk_size)
        (0L..<num_chunks).each { chunk ->
            def start = chunk * chunk_size
            regions << tuple(chromosome, start, Math.min(start + chunk_size, length))
        }
    }

    mappability_region(Channel.fromList(regions), genome_fasta, genome_fai, ch_bwa_index)

    // Merge in genome order
    ch_runs = mappability_region.out
        .toSortedList { a, b -> chromosomes.indexOf(a[0]) <=> chromosomes.indexOf(b[0]) ?: a[1] <=> b[1] }
        .map { runs -> runs.collect { r -> r[2] } }

    merge_mappability(ch_runs, genome.mappability)
}
