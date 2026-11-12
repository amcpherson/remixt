#!/usr/bin/env nextflow

/*
 * Processes for the bin level depth model and the depth panel of normals.
 */

include { remixt_args; id_path_args } from './utils'


process annotate_bins {
    label 'mem_very_high'

    input:
    path genome_fasta
    path genome_fai
    path mappability
    val bin_size

    output:
    path "bins.${bin_size}.tsv.gz"

    script:
    """
    annotate_bins.py \
        --genome_fasta ${genome_fasta} \
        --genome_fai ${genome_fai} \
        --mappability ${mappability} \
        --bin_size ${bin_size} \
        --output bins.${bin_size}.tsv.gz \
        ${remixt_args('annotate_bins')}
    """
}


process count_bins {
    label 'mem_high'
    tag "${sample_id}"

    input:
    tuple val(sample_id), path(seqdata)
    path bins

    output:
    tuple val(sample_id), path("${sample_id}.bin_counts.tsv.gz")

    script:
    """
    count_bins.py \
        --seqdata ${seqdata} \
        --bins ${bins} \
        --output ${sample_id}.bin_counts.tsv.gz \
        ${remixt_args('count_bins')}
    """
}


process build_depth_panel {
    label 'mem_very_high'

    input:
    val sample_ids
    path counts
    path bins
    val female_ids

    output:
    path "depth_panel.tsv.gz", emit: panel
    path "depth_panel_mask.tsv.gz", emit: mask
    path "depth_panel.log", emit: log

    script:
    """
    build_depth_panel.py \
        --bins ${bins} \
        --female_ids '${female_ids.join(',')}' \
        --output_panel depth_panel.tsv.gz \
        --output_mask depth_panel_mask.tsv.gz \
        ${remixt_args('build_depth_panel')} \
        ${id_path_args(sample_ids, counts)} \
        > depth_panel.log
    """
}


process bin_segment_lengths {
    label 'mem_high'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(seqdata), path(rawcounts), path(fragstats)
    path bins_or_panel
    path mask
    val method

    output:
    tuple val(tumour_id), path("${tumour_id}.segment_lengths.tsv"), emit: segment_lengths
    tuple val(tumour_id), path("${tumour_id}.bin_depth.tsv.gz"), emit: bin_depth

    script:
    def ref_arg = method == 'panel' ? "--panel ${bins_or_panel}" : "--bins ${bins_or_panel}"
    def mask_arg = (method == 'panel' && mask.name != 'NO_FILE') ? "--mask ${mask}" : ''
    """
    bin_segment_lengths.py \
        --counts ${rawcounts} \
        --seqdata ${seqdata} \
        --fragstats ${fragstats} \
        --output ${tumour_id}.segment_lengths.tsv \
        --output_bins ${tumour_id}.bin_depth.tsv.gz \
        ${ref_arg} ${mask_arg} \
        ${remixt_args('bin_segment_lengths')}
    """
}
