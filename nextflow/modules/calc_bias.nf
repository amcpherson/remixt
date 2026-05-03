#!/usr/bin/env nextflow

/*
 * Processes for GC/mappability bias calculation.
 * Mirrors create_calc_bias_workflow from pypeliner.
 */


process calc_fragment_stats {
    label 'mem_medium'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(seqdata)

    output:
    tuple val(tumour_id), path("${tumour_id}.fragstats.json")

    script:
    def args = task.ext.args ?: ''
    """
    calc_fragment_stats.py \
        --seqdata ${seqdata} \
        --output ${tumour_id}.fragstats.json \
        ${args}
    """
}


process sample_gc {
    label 'mem_medium'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(seqdata), path(fragstats)
    val ref_data_dir

    output:
    tuple val(tumour_id), path("${tumour_id}.gcsamples.tsv")

    script:
    def args = task.ext.args ?: ''
    """
    FRAG_MEAN=\$(python3 -c "import json; print(json.load(open('${fragstats}'))['fragment_mean'])")
    sample_gc.py \
        --seqdata ${seqdata} \
        --fragment_mean \$FRAG_MEAN \
        --ref_data_dir ${ref_data_dir} \
        --output ${tumour_id}.gcsamples.tsv \
        ${args}
    """
}


process gc_lowess {
    label 'mem_medium'
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(gc_samples)

    output:
    tuple val(tumour_id), path("${tumour_id}.gcloess.tsv"), path("${tumour_id}.gctable.tsv")

    script:
    """
    gc_lowess.py \
        --gc_samples ${gc_samples} \
        --output_dist ${tumour_id}.gcloess.tsv \
        --output_table ${tumour_id}.gctable.tsv
    """
}


process split_segments {
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(segment_file)

    output:
    tuple val(tumour_id), path("chunks.*.tsv")

    script:
    """
    split_table.py \
        --input ${segment_file} \
        --num_rows 100 \
        --output_prefix chunks
    """
}


process gc_map_bias {
    label 'mem_medium'
    tag "${tumour_id}_${chunk_idx}"

    input:
    tuple val(tumour_id), val(chunk_idx), path(segment_chunk), path(fragstats), path(gc_dist)
    val ref_data_dir

    output:
    tuple val(tumour_id), path("${tumour_id}.bias.${chunk_idx}.tsv")

    script:
    def args = task.ext.args ?: ''
    """
    gc_map_bias.py \
        --segments ${segment_chunk} \
        --fragstats ${fragstats} \
        --gc_dist ${gc_dist} \
        --ref_data_dir ${ref_data_dir} \
        --output ${tumour_id}.bias.${chunk_idx}.tsv \
        ${args}
    """
}


process merge_biases {
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(bias_files)

    output:
    tuple val(tumour_id), path("${tumour_id}.biases.tsv")

    script:
    """
    merge_tables.py --output ${tumour_id}.biases.tsv ${bias_files}
    """
}


process biased_length {
    tag "${tumour_id}"

    input:
    tuple val(tumour_id), path(biases)

    output:
    tuple val(tumour_id), path("${tumour_id}.segment_lengths.tsv")

    script:
    """
    biased_length.py \
        --biases ${biases} \
        --output ${tumour_id}.segment_lengths.tsv
    """
}
