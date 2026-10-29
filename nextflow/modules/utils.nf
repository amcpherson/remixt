#!/usr/bin/env nextflow

/*
 * remixt params forwarded to each script as command line options. Params left
 * null are not passed and use the remixt default, see remixt/defaults.py.
 */
def remixt_param_names() {
    [
        create_chromosome_seqdata:      ['bam_max_fragment_length', 'bam_max_soft_clipped', 'bam_check_proper_pair'],
        create_segments:                ['chromosomes', 'chr_name_prefix', 'segment_length'],
        infer_snp_genotype_from_normal: ['sequencing_base_call_error', 'het_snp_call_threshold'],
        infer_snp_genotype_from_tumour: ['sequencing_base_call_error', 'homozygous_p_value_threshold'],
        infer_haps_shapeit4:            ['chr_name_prefix', 'is_female', 'shapeit_num_samples', 'shapeit_confidence_threshold'],
        infer_haps_shapeit2:            ['is_female', 'shapeit_num_samples', 'shapeit_confidence_threshold'],
        segment_readcount:              ['filter_duplicates', 'map_qual_threshold'],
        haplotype_allele_readcount:     ['filter_duplicates', 'map_qual_threshold'],
        calc_fragment_stats:            ['filter_duplicates', 'map_qual_threshold'],
        sample_gc:                      ['chromosomes', 'chr_name_prefix', 'sample_gc_num_positions', 'gc_position_offset',
                                         'filter_duplicates', 'map_qual_threshold', 'random_seed'],
        gc_map_bias:                    ['do_gc_correction', 'do_mappability_correction', 'gc_position_offset',
                                         'map_qual_threshold', 'mappability_length'],
        ploidy_analysis_plots:          ['chromosomes'],
        init_model:                     ['min_ploidy', 'max_ploidy', 'h_normal', 'h_tumour', 'tumour_mix_fractions',
                                         'divergence_weights', 'max_copy_number', 'random_seed'],
        fit_model:                      ['is_female', 'normal_contamination', 'max_copy_number',
                                         'likelihood_min_segment_length', 'likelihood_min_proportion_genotyped',
                                         'num_em_iter', 'num_update_iter', 'disable_breakpoints', 'do_h_update', 'random_seed'],
        collate_results:                ['max_prop_diverge'],
    ]
}


/*
 * "id:path" command line args from a list of ids and a matching list of staged files.
 */
def id_path_args(ids, files) {
    def file_list = files instanceof List ? files : [files]
    [ids, file_list].transpose()
        .collect { pair -> "${pair[0]}:${pair[1]}" }
        .join(' ')
}


/*
 * Command line options for the remixt params of a script.
 */
def remixt_args(String script) {
    remixt_param_names()[script]
        .findAll { name -> params[name] != null }
        .collect { name ->
            def value = params[name] instanceof List ? params[name].join(',') : params[name]
            "--${name} '${value}'"
        }
        .join(' ')
}
