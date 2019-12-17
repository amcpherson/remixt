import os
import pypeliner
import pypeliner.managed as mgd

import remixt.config


def create_extract_seqdata_workflow(
     bam_filename,
     seqdata_filename,
     config,
     ref_data_dir,
     no_parallelism=False
):
    chromosomes = remixt.config.get_chromosomes(config, ref_data_dir)
    snp_positions_filename = remixt.config.get_filename(config, ref_data_dir, 'snp_positions')

    bam_max_fragment_length = remixt.config.get_param(config, 'bam_max_fragment_length')
    bam_max_soft_clipped = remixt.config.get_param(config, 'bam_max_soft_clipped')
    bam_check_proper_pair = remixt.config.get_param(config, 'bam_check_proper_pair')

    workflow = pypeliner.workflow.Workflow()

    workflow.setobj(obj=mgd.OutputChunks('chromosome'), value=chromosomes)

    if no_parallelism:
        workflow.transform(
            name='create_seqdata',
            ctx={'mem': 16},
            func='remixt.seqdataio.create_seqdata',
            args=(
                mgd.OutputFile(seqdata_filename),
                mgd.InputFile(bam_filename, extensions=['.bai']),
                snp_positions_filename,
                bam_max_fragment_length,
                bam_max_soft_clipped,
                bam_check_proper_pair,
                mgd.TempSpace('create_seqdata_temp'),
                chromosomes
            )
        )
    else:
        workflow.transform(
            name='create_chromosome_seqdata',
            axes=('chromosome',),
            ctx={'mem': 16},
            func='remixt.seqdataio.create_chromosome_seqdata',
            args=(
                mgd.TempOutputFile('seqdata', 'chromosome'),
                mgd.InputFile(bam_filename, extensions=['.bai']),
                snp_positions_filename,
                mgd.InputInstance('chromosome'),
                bam_max_fragment_length,
                bam_max_soft_clipped,
                bam_check_proper_pair,
            ),
        )

        workflow.transform(
            name='merge_seqdata',
            ctx={'mem': 16},
            func='remixt.seqdataio.merge_seqdata',
            args=(
                mgd.OutputFile(seqdata_filename),
                mgd.TempInputFile('seqdata', 'chromosome'),
            ),
        )

    return workflow


def create_infer_haps_workflow(
    seqdata_filenames,
    haps_filename,
    config,
    ref_data_dir,
    normal_id=None,
):
    chromosomes = remixt.config.get_chromosomes(config, ref_data_dir)

    workflow = pypeliner.workflow.Workflow()

    workflow.setobj(obj=mgd.OutputChunks('chromosome'), value=chromosomes)

    if normal_id is not None:
        normal_seqdata_filename = seqdata_filenames[normal_id]
        
        workflow.transform(
            name='infer_snp_genotype_from_normal',
            axes=('chromosome',),
            ctx={'mem': 16},
            func='remixt.analysis.haplotype.infer_snp_genotype_from_normal',
            args=(
                mgd.TempOutputFile('snp_genotype.tsv', 'chromosome'),
                mgd.InputFile(normal_seqdata_filename),
                mgd.InputInstance('chromosome'),
                config,
            ),
        )
    
    else:
        workflow.setobj(
            obj=mgd.OutputChunks('tumour_id'),
            value=list(seqdata_filenames.keys()),
        )

        workflow.transform(
            name='infer_snp_genotype_from_tumour',
            axes=('chromosome',),
            ctx={'mem': 16},
            func='remixt.analysis.haplotype.infer_snp_genotype_from_tumour',
            args=(
                mgd.TempOutputFile('snp_genotype.tsv', 'chromosome'),
                mgd.InputFile('tumour_seqdata', 'tumour_id', fnames=seqdata_filenames),
                mgd.InputInstance('chromosome'),
                config,
            ),
        )

    workflow.transform(
        name='infer_haps',
        axes=('chromosome',),
        ctx={'mem': 16},
        func='remixt.analysis.haplotype.infer_haps',
        args=(
            mgd.TempOutputFile('haps.tsv', 'chromosome'),
            mgd.TempInputFile('snp_genotype.tsv', 'chromosome'),
            mgd.InputInstance('chromosome'),
            mgd.TempSpace('haplotyping', 'chromosome'),
            config,
            ref_data_dir,
        )
    )

    workflow.transform(
        name='merge_haps',
        ctx={'mem': 16},
        func='remixt.utils.merge_tables',
        args=(
            mgd.OutputFile(haps_filename),
            mgd.TempInputFile('haps.tsv', 'chromosome'),
        )
    )

    return workflow


def create_calc_bias_workflow(
    tumour_seqdata_filename,
    segment_filename,
    segment_length_filename,
    config,
    ref_data_dir,
):
    workflow = pypeliner.workflow.Workflow(default_ctx={'mem': 4})

    workflow.transform(
        name='calc_fragment_stats',
        ctx={'mem': 16},
        func='remixt.analysis.stats.calculate_fragment_stats',
        ret=mgd.TempOutputObj('fragstats'),
        args=(
            mgd.InputFile(tumour_seqdata_filename),
            config,
        )
    )

    workflow.transform(
        name='sample_gc',
        ctx={'mem': 16},
        func='remixt.analysis.gcbias.sample_gc',
        args=(
            mgd.TempOutputFile('gcsamples.tsv'),
            mgd.InputFile(tumour_seqdata_filename),
            mgd.TempInputObj('fragstats').prop('fragment_mean'),
            config,
            ref_data_dir,
        )
    )

    workflow.transform(
        name='gc_lowess',
        ctx={'mem': 16},
        func='remixt.analysis.gcbias.gc_lowess',
        args=(
            mgd.TempInputFile('gcsamples.tsv'),
            mgd.TempOutputFile('gcloess.tsv'),
            mgd.TempOutputFile('gctable.tsv'),
        )
    )

    workflow.transform(
        name='split_segments',
        func='remixt.utils.split_table',
        args=(
            mgd.TempOutputFile('segments.tsv', 'segment_rows_idx'),
            mgd.InputFile(segment_filename),
            100,
        ),
    )

    workflow.transform(
        name='gc_map_bias',
        axes=('segment_rows_idx',),
        ctx={'mem': 16},
        func='remixt.analysis.gcbias.gc_map_bias',
        args=(
            mgd.TempInputFile('segments.tsv', 'segment_rows_idx'),
            mgd.TempInputObj('fragstats').prop('fragment_mean'),
            mgd.TempInputObj('fragstats').prop('fragment_stddev'),
            mgd.TempInputFile('gcloess.tsv'),
            mgd.TempOutputFile('biases.tsv', 'segment_rows_idx'),
            config,
            ref_data_dir,
        )
    )

    workflow.transform(
        name='merge_biases',
        func='remixt.utils.merge_tables',
        args=(
            mgd.TempOutputFile('biases.tsv'),
            mgd.TempInputFile('biases.tsv', 'segment_rows_idx'),
        ),
    )

    workflow.transform(
        name='biased_length',
        func='remixt.analysis.gcbias.biased_length',
        args=(
            mgd.OutputFile(segment_length_filename),
            mgd.TempInputFile('biases.tsv'),
        ),
    )

    return workflow


def create_prepare_counts_workflow(
    segment_filename,
    haplotypes_filename,
    tumour_filenames,
    count_filenames,
    config,
):
    count_filenames = dict([(tumour_id, count_filenames[tumour_id]) for tumour_id in tumour_filenames.keys()])

    workflow = pypeliner.workflow.Workflow()

    workflow.setobj(
        obj=mgd.OutputChunks('tumour_id'),
        value=list(tumour_filenames.keys()),
    )

    workflow.transform(
        name='segment_readcount',
        axes=('tumour_id',),
        ctx={'mem': 20},
        func='remixt.analysis.readcount.segment_readcount',
        args=(
            mgd.TempOutputFile('segment_counts.tsv', 'tumour_id'),
            mgd.InputFile(segment_filename),
            mgd.InputFile('tumour_file', 'tumour_id', fnames=tumour_filenames),
            config,
        ),
    )

    workflow.transform(
        name='haplotype_allele_readcount',
        axes=('tumour_id',),
        ctx={'mem': 20},
        func='remixt.analysis.readcount.haplotype_allele_readcount',
        args=(
            mgd.TempOutputFile('allele_counts.tsv', 'tumour_id'),
            mgd.InputFile(segment_filename),
            mgd.InputFile('tumour_file', 'tumour_id', fnames=tumour_filenames),
            mgd.InputFile(haplotypes_filename),
            config,
        ),
    )

    workflow.transform(
        name='phase_segments',
        ctx={'mem': 16},
        func='remixt.analysis.readcount.phase_segments',
        args=(
            mgd.TempInputFile('allele_counts.tsv', 'tumour_id'),
            mgd.TempOutputFile('phased_allele_counts.tsv', 'tumour_id', axes_origin=[]),
        ),
    )

    workflow.transform(
        name='prepare_readcount_table',
        axes=('tumour_id',),
        ctx={'mem': 16},
        func='remixt.analysis.readcount.prepare_readcount_table',
        args=(
            mgd.TempInputFile('segment_counts.tsv', 'tumour_id'),
            mgd.TempInputFile('phased_allele_counts.tsv', 'tumour_id'),
            mgd.OutputFile('count_file', 'tumour_id', fnames=count_filenames),
        ),
    )

    return workflow


def create_fit_model_workflow(
    experiment_filename,
    results_filename,
    config,
    ref_data_dir,
    tumour_id=None,
):
    config = remixt.config.get_sample_config(config, tumour_id)
    
    workflow = pypeliner.workflow.Workflow(default_ctx={'mem': 16})

    workflow.transform(
        name='init',
        func='remixt.analysis.pipeline.init',
        ret=mgd.TempOutputObj('init_params', 'init_id'),
        args=(
            mgd.TempOutputFile('init_results'),
            mgd.InputFile(experiment_filename),
            config,
        ),
    )

    workflow.transform(
        name='fit',
        axes=('init_id',),
        func='remixt.analysis.pipeline.fit_task',
        args=(
            mgd.TempOutputFile('fit_results', 'init_id'),
            mgd.InputFile(experiment_filename),
            mgd.TempInputObj('init_params', 'init_id'),
            config,
        ),
    )

    workflow.transform(
        name='collate',
        func='remixt.analysis.pipeline.collate',
        args=(
            mgd.OutputFile(results_filename),
            mgd.InputFile(experiment_filename),
            mgd.TempInputFile('init_results'),
            mgd.TempInputFile('fit_results', 'init_id'),
            config,
        ),
    )

    return workflow


def create_remixt_seqdata_workflow(
    breakpoint_filename,
    seqdata_filenames,
    results_filenames,
    raw_data_directory,
    config,
    ref_data_dir,
    normal_id=None,
):
    sample_ids = list(seqdata_filenames.keys())
    
    tumour_ids = list(seqdata_filenames.keys())
    if normal_id is not None:
        tumour_ids.remove(normal_id)

    results_filenames = dict([(tumour_id, results_filenames[tumour_id]) for tumour_id in tumour_ids])

    segment_filename = os.path.join(raw_data_directory, 'segments.tsv')
    haplotypes_filename = os.path.join(raw_data_directory, 'haplotypes.tsv')
    counts_table_template = os.path.join(raw_data_directory, 'counts', 'sample_{tumour_id}.tsv')
    experiment_template = os.path.join(raw_data_directory, 'experiment', 'sample_{tumour_id}.pickle')
    ploidy_plots_template = os.path.join(raw_data_directory, 'ploidy_plots', 'sample_{tumour_id}.pdf')

    workflow = pypeliner.workflow.Workflow()

    workflow.setobj(
        obj=mgd.OutputChunks('sample_id'),
        value=sample_ids,
    )

    workflow.setobj(
        obj=mgd.OutputChunks('tumour_id'),
        value=tumour_ids,
    )

    workflow.transform(
        name='create_segments',
        func='remixt.analysis.segment.create_segments',
        args=(
            mgd.OutputFile(segment_filename),
            config,
            ref_data_dir,
        ),
        kwargs={
            'breakpoint_filename': mgd.InputFile(breakpoint_filename),
        },
    )

    workflow.subworkflow(
        name='infer_haps_workflow',
        func='remixt.workflow.create_infer_haps_workflow',
        args=(
            mgd.InputFile('seqdata', 'sample_id', fnames=seqdata_filenames),
            mgd.OutputFile(haplotypes_filename),
            config,
            ref_data_dir,
        ),
        kwargs={
            'normal_id': normal_id,
        }
    )

    workflow.subworkflow(
        name='prepare_counts_workflow',
        func='remixt.workflow.create_prepare_counts_workflow',
        args=(
            mgd.InputFile(segment_filename),
            mgd.InputFile(haplotypes_filename),
            mgd.InputFile('seqdata', 'tumour_id', fnames=seqdata_filenames),
            mgd.TempOutputFile('rawcounts', 'tumour_id', axes_origin=[]),
            config,
        ),
    )

    workflow.subworkflow(
        name='calc_bias_workflow',
        axes=('tumour_id',),
        func='remixt.workflow.create_calc_bias_workflow',
        args=(
            mgd.InputFile('seqdata', 'tumour_id', fnames=seqdata_filenames),
            mgd.TempInputFile('rawcounts', 'tumour_id'),
            mgd.OutputFile('counts', 'tumour_id', template=counts_table_template),
            config,
            ref_data_dir,
        ),
    )

    workflow.transform(
        name='create_experiment',
        axes=('tumour_id',),
        ctx={'mem': 8},
        func='remixt.analysis.experiment.create_experiment',
        args=(
            mgd.InputFile('counts', 'tumour_id', template=counts_table_template),
            mgd.InputFile(breakpoint_filename),
            mgd.OutputFile('experiment', 'tumour_id', template=experiment_template),
        ),
    )

    workflow.transform(
        name='ploidy_analysis_plots',
        axes=('tumour_id',),
        ctx={'mem': 8},
        func='remixt.cn_plot.ploidy_analysis_plots',
        args=(
            mgd.InputFile('experiment', 'tumour_id', template=experiment_template),
            mgd.OutputFile('plots', 'tumour_id', template=ploidy_plots_template),
        ),
    )

    workflow.subworkflow(
        name='fit_model',
        axes=('tumour_id',),
        func='remixt.workflow.create_fit_model_workflow',
        args=(
            mgd.InputFile('experiment', 'tumour_id', template=experiment_template),
            mgd.OutputFile('results', 'tumour_id', fnames=results_filenames),
            config,
            ref_data_dir,
        ),
        kwargs={
            'tumour_id': mgd.InputInstance('tumour_id'),
        },
    )

    return workflow


def create_remixt_bam_workflow(
    breakpoint_filename,
    bam_filenames,
    results_filenames,
    raw_data_directory,
    config,
    ref_data_dir,
    normal_id=None,
):
    sample_ids = list(bam_filenames.keys())
    
    tumour_ids = list(bam_filenames.keys())
    if normal_id is not None:
        tumour_ids.remove(normal_id)

    seqdata_template = os.path.join(raw_data_directory, 'seqdata', 'sample_{sample_id}.h5')

    results_filenames = dict([(tumour_id, results_filenames[tumour_id]) for tumour_id in tumour_ids])

    workflow = pypeliner.workflow.Workflow()

    workflow.setobj(
        obj=mgd.OutputChunks('sample_id'),
        value=sample_ids,
    )

    workflow.setobj(
        obj=mgd.OutputChunks('tumour_id'),
        value=tumour_ids,
    )

    workflow.subworkflow(
        name='extract_seqdata_workflow',
        axes=('sample_id',),
        func='remixt.workflow.create_extract_seqdata_workflow',
        args=(
            mgd.InputFile('bam', 'sample_id', fnames=bam_filenames),
            mgd.OutputFile('seqdata', 'sample_id', template=seqdata_template),
            config,
            ref_data_dir,
        ),
    )

    workflow.subworkflow(
        name='remixt_seqdata_workflow',
        func='remixt.workflow.create_remixt_seqdata_workflow',
        args=(
            mgd.InputFile(breakpoint_filename),
            mgd.InputFile('seqdata', 'sample_id', template=seqdata_template),
            mgd.OutputFile('results', 'tumour_id', fnames=results_filenames, axes_origin=[]),
            raw_data_directory,
            config,
            ref_data_dir,
        ),
        kwargs={
            'normal_id': normal_id,
        },
    )

    return workflow
