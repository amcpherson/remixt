import pypeliner
import pypeliner.managed as mgd

import remixt.config
import remixt.analysis.gcbias
import remixt.analysis.pipeline
import remixt.analysis.stats
import remixt.seqdataio
import remixt.utils


def create_extract_seqdata_workflow(
     bam_filename,
     seqdata_filename,
     config,
):
    chromosomes = remixt.config.get_param(config, 'chromosomes')
    snp_positions_filename = remixt.config.get_filename(config, 'snp_positions')

    bam_max_fragment_length = remixt.config.get_param(config, 'bam_max_fragment_length')
    bam_max_soft_clipped = remixt.config.get_param(config, 'bam_max_soft_clipped')

    workflow = pypeliner.workflow.Workflow()

    workflow.setobj(obj=mgd.OutputChunks('chromosome'), value=chromosomes)

    workflow.transform(
        name='create_chromosome_seqdata',
        axes=('chromosome',),
        ctx={'mem': 16},
        func=remixt.seqdataio.create_chromosome_seqdata,
        args=(
            mgd.TempOutputFile('seqdata', 'chromosome'),
            mgd.InputFile(bam_filename),
            mgd.InputFile(snp_positions_filename),
            mgd.InputInstance('chromosome'),
            bam_max_fragment_length,
            bam_max_soft_clipped,
        ),
    )

    workflow.transform(
        name='merge_seqdata',
        ctx={'mem': 4},
        func=remixt.seqdataio.merge_seqdata,
        args=(
            mgd.OutputFile(seqdata_filename),
            mgd.TempInputFile('seqdata', 'chromosome'),
        ),
    )

    return workflow


def create_infer_haps_workflow(
    normal_seqdata_filename,
    haps_filename,
    config,
):
    chromosomes = remixt.config.get_param(config, 'chromosomes')

    workflow = pypeliner.workflow.Workflow()

    workflow.setobj(obj=mgd.OutputChunks('chromosome'), value=chromosomes)

    workflow.transform(
        name='infer_haps',
        axes=('chromosome',),
        ctx={'mem': 16},
        func=remixt.analysis.haplotype.infer_haps,
        args=(
            mgd.TempOutputFile('haps.tsv', 'chromosome'),
            mgd.InputFile(normal_seqdata_filename),
            mgd.InputInstance('chromosome'),
            mgd.TempSpace('haplotyping', 'chromosome'),
            config,
        )
    )

    workflow.transform(
        name='merge_haps',
        ctx={'mem': 16},
        func=remixt.utils.merge_tables,
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
):
    workflow = pypeliner.workflow.Workflow(default_ctx={'mem': 4})

    workflow.transform(
        name='calc_fragment_stats',
        ctx={'mem': 16},
        func=remixt.analysis.stats.calculate_fragment_stats,
        ret=mgd.TempOutputObj('fragstats'),
        args=(mgd.InputFile(tumour_seqdata_filename),)
    )

    workflow.transform(
        name='sample_gc',
        ctx={'mem': 16},
        func=remixt.analysis.gcbias.sample_gc,
        args=(
            mgd.TempOutputFile('gcsamples.tsv'),
            mgd.InputFile(tumour_seqdata_filename),
            mgd.TempInputObj('fragstats').prop('fragment_mean'),
            config,
        )
    )

    workflow.transform(
        name='gc_lowess',
        ctx={'mem': 16},
        func=remixt.analysis.gcbias.gc_lowess,
        args=(
            mgd.TempInputFile('gcsamples.tsv'),
            mgd.TempOutputFile('gcloess.tsv'),
            mgd.TempOutputFile('gctable.tsv'),
        )
    )

    workflow.transform(
        name='split_segments',
        func=remixt.utils.split_table,
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
        func=remixt.analysis.gcbias.gc_map_bias,
        args=(
            mgd.TempInputFile('segments.tsv', 'segment_rows_idx'),
            mgd.TempInputObj('fragstats').prop('fragment_mean'),
            mgd.TempInputObj('fragstats').prop('fragment_stddev'),
            mgd.TempInputFile('gcloess.tsv'),
            mgd.TempOutputFile('biases.tsv', 'segment_rows_idx'),
            config,
        )
    )

    workflow.transform(
        name='merge_biases',
        func=remixt.utils.merge_tables,
        args=(
            mgd.TempOutputFile('biases.tsv'),
            mgd.TempInputFile('biases.tsv', 'segment_rows_idx'),
        ),
    )

    workflow.transform(
        name='biased_length',
        func=remixt.analysis.gcbias.biased_length,
        args=(
            mgd.OutputFile(segment_length_filename),
            mgd.TempInputFile('biases.tsv'),
        ),
    )

    return workflow


def create_remixt_workflow(
    experiment_filename,
    results_filename,
    fit_method='hmm_graph',
    cn_proportions_filename=None,
    num_clones=None,
):
    workflow = pypeliner.workflow.Workflow(default_ctx={'mem': 8})

    workflow.transform(
        name='init',
        func=remixt.analysis.pipeline.init,
        args=(
            mgd.TempOutputFile('h_init', 'byh'),
            mgd.TempOutputFile('init_results'),
            mgd.InputFile(experiment_filename),
        ),
        kwargs={
            'num_clones': num_clones,
        }
    )

    workflow.transform(
        name='fit',
        axes=('byh',),
        func=remixt.analysis.pipeline.fit,
        args=(
            mgd.TempOutputFile('fit_results', 'byh'),
            mgd.InputFile(experiment_filename),
            mgd.TempInputFile('h_init', 'byh'),
        ),
        kwargs={
            'fit_method': fit_method,
            'cn_proportions_filename': cn_proportions_filename,
        }
    )

    workflow.transform(
        name='collate',
        func=remixt.analysis.pipeline.collate,
        args=(
            mgd.OutputFile(results_filename),
            mgd.InputFile(experiment_filename),
            mgd.TempInputFile('init_results'),
            mgd.TempInputFile('fit_results', 'byh'),
        ),
    )

    return workflow
