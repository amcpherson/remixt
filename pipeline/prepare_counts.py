import argparse

import pypeliner
import pypeliner.workflow
import pypeliner.managed as mgd

import remixt
import remixt.seqdataio
import remixt.segalg
import remixt.utils
import remixt.workflow
import remixt.analysis.haplotype
import remixt.analysis.segment
import remixt.analysis.gcbias
import remixt.analysis.stats
import remixt.analysis.readcount


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('segment_file',
        help='Input segments file')

    argparser.add_argument('normal_file',
        help='Input normal sequence data filename')

    argparser.add_argument('--tumour_files', nargs='+', required=True,
        help='Input tumour sequence data filenames')

    argparser.add_argument('--count_files', nargs='+', required=True,
        help='Output count TSV filenames')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    if len(args['tumour_files']) != len(args['count_files']):
        raise Exception('--count_files must correspond one to one with --tumour_files')

    config = {'ref_data_directory': args['ref_data_dir']}

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([remixt], config)

    workflow = pypeliner.workflow.Workflow()

    tumour_fnames = dict(enumerate(args['tumour_files']))
    count_fnames = dict(enumerate(args['count_files']))

    workflow.setobj(obj=mgd.OutputChunks('bytumour'), value=tumour_fnames.keys())

    workflow.subworkflow(
        name='infer_haps',
        func=remixt.workflow.create_infer_haps_workflow,
        args=(
            mgd.InputFile(args['normal_file']),
            mgd.TempOutputFile('haps.tsv'),
            config,
        ),
    )

    workflow.transform(
        name='segment_readcount',
        axes=('bytumour',),
        ctx={'mem': 16},
        func=remixt.analysis.readcount.segment_readcount,
        args=(
            mgd.TempOutputFile('segment_counts.tsv', 'bytumour'),
            mgd.InputFile(args['segment_file']),
            mgd.InputFile('tumour_file', 'bytumour', fnames=tumour_fnames),
        ),
    )

    workflow.transform(
        name='haplotype_allele_readcount',
        axes=('bytumour',),
        ctx={'mem': 16},
        func=remixt.analysis.readcount.haplotype_allele_readcount,
        args=(
            mgd.TempOutputFile('allele_counts.tsv', 'bytumour'),
            mgd.InputFile(args['segment_file']),
            mgd.InputFile('tumour_file', 'bytumour', fnames=tumour_fnames),
            mgd.TempInputFile('haps.tsv'),
        ),
    )

    workflow.transform(
        name='phase_segments',
        ctx={'mem': 16},
        func=remixt.analysis.readcount.phase_segments,
        args=(
            mgd.TempInputFile('allele_counts.tsv', 'bytumour'),
            mgd.TempOutputFile('phased_allele_counts.tsv', 'bytumour', axes_origin=[]),
        ),
    )

    workflow.subworkflow(
        name='calc_bias',
        axes=('bytumour',),
        func=remixt.workflow.create_calc_bias_workflow,
        args=(
            mgd.InputFile('tumour_file', 'bytumour', fnames=tumour_fnames),
            mgd.TempInputFile('segment_counts.tsv', 'bytumour'),
            mgd.TempOutputFile('segment_counts_lengths.tsv', 'bytumour'),
            config,
        ),
    )

    workflow.transform(
        name='prepare_readcount_table',
        axes=('bytumour',),
        ctx={'mem': 16},
        func=remixt.analysis.readcount.prepare_readcount_table,
        args=(
            mgd.TempInputFile('segment_counts_lengths.tsv', 'bytumour'),
            mgd.TempInputFile('phased_allele_counts.tsv', 'bytumour'),
            mgd.OutputFile('count_file', 'bytumour', fnames=count_fnames),
        ),
    )

    pyp.run(workflow)

