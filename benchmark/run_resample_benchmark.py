import os
import argparse
import yaml

import pypeliner
import pypeliner.managed as mgd

import remixt.simulations.pipeline
import remixt.simulations.workflow
import remixt.cn_plot

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('sim_defs',
        help='Simulation Definition Filename')

    argparser.add_argument('tool_defs',
        help='Tool Definition Filename')

    argparser.add_argument('source',
        help='Input Source SeqData Filename')

    argparser.add_argument('raw_data_dir',
        help='Raw data directory')

    argparser.add_argument('table',
        help='Output Table Filename')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    config = {}
    if args['config'] is not None:
        execfile(args['config'], {}, config)

    yaml_text = open(args['tool_defs']).read().format(ref_data_dir=args['ref_data_dir'])
    tool_defs = yaml.load(yaml_text)['tools']

    remixt_ref_data_dir = tool_defs['remixt']['kwargs']['ref_data_dir']

    pyp = pypeliner.app.Pypeline(config=args)

    normal_seqdata_template = os.path.join(args['raw_data_dir'], '{sim_id}', 'normal.h5')
    tumour_seqdata_template = os.path.join(args['raw_data_dir'], '{sim_id}', 'tumour.h5')
    genome_mixture_template = os.path.join(args['raw_data_dir'], '{sim_id}', 'genome_mixture.pickle')
    genome_mixture_plot_template = os.path.join(args['raw_data_dir'], '{sim_id}', 'genome_mixture_plot.pdf')
    breakpoints_template = os.path.join(args['raw_data_dir'], '{sim_id}', 'breakpoints.tsv')
    results_template = os.path.join(args['raw_data_dir'], '{sim_id}', '{tool_name}', 'results.h5')
    evaluation_template = os.path.join(args['raw_data_dir'], '{sim_id}', '{tool_name}', 'evaluation.h5')

    workflow = pypeliner.workflow.Workflow(default_ctx={'mem': 4})

    workflow.transform(
        name='read_sim_defs',
        ctx={'local': True},
        func=remixt.simulations.pipeline.create_simulations,
        ret=mgd.TempOutputObj('sim_defs', 'sim_id'),
        args=(
            mgd.InputFile(args['sim_defs']),
            config,
            remixt_ref_data_dir,
        ),
    )

    workflow.setobj(
        obj=mgd.TempOutputObj('tool_defs', 'sim_id', 'tool_name'),
        value=tool_defs,
        axes=('sim_id',),
    )

    workflow.subworkflow(
        name='resample_read_data',
        axes=('sim_id',),
        func=remixt.simulations.workflow.create_resample_simulation_workflow,
        args=(
            mgd.TempInputObj('sim_defs', 'sim_id'),
            mgd.InputFile(args['source']),
            mgd.OutputFile('normal_seqdata', 'sim_id', template=normal_seqdata_template),
            mgd.OutputFile('tumour_seqdata', 'sim_id', template=tumour_seqdata_template),
            mgd.OutputFile('genome_mixture', 'sim_id', template=genome_mixture_template),
            mgd.OutputFile('genome_mixture_plot', 'sim_id', template=genome_mixture_plot_template),
            mgd.OutputFile('breakpoints', 'sim_id', template=breakpoints_template),
            config,
            remixt_ref_data_dir,
        ),
    )

    workflow.subworkflow(
        name='create_tool_workflow',
        axes=('sim_id', 'tool_name'),
        func=remixt.simulations.pipeline.create_tool_workflow,
        args=(
            mgd.TempInputObj('tool_defs', 'sim_id', 'tool_name'),
            mgd.InputFile('normal_seqdata', 'sim_id', template=normal_seqdata_template),
            mgd.InputFile('tumour_seqdata', 'sim_id', template=tumour_seqdata_template),
            mgd.InputFile('breakpoints', 'sim_id', template=breakpoints_template),
            mgd.OutputFile('results', 'sim_id', 'tool_name', template=results_template),
            mgd.TempSpace('raw_data', 'sim_id', 'tool_name', cleanup=None),
        ),
    )

    workflow.transform(
        name='evaluate_results',
        axes=('sim_id', 'tool_name'),
        func=remixt.simulations.pipeline.evaluate_results_task,
        args=(
            mgd.OutputFile('evaluation', 'sim_id', 'tool_name', template=evaluation_template),
            mgd.InputFile('results', 'sim_id', 'tool_name', template=results_template),
        ),
        kwargs={
            'mixture_filename': mgd.InputFile('genome_mixture', 'sim_id', template=genome_mixture_template),
            'key_prefix': '/sample_tumour',
        },
    )

    workflow.transform(
        name='merge_evaluations',
        func=remixt.simulations.pipeline.merge_evaluations,
        args=(
            mgd.OutputFile(args['table']),
            mgd.TempInputObj('sim_defs', 'sim_id'),
            mgd.InputFile('evaluation', 'sim_id', 'tool_name', template=evaluation_template),
            ['sim_id', 'tool_name'],
        ),
    )

    pyp.run(workflow)

