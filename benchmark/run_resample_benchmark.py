import argparse
import yaml

import pypeliner
import pypeliner.managed as mgd

import remixt.simulations.pipeline
import remixt.simulations.workflow

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

    argparser.add_argument('table',
        help='Output Table Filename')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    config = {}
    if args['config'] is not None:
        execfile(args['config'], {}, config)

    yaml_text = open(args['tool_defs']).read().format(ref_data_dir=args['ref_data_dir'])
    tool_defs = yaml.load(yaml_text)
    del tool_defs['databases']

    pyp = pypeliner.app.Pypeline(config=args)

    workflow = pypeliner.workflow.Workflow(default_ctx={'mem': 4})

    workflow.transform(
        name='read_sim_defs',
        ctx={'local': True},
        func=remixt.simulations.pipeline.create_simulations,
        ret=mgd.TempOutputObj('sim_defs', 'sim_id'),
        args=(
            mgd.InputFile(args['sim_defs']),
            config,
            args['ref_data_dir'],
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
            mgd.TempOutputFile('normal_seqdata', 'sim_id'),
            mgd.TempOutputFile('tumour_seqdata', 'sim_id'),
            mgd.TempOutputFile('genome_mixture', 'sim_id'),
            mgd.TempOutputFile('breakpoints', 'sim_id'),
            config,
            args['ref_data_dir'],
        ),
    )

    workflow.subworkflow(
        name='create_tool_workflow',
        axes=('sim_id', 'tool_name'),
        func=remixt.simulations.pipeline.create_tool_workflow,
        args=(
            mgd.TempInputObj('tool_defs', 'sim_id', 'tool_name'),
            mgd.TempInputFile('normal_seqdata', 'sim_id'),
            mgd.TempInputFile('tumour_seqdata', 'sim_id'),
            mgd.TempInputFile('breakpoints', 'sim_id'),
            mgd.TempOutputFile('results', 'sim_id', 'tool_name'),
            mgd.TempSpace('raw_data', 'sim_id', 'tool_name', cleanup=None),
        ),
    )

    workflow.transform(
        name='evaluate_results',
        axes=('sim_id', 'tool_name'),
        func=remixt.simulations.pipeline.evaluate_results_task,
        args=(
            mgd.TempOutputFile('evaluation', 'sim_id', 'tool_name'),
            mgd.TempInputFile('results', 'sim_id', 'tool_name'),
        ),
        kwargs={
            'mixture_filename': mgd.TempInputFile('genome_mixture', 'sim_id'),
            'key_prefix': '/sample_tumour',
        },
    )

    workflow.transform(
        name='merge_evaluations',
        func=remixt.simulations.pipeline.merge_evaluations,
        args=(
            mgd.OutputFile(args['table']),
            mgd.TempInputObj('sim_defs', 'sim_id'),
            mgd.TempInputFile('evaluation', 'sim_id', 'tool_name'),
            ['sim_id', 'tool_name'],
        ),
    )

    pyp.run(workflow)

