import argparse

import pypeliner
import pypeliner.managed as mgd

import remixt.simulations.pipeline
import remixt.simulations.workflow
import remixt.workflow

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('sim_defs',
        help='Simulation Definition Filename')

    argparser.add_argument('table',
        help='Output Table Filename')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    config = {}
    if args['config'] is not None:
        execfile(args['config'], {}, config)

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

    workflow.subworkflow(
        name='simulate_read_data',
        axes=('sim_id',),
        func=remixt.simulations.workflow.create_segment_simulation_workflow,
        args=(
            mgd.TempInputObj('sim_defs', 'sim_id'),
            mgd.TempOutputFile('experiment', 'sim_id'),
        ),
    )

    workflow.subworkflow(
        name='create_tool_workflow',
        axes=('sim_id',),
        func=remixt.workflow.create_fit_model_workflow,
        args=(
            mgd.TempInputFile('experiment', 'sim_id'),
            mgd.TempOutputFile('results', 'sim_id'),
            config,
            args['ref_data_dir'],
        ),
    )

    workflow.transform(
        name='evaluate_results',
        axes=('sim_id',),
        func=remixt.simulations.pipeline.evaluate_results_task,
        args=(
            mgd.TempOutputFile('evaluation', 'sim_id'),
            mgd.TempInputFile('results', 'sim_id'),
        ),
        kwargs={
            'experiment_filename': mgd.TempInputFile('experiment', 'sim_id'),
        },
    )

    workflow.transform(
        name='merge_evaluations',
        func=remixt.simulations.pipeline.merge_evaluations,
        args=(
            mgd.OutputFile(args['table']),
            mgd.TempInputObj('sim_defs', 'sim_id'),
            mgd.TempInputFile('evaluation', 'sim_id'),
            ['sim_id'],
        ),
    )

    pyp.run(workflow)

