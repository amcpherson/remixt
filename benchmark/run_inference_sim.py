import argparse

import pypeliner
import pypeliner.managed as mgd

import remixt.simulations

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

    workflow = pypeliner.workflow.Workflow()

    workflow.transform(
        name='read_sim_defs',
        ctx={'mem': 1, 'local': True},
        func=remixt.simulations.read_sim_defs,
        ret=mgd.TempOutputObj('sim_defs', 'sim_id'),
        args=(mgd.InputFile(args['sim_defs']),),
    )

    workflow.subworkflow(
        name='simulate_experiment',
        axes=('sim_id',),
        func=remixt.simulations.workflow.create_simulate_experiment_workflow,
        args=(
            mgd.TempInputObj('sim_defs', 'sim_id'),
            mgd.TempOutputFile('experiment', 'sim_id'),
        ),
    )

    workflow.subworkflow(
        name='run_remixt',
        func=remixt.workflow.create_fit_model_workflow,
        args=(
            mgd.TempInputFile('experiment', 'sim_id'),
            mgd.TempOutputFile('results', 'sim_id'),
            config,
            args['ref_data_dir'],
        ),
    )

    workflow.transform(
        name='tabulate_results',
        ctx={'mem': 1},
        func=remixt.simulations.pipeline.tabulate_results,
        args=(
            mgd.OutputFile(args['table']),
            mgd.TempInputObj('sim_defs', 'sim_id'),
            mgd.TempInputFile('experiment', 'sim_id'),
            mgd.TempInputFile('results', 'sim_id'),
        ),
    )

    pyp.run(workflow)

