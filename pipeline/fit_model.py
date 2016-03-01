import argparse

import pypeliner
import pypeliner.managed as mgd

import remixt
import remixt.workflow


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('counts',
        help='Input segment counts filename')

    argparser.add_argument('breakpoints',
        help='Input breakpoints filename')

    argparser.add_argument('results',
        help='Output results filename')

    argparser.add_argument('--experiment',
        help='Debug output experiment pickle')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    config = {}
    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([remixt], config)

    if args['experiment'] is None:
        experiment_file = mgd.TempFile('experiment.pickle')
    else:
        experiment_file = mgd.File(args['experiment'])

    workflow = pypeliner.workflow.Workflow()

    workflow.transform(
        name='create_experiment',
        ctx={'mem': 8},
        func=remixt.analysis.experiment.create_experiment,
        args=(
            mgd.InputFile(args['counts']),
            mgd.InputFile(args['breakpoints']),
            experiment_file.as_output(),
        ),
    )

    workflow.subworkflow(
        name='run_remixt',
        func=remixt.workflow.create_fit_model_workflow,
        args=(
            experiment_file.as_input(),
            mgd.OutputFile(args['results']),
            config,
            args['ref_data_dir'],
        ),
    )

    pyp.run(workflow)

