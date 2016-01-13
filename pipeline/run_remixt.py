import argparse

import pypeliner
import pypeliner.managed as mgd

import remixt
import remixt.workflow


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('counts',
        help='Input segment counts filename')

    argparser.add_argument('breakpoints',
        help='Input breakpoints filename')

    argparser.add_argument('results',
        help='Output results filename')

    argparser.add_argument('--num_clones', type=int,
        help='Number of clones')

    argparser.add_argument('--fit_method',
        help='Method for learning copy number')

    argparser.add_argument('--cn_proportions',
        help='Copy number state proportions table for prior')

    argparser.add_argument('--experiment',
        help='Debug output experiment pickle')

    args = vars(argparser.parse_args())

    pyp = pypeliner.app.Pypeline([remixt], args)

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
        func=remixt.workflow.create_remixt_workflow,
        args=(
            experiment_file.as_input(),
            mgd.OutputFile(args['results']),
        ),
        kwargs={
            'fit_method': args['fit_method'],
            'cn_proportions_filename': args['cn_proportions'],
            'num_clones': args['num_clones'],
        },
    )

    pyp.run(workflow)

