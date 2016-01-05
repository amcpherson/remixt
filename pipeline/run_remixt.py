import os
import sys
import itertools
import argparse
import pickle
import collections
import pandas as pd
import numpy as np

import pypeliner
import pypeliner.workflow
import pypeliner.managed as mgd

import remixt
import remixt.analysis.pipeline


remixt_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
data_directory = os.path.join(remixt_directory, 'data')
default_cn_proportions_filename = os.path.join(data_directory, 'cn_proportions.tsv')


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

    argparser.add_argument('--cn_proportions',
        default=default_cn_proportions_filename,
        help='Number of clones')

    argparser.add_argument('--experiment',
        help='Debug output experiment pickle')

    args = vars(argparser.parse_args())

    pyp = pypeliner.app.Pypeline([remixt], args)

    workflow = pypeliner.workflow.Workflow(default_ctx={'mem':8})

    if args['experiment'] is None:
        experiment_file = mgd.TempFile('experiment.pickle')
    else:
        experiment_file = mgd.File(args['experiment'])

    workflow.transform(
        name='init',
        func=remixt.analysis.pipeline.init,
        args=(
            experiment_file.as_output(),
            mgd.TempOutputFile('h_init', 'byh'),
            mgd.TempOutputFile('init_results'),
            mgd.InputFile(args['counts']),
            mgd.InputFile(args['breakpoints']),
        ),
        kwargs={
            'num_clones':args['num_clones'],
        }
    )

    workflow.transform(
        name='fit',
        axes=('byh',),
        func=remixt.analysis.pipeline.fit,
        args=(
            mgd.TempOutputFile('fit_results', 'byh'),
            experiment_file.as_input(),
            mgd.TempInputFile('h_init', 'byh'),
            args['cn_proportions'],
        ),
    )

    workflow.transform(
        name='collate',
        func=remixt.analysis.pipeline.collate,
        args=(
            mgd.OutputFile(args['results']),
            mgd.InputFile(args['breakpoints']),
            experiment_file.as_input(),
            mgd.TempInputFile('init_results'),
            mgd.TempInputFile('fit_results', 'byh'),
        ),
    )

    pyp.run(workflow)




