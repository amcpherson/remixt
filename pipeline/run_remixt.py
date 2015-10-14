import os
import sys
import itertools
import argparse
import pickle
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pypeliner
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

    args = vars(argparser.parse_args())

    pyp = pypeliner.app.Pypeline([remixt], args)

    pyp.sch.transform('init', (), {'mem':8},
        remixt.analysis.pipeline.init,
        None,
        mgd.TempOutputFile('experiment.pickle'),
        mgd.TempOutputFile('h_init', 'byh'),
        mgd.TempOutputFile('init_results'),
        mgd.InputFile(args['counts']),
        mgd.InputFile(args['breakpoints']),
        num_clones=args['num_clones'],
    )

    pyp.sch.transform('fit', ('byh',), {'mem':8},
        remixt.analysis.pipeline.fit,
        None,
        mgd.TempOutputFile('fit_results', 'byh'),
        mgd.TempInputFile('experiment.pickle'),
        mgd.TempInputFile('h_init', 'byh'),
        args['cn_proportions'],
    )

    pyp.sch.transform('collate', (), {'mem':1},
        remixt.analysis.pipeline.collate,
        None,
        mgd.OutputFile(args['results']),
        mgd.InputFile(args['breakpoints']),
        mgd.TempInputFile('experiment.pickle'),
        mgd.TempInputFile('init_results'),
        mgd.TempInputFile('fit_results', 'byh'),
    )

    pyp.run()




