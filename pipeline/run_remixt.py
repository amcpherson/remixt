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

import demix
import demix.analysis.pipeline


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('counts',
        help='Input segment counts filename')

    argparser.add_argument('breakpoints',
        help='Input breakpoints filename')

    argparser.add_argument('cn',
        help='Output segment copy number filename')

    argparser.add_argument('brk_cn',
        help='Output breakpoint copy number filename')

    argparser.add_argument('cn_plot',
        help='Output segment copy number plot pdf filename')

    argparser.add_argument('mix',
        help='Output mixture filename')

    args = vars(argparser.parse_args())

    pyp = pypeliner.app.Pypeline([demix], args)

    pyp.sch.transform('init', (), {'mem':8},
        demix.analysis.pipeline.init,
        None,
        mgd.TempOutputFile('experiment_learn.pickle'),
        mgd.TempOutputFile('model_learn.pickle'),
        mgd.TempOutputFile('experiment_infer.pickle'),
        mgd.TempOutputFile('model_infer.pickle'),
        mgd.TempOutputFile('h_init', 'byh'),
        mgd.TempOutputFile('h_plot.pdf'),
        mgd.InputFile(args['counts']),
        mgd.InputFile(args['breakpoints']),
    )

    pyp.sch.transform('learn_h', ('byh',), {'mem':8},
        demix.analysis.pipeline.learn_h,
        None,
        mgd.TempOutputFile('h_opt', 'byh'),
        mgd.TempInputFile('experiment_learn.pickle'),
        mgd.TempInputFile('model_learn.pickle'),
        mgd.TempInputFile('h_init', 'byh'),
    )

    pyp.sch.transform('tabulate_h', (), {'mem':1},
        demix.analysis.pipeline.tabulate_h,
        None,
        mgd.TempOutputFile('h_table.tsv'),
        mgd.TempInputFile('h_opt', 'byh'),
    )

    pyp.sch.transform('infer_cn', (), {'mem':24},
        demix.analysis.pipeline.infer_cn,
        None,
        mgd.OutputFile(args['cn']),
        mgd.OutputFile(args['brk_cn']),
        mgd.OutputFile(args['cn_plot']),
        mgd.OutputFile(args['mix']),
        mgd.TempInputFile('experiment_infer.pickle'),
        mgd.TempInputFile('model_infer.pickle'),
        mgd.TempInputFile('h_table.tsv'),
        mgd.TempOutputFile('model_infer_debug.pickle'),
    )

    pyp.run()




