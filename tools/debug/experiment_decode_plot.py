import argparse
import collections
import os
import pickle
import numpy as np

import demix.cn_plot

argparser = argparse.ArgumentParser()

argparser.add_argument('experiment',
                       help='Input pickled experiment filename')

argparser.add_argument('model',
                       help='Input pickled model filename')

argparser.add_argument('plotpdf',
                       help='Output plot pdf filename')

argparser.add_argument('--hapdepth', nargs='+', type=float,
                       help='Input haploid read depths')

args = vars(argparser.parse_args())

with open(args['experiment'], 'r') as experiment_file:
    experiment = pickle.load(experiment_file)

with open(args['model'], 'r') as model_file:
    model = pickle.load(model_file)

if args['hapdepth'] is None:
    h = experiment.h
else:
    h = np.array(args['hapdepth'])

cn, brk_cn = model.decode(experiment.x, experiment.l, h)

fig = demix.cn_plot.experiment_plot(experiment, cn=cn, h=h, p=model.p)

fig.savefig(args['plotpdf'], format='pdf', bbox_inches='tight', dpi=300)
