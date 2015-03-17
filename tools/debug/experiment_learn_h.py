import argparse
import collections
import os
import pickle
import numpy as np

argparser = argparse.ArgumentParser()

argparser.add_argument('experiment',
                       help='Input pickled experiment filename')

argparser.add_argument('model',
                       help='Input pickled model filename')

argparser.add_argument('--hapdepth', nargs='+', type=float,
                       help='Input haploid read depths')

args = vars(argparser.parse_args())

with open(args['experiment'], 'r') as experiment_file:
    experiment = pickle.load(experiment_file)

with open(args['model'], 'r') as model_file:
    model = pickle.load(model_file)

if args['hapdepth'] is not None:
    h = np.array(args['hapdepth'])
elif args['experiment'] is not None:
    h = experiment.h

h, log_posterior, converged = model.optimize_h(experiment.x, experiment.l, h)

print 'log posterior = ', log_posterior
print 'h = ', repr(h)
print 'converged = ', repr(converged)

