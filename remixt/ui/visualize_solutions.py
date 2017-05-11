import argparse
import yaml
import pandas as pd
import numpy as np

import remixt.visualize


def create_visualization(**args):
    remixt.visualize.create_solutions_visualization(args['results'], args['html'])


def add_arguments(argparser):
    argparser.add_argument('results',
        help='Results to visualize')

    argparser.add_argument('html',
        help='HTML output visualization')

    argparser.set_defaults(func=create_visualization)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    add_arguments(argparser)

    args = vars(argparser.parse_args())
    func = args.pop('func')
    func(**args)


