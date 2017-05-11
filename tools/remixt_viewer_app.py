import argparse
import logging

import remixt.visualize

import warnings
warnings.filterwarnings('error')
logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('results',
        help='Results to visualize')

    argparser.add_argument('html',
        help='HTML output visualization')

    args = vars(argparser.parse_args())

    remixt.visualize.create_solutions_visualization(args['results'], args['html'])

