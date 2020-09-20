
import sys
import argparse
import logging

import remixt.visualize

import warnings
# warnings.filterwarnings('error')
logging.basicConfig(level=logging.DEBUG)

# import pandas as pd
# print(pd.HDFStore('results_T2-T-A.h5', 'r')['/solutions/solution_14/mix'])


if __name__ == '__main__':
    LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(lineno)d - %(message)s"
    logging.basicConfig(format=LOGGING_FORMAT, stream=sys.stderr, level=logging.INFO)

    argparser = argparse.ArgumentParser()

    argparser.add_argument('results',
        help='Results to visualize')

    argparser.add_argument('html',
        help='HTML output visualization')

    args = vars(argparser.parse_args())

    remixt.visualize.create_solutions_visualization(args['results'], args['html'])

