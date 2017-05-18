import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

import remixt.cn_plot
import remixt.analysis.readdepth


plot_choices = [
    'depth',
    'scatter',
    'raw',
]

def create_plot(**args):
    store = pd.HDFStore(args['results'], 'r')

    seaborn.set_style('ticks')

    if args['plot_type'] == 'depth':
        read_depth = store['/read_depth']
        minor_modes = store['/minor_modes']

        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()

        remixt.cn_plot.plot_depth(ax, read_depth, minor_modes=minor_modes)
        seaborn.despine(trim=True)

        fig.savefig(args['plot_file'], bbox_inches='tight')

    elif args['plot_type'] == 'scatter':
        cnv = store['/cn']

        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()

        remixt.cn_plot.plot_cnv_scatter(ax, cnv, major_col='major_raw', minor_col='minor_raw')

        fig.savefig(args['plot_file'], bbox_inches='tight')

    elif args['plot_type'] == 'raw':
        cnv = store['/cn']

        cnv['actual_length'] = cnv['end'] - cnv['start']

        cnv = cnv[
            (cnv['length'] > 1e5) &
            (cnv['length'] > 0.75 * cnv['actual_length'])]

        fig = plt.figure(figsize=(12, 2))
        ax = plt.gca()

        remixt.cn_plot.plot_cnv_genome(ax, cnv, major_col='major_raw', minor_col='minor_raw', maxcopies=6)

        fig.savefig(args['plot_file'], bbox_inches='tight')


def add_arguments(argparser):
    argparser.add_argument('results',
        help='Results to visualize')

    argparser.add_argument('plot_file',
        help='Output plot filename')

    argparser.add_argument('plot_type',
        help='Output plot type',
        choices=plot_choices)

    argparser.set_defaults(func=create_plot)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    add_arguments(argparser)

    args = vars(argparser.parse_args())
    func = args.pop('func')
    func(**args)


