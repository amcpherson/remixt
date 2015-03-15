import sys
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter

import cn_model


def plot_cnv_segments(ax, cnv, major_col='major', minor_col='minor'):
    """
    Plot raw major/minor copy number as line plots

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): cnv table
        major_col (str): name of major copies column
        minor_col (str): name of minor copies column

    Plot major and minor copy number as line plots.  The columns 'start' and 'end'
    are expected and should be adjusted for full genome plots.  Values from the
    'major_col' and 'minor_col' columns are plotted.

    """ 

    color_major = plt.get_cmap('RdBu')(0.1)
    color_minor = plt.get_cmap('RdBu')(0.9)

    cnv = cnv.sort('start')

    def plot_segment(ax, row, field, color):
        ax.plot([row['start'], row['end']], [row[field]]*2, color=color, lw=1)

    def plot_connectors(ax, row, next_row, field, color):
        mid = (row[field] + next_row[field]) / 2.0
        ax.plot([row['end'], row['end']], [row[field], mid], color=color, lw=1)
        ax.plot([next_row['start'], next_row['start']], [mid, next_row[field]], color=color, lw=1)
    
    for (idx, row), (next_idx, next_row) in itertools.izip_longest(cnv.iterrows(), cnv.iloc[1:].iterrows(), fillvalue=(None, None)):
        plot_segment(ax, row, major_col, color_major)
        plot_segment(ax, row, minor_col, color_minor)
        if next_row is not None:
            plot_connectors(ax, row, next_row, major_col, color_major)
            plot_connectors(ax, row, next_row, minor_col, color_minor)


def plot_cnv_genome(ax, cnv, maxcopies=4, minlength=1000, major_col='major', minor_col='minor'):
    """
    Plot major/minor copy number across the genome

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): `cnv_site` table
        maxcopies (int): maximum number of copies for setting y limits
        minlength (int): minimum length of segments to be drawn
        major_col (str): name of major copies column
        minor_col (str): name of minor copies column

    """

    cnv = cnv[['chromosome', 'start', 'end', 'length', major_col, minor_col]].copy()

    chromosomes = cnv['chromosome'].unique()

    chromosome_length = cnv.groupby('chromosome')['end'].max()
    chromosome_length.sort(ascending=False)

    chromosomes = chromosome_length.index.values

    chromosome_end = np.cumsum(chromosome_length)
    chromosome_start = chromosome_end - chromosome_length
    chromosome_mid = (chromosome_start + chromosome_end) / 2.

    cnv.set_index('chromosome', inplace=True)
    cnv['chromosome_start'] = chromosome_start
    cnv.reset_index(inplace=True)

    cnv['start'] = cnv['start'] + cnv['chromosome_start']
    cnv['end'] = cnv['end'] + cnv['chromosome_start']

    plot_cnv_segments(ax, cnv, major_col=major_col, minor_col=minor_col)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylim((-0.05*maxcopies, maxcopies+.6))
    ax.set_xlim((-0.5, chromosome_end.max()))
    ax.set_xlabel('chromosome')
    ax.set_xticks([0] + list(chromosome_end.values))
    ax.set_xticklabels([])
    ax.set_yticks(range(0, maxcopies+1))
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(chromosome_mid))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(chromosomes))
    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')


def experiment_plot(data):
    """ Plot a sequencing experiment

    Args:
        data (pandas.DataFrame): dataframe with specific copy number columns for plotting

    Returns:
        matplotlib.Figure: figure object of plots

    """

    fig = plt.figure(figsize=(20, 10))

    ax = plt.subplot(5, 1, 1)

    plot_cnv_genome(ax, data, maxcopies=4, major_col='major_raw', minor_col='minor_raw')

    ax.set_xlabel('')
    ax.set_ylabel('raw')

    ax = plt.subplot(5, 1, 2)

    plot_cnv_genome(ax, data, maxcopies=4, major_col='major_raw_e', minor_col='minor_raw_e')

    ax.set_xlabel('')
    ax.set_ylabel('expected')

    ax = plt.subplot(5, 1, 3)

    plot_cnv_genome(ax, data, maxcopies=4, major_col='major_1', minor_col='minor_1')

    ax.set_xlabel('')
    ax.set_ylabel('clone 1')

    ax = plt.subplot(5, 1, 4)

    plot_cnv_genome(ax, data, maxcopies=4, major_col='major_2', minor_col='minor_2')

    ax.set_xlabel('')
    ax.set_ylabel('clone 2')

    ax = plt.subplot(5, 1, 5)

    plot_cnv_genome(ax, data, maxcopies=2, major_col='major_diff', minor_col='minor_diff')

    ax.set_xlabel('chromosome')
    ax.set_ylabel('clone diff')

    plt.tight_layout()

    return fig


def mixture_plot(mixture):
    """ Plot a genome mixture

    Args:
        mixture (GenomeMixture): information about the genomes and their proportions

    Returns:
        matplotlib.Figure: figure object of plots

    """

    data = pd.DataFrame({
            'chromosome':mixture.segment_chromosome_id,
            'start':mixture.segment_start,
            'end':mixture.segment_end,
            'length':mixture.l,
    })

    tumour_frac = mixture.frac[1:] / mixture.frac[1:].sum()

    data['major_expected'] = np.einsum('ij,j->i', mixture.cn[:,1:,0], tumour_frac)
    data['minor_expected'] = np.einsum('ij,j->i', mixture.cn[:,1:,1], tumour_frac)

    for m in xrange(1, mixture.cn.shape[1]):
        data['major_{0}'.format(m)] = mixture.cn[:,m,0]
        data['minor_{0}'.format(m)] = mixture.cn[:,m,1]

    data['major_diff'] = np.absolute(data['major_1'] - data['major_2'])
    data['minor_diff'] = np.absolute(data['minor_1'] - data['minor_2'])

    fig = plt.figure(figsize=(20, 10))

    ax = plt.subplot(4, 1, 1)

    plot_cnv_genome(ax, data, maxcopies=4, major_col='major_expected', minor_col='minor_expected')

    ax.set_xlabel('')
    ax.set_ylabel('expected')

    ax = plt.subplot(4, 1, 2)

    plot_cnv_genome(ax, data, maxcopies=4, major_col='major_1', minor_col='minor_1')

    ax.set_xlabel('')
    ax.set_ylabel('clone 1')

    ax = plt.subplot(4, 1, 3)

    plot_cnv_genome(ax, data, maxcopies=4, major_col='major_2', minor_col='minor_2')

    ax.set_xlabel('')
    ax.set_ylabel('clone 2')

    ax = plt.subplot(4, 1, 4)

    plot_cnv_genome(ax, data, maxcopies=2, major_col='major_diff', minor_col='minor_diff')

    ax.set_xlabel('chromosome')
    ax.set_ylabel('clone diff')

    plt.tight_layout()

    return fig



