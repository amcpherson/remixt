import sys
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

import remixt.analysis.experiment


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
        ax.add_patch(
            Rectangle((row['start'], 0), row['end'] - row['start'], row[field],
                facecolor=colorConverter.to_rgba(color, alpha=0.5),
                edgecolor=colorConverter.to_rgba(color, alpha=0.0),
            )
        )

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


def plot_cnv_genome(ax, cnv, maxcopies=4, minlength=1000, major_col='major', minor_col='minor', 
                    chromosome=None, start=None, end=None):
    """
    Plot major/minor copy number across the genome

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): 'cnv_site' table

    KwArgs:
        maxcopies (int): maximum number of copies for setting y limits
        minlength (int): minimum length of segments to be drawn
        major_col (str): name of major copies column
        minor_col (str): name of minor copies column
        chromosome (str): name of chromosome to plot, None for all chromosomes
        start (int): start of region in chromosome, None for beginning
        end (int): end of region in chromosome, None for end of chromosome

    """
    
    if chromosome is None and (start is not None or end is not None):
        raise ValueError('start and end require chromosome arg')

    cnv = cnv[['chromosome', 'start', 'end', 'length', major_col, minor_col]].copy()
    
    # Create chromosome info table
    chromosome_length = cnv.groupby('chromosome')['end'].max()
    chromosome_info = pd.DataFrame({'length':chromosome_length})
    
    if issubclass(chromosome_info.index.dtype.type, np.integer):
        chromosome_info.sort_index(inplace=True)
    else:
        # Reorder in a standard way with X and Y last
        chromosome_info = chromosome_info.reindex([str(a) for a in xrange(50)] + ['X', 'Y']).dropna()
        if len(chromosome_info.index) != len(cnv['chromosome'].unique()):
            raise Exception('Unable to reindex chromosomes')

    # Calculate start and end in plot
    chromosome_info['end'] = np.cumsum(chromosome_info['length'])
    chromosome_info['start'] = chromosome_info['end'] - chromosome_info['length']
    chromosome_info['mid'] = (chromosome_info['start'] + chromosome_info['end']) / 2.
        
    if minlength is not None:
        cnv = cnv[cnv['length'] >= minlength]

    cnv.set_index('chromosome', inplace=True)
    cnv['chromosome_start'] = chromosome_info['start']
    cnv.reset_index(inplace=True)

    cnv['start'] = cnv['start'] + cnv['chromosome_start']
    cnv['end'] = cnv['end'] + cnv['chromosome_start']

    plot_cnv_segments(ax, cnv, major_col=major_col, minor_col=minor_col)

    ax.set_ylim((0, maxcopies+.6))
    ax.set_yticks(range(0, maxcopies+1))
    ax.set_yticklabels(ax.get_yticks(), ha='left')
    ax.yaxis.tick_left()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    ax.yaxis.set_tick_params(direction='out', labelsize=12)
    ax.xaxis.set_tick_params(direction='out', labelsize=12)

    ax.xaxis.tick_bottom()

    if chromosome is not None:
        if start is None:
            start = 0
        if end is None:
            end = chromosome_info.loc[chromosome, 'length']
        plot_start = start + chromosome_info.loc[chromosome, 'start']
        plot_end = end + chromosome_info.loc[chromosome, 'start']
        ax.set_xlim((plot_start, plot_end))
        ax.set_xlabel('chromosome ' + chromosome, fontsize=20)
        step = (end - start) / 12.
        step = np.round(step, decimals=-int(np.floor(np.log10(step))))
        xticks = np.arange(plot_start, plot_end, step)
        xticklabels = np.arange(start, end, step)
        ax.set_xticks(xticks)
        ax.set_xticklabels(['{:g}'.format(a/1e6) + 'Mb' for a in xticklabels])
    else:
        ax.set_xlim((0, chromosome_info['end'].max()))
        ax.set_xlabel('chromosome')
        ax.set_xticks([0] + list(chromosome_info['end'].values))
        ax.set_xticklabels([])
        ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(chromosome_info['mid']))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(chromosome_info.index.values))

    ax.yaxis.set_tick_params(pad=8)
    ax.yaxis.set_tick_params(pad=8)
    
    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')
    
    return chromosome_info


def plot_breakpoints_genome(ax, breakpoint, chromosome_info, scale_height=1.0):
    """ Plot breakpoint arcs

    Args:
        ax (matplotlib.axes.Axes): plot axes
        breakpoint (pandas.DataFrame): breakpoint
        chromosome_info (pandas.DataFrame): per chromosome start and end in plot returned from plot_cnv_genome

    """

    plot_height = ax.get_ylim()[1] * 0.8
    plot_length = ax.get_xlim()[1] - ax.get_xlim()[0]

    for side in ('1', '2'):
        
        breakpoint.set_index('chromosome_'+side, inplace=True)
        breakpoint['chromosome_start_'+side] = chromosome_info['start']
        breakpoint.reset_index(inplace=True)
        
        breakpoint['plot_position_'+side] = breakpoint['position_'+side] + breakpoint['chromosome_start_'+side]

    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    for idx, row in breakpoint.iterrows():

        pos_1, pos_2 = sorted(row[['plot_position_1', 'plot_position_2']])
        height = scale_height * 2. * plot_height * (pos_2 - pos_1) / float(plot_length)
        
        visible_1 = pos_1 >= ax.get_xlim()[0] and pos_1 <= ax.get_xlim()[1]
        visible_2 = pos_2 >= ax.get_xlim()[0] and pos_2 <= ax.get_xlim()[1]
        
        if not visible_1 and not visible_2:
            continue

        if not visible_1 or not visible_2:
            height = plot_height * 10.

        verts = [(pos_1, 0.), (pos_1, height), (pos_2, height), (pos_2, 0.)]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', edgecolor='#2eb036', lw=2)
        ax.add_patch(patch)


def experiment_plot(experiment, likelihood, cn, h):
    """ Plot a sequencing experiment

    Args:
        experiment (Experiment): experiment object containing simulation information
        likelihood (ReadCountLikelihood): likelihood model
        cn (numpy.array): segment copy number
        h (numpy.array): haploid depths

    Returns:
        matplotlib.Figure: figure object of plots

    """

    data = remixt.analysis.experiment.create_cn_table(experiment, likelihood, cn, h)

    num_plots = 3
    width = 20
    height = 6
    if 'major_2' in data:
        num_plots = 5
        height = 10

    plot_idx = 1

    fig = plt.figure(figsize=(width, height))

    ax = plt.subplot(num_plots, 1, plot_idx)
    plot_idx += 1

    plot_cnv_genome(ax, data, maxcopies=4, major_col='major_raw', minor_col='minor_raw')

    ax.set_xlabel('')
    ax.set_ylabel('raw')

    ax = plt.subplot(num_plots, 1, plot_idx)
    plot_idx += 1

    plot_cnv_genome(ax, data, maxcopies=4, major_col='major_raw_e', minor_col='minor_raw_e')

    ax.set_xlabel('')
    ax.set_ylabel('expected')

    ax = plt.subplot(num_plots, 1, plot_idx)
    plot_idx += 1

    plot_cnv_genome(ax, data, maxcopies=4, major_col='major_1', minor_col='minor_1')

    ax.set_xlabel('')
    ax.set_ylabel('clone 1')

    if 'major_2' in data:

        ax = plt.subplot(num_plots, 1, plot_idx)
        plot_idx += 1

        plot_cnv_genome(ax, data, maxcopies=4, major_col='major_2', minor_col='minor_2')

        ax.set_xlabel('')
        ax.set_ylabel('clone 2')

        ax = plt.subplot(num_plots, 1, plot_idx)
        plot_idx += 1

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



