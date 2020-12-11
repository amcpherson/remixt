import sys
import os
import itertools
import pickle
import pandas as pd
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.backends.backend_pdf

import remixt.analysis.experiment
import remixt.analysis.readdepth


def filled_density_weighted(ax, data, weights, c, a, xmin, xmax, cov, rotate=False):
    """ Weighted filled density plot.
    """
    density = scipy.stats.gaussian_kde(data, bw_method=cov, weights=weights)
    xs = [xmin] + list(np.linspace(xmin, xmax, 2000)) + [xmax]
    ys = density(np.array(xs))
    ys[0] = 0.0
    ys[-1] = 0.0
    if rotate:
        ax.plot(ys, xs, color=c, alpha=a)
        ax.fill_betweenx(xs, ys, color=c, alpha=a)
    else:
        ax.plot(xs, ys, color=c, alpha=a)
        ax.fill(xs, ys, color=c, alpha=a)


def plot_cnv_segments(ax, cnv, major_col='major', minor_col='minor', do_fill=False,):
    """ Plot raw major/minor copy number as line plots

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): cnv table
        major_col (str): name of major copies column
        minor_col (str): name of minor copies column
        do_fill (boolean): fill from 0

    Plot major and minor copy number as line plots.  The columns 'start' and 'end'
    are expected and should be adjusted for full genome plots.  Values from the
    'major_col' and 'minor_col' columns are plotted.

    """ 

    segment_color_major = plt.get_cmap('RdBu')(0.1)
    segment_color_minor = plt.get_cmap('RdBu')(0.9)

    quad_color_major = colorConverter.to_rgba(segment_color_major, alpha=0.5)
    quad_color_minor = colorConverter.to_rgba(segment_color_minor, alpha=0.5)

    cnv = cnv.sort_values('start')

    def create_segments(df, field):
        segments = np.array([[df['start'].values, df[field].values], [df['end'].values, df[field].values]])
        segments = np.transpose(segments, (2, 0, 1))
        return segments

    def create_connectors(df, field):
        prev = df.iloc[:-1].reset_index()
        next = df.iloc[1:].reset_index()
        mids = ((prev[field] + next[field]) / 2.0).values
        prev_cnct = np.array([[prev['end'].values, prev[field].values], [prev['end'].values, mids]])
        prev_cnct = np.transpose(prev_cnct, (2, 0, 1))
        next_cnct = np.array([[next['start'].values, mids], [next['start'].values, next[field].values]])
        next_cnct = np.transpose(next_cnct, (2, 0, 1))
        return np.concatenate([prev_cnct, next_cnct])

    def create_quads(df, field):
        quads = np.array([
            [df['start'].values, np.zeros(len(df.index))],
            [df['start'].values, df[field].values],
            [df['end'].values, df[field].values],
            [df['end'].values, np.zeros(len(df.index))],
        ])
        quads = np.transpose(quads, (2, 0, 1))
        return quads

    major_segments = create_segments(cnv, major_col)
    minor_segments = create_segments(cnv, minor_col)
    ax.add_collection(matplotlib.collections.LineCollection(major_segments, colors=segment_color_major, lw=1))
    ax.add_collection(matplotlib.collections.LineCollection(minor_segments, colors=segment_color_minor, lw=1))

    major_connectors = create_connectors(cnv, major_col)
    minor_connectors = create_connectors(cnv, minor_col)
    ax.add_collection(matplotlib.collections.LineCollection(major_connectors, colors=segment_color_major, lw=1))
    ax.add_collection(matplotlib.collections.LineCollection(minor_connectors, colors=segment_color_minor, lw=1))

    if do_fill:
        major_quads = create_quads(cnv, major_col)
        minor_quads = create_quads(cnv, minor_col)
        ax.add_collection(matplotlib.collections.PolyCollection(major_quads, facecolors=quad_color_major, edgecolors=quad_color_major, lw=0))
        ax.add_collection(matplotlib.collections.PolyCollection(minor_quads, facecolors=quad_color_minor, edgecolors=quad_color_minor, lw=0))


def plot_cnv_genome(ax, cnv, mincopies=-0.4, maxcopies=4, minlength=1000, major_col='major', minor_col='minor', 
                    chromosome=None, start=None, end=None, tick_step=None, do_fill=False):
    """ Plot major/minor copy number across the genome

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): copy number table

    KwArgs:
        mincopies (float): minimum number of copies for setting y limits
        maxcopies (float): maximum number of copies for setting y limits
        minlength (int): minimum length of segments to be drawn
        major_col (str): name of major copies column
        minor_col (str): name of minor copies column
        chromosome (str): name of chromosome to plot, None for all chromosomes
        start (int): start of region in chromosome, None for beginning
        end (int): end of region in chromosome, None for end of chromosome
        tick_step (float): genomic length between x steps
        do_fill (boolean): fill to 0 for copy number

    """

    if chromosome is None and (start is not None or end is not None):
        raise ValueError('start and end require chromosome arg')

    # Ensure we dont modify the calling function's table
    cnv = cnv[['chromosome', 'start', 'end', major_col, minor_col]].copy()

    if 'length' not in cnv:
        cnv['length'] = cnv['end'] - cnv['start']

    # Restrict segments to those plotted
    if chromosome is not None:
        cnv = cnv[cnv['chromosome'] == chromosome]
    if start is not None:
        cnv = cnv[cnv['end'] > start]
    if end is not None:
        cnv = cnv[cnv['start'] < end]

    # Create chromosome info table
    chromosomes = remixt.utils.sort_chromosome_names(cnv['chromosome'].unique())
    chromosome_length = cnv.groupby('chromosome')['end'].max()
    chromosome_info = pd.DataFrame({'length':chromosome_length}, index=chromosomes)

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

    plot_cnv_segments(ax, cnv, major_col=major_col, minor_col=minor_col, do_fill=do_fill)

    ax.set_yticks(range(int(mincopies), int(maxcopies - mincopies) + 1))
    ax.set_ylim((mincopies, maxcopies))
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
        ax.set_xlabel('Chromosome ' + chromosome, fontsize=14)
        if tick_step is None:
            tick_step = (end - start) / 12.
            tick_step = np.round(tick_step, decimals=-int(np.floor(np.log10(tick_step))))
        xticks = np.arange(plot_start, plot_end, tick_step)
        xticklabels = np.arange(start, end, tick_step)
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


def plot_cnv_genome_density(fig, transform, cnv):
    """ Plot major/minor copy number across the genome and as a density

    Args:
        fig (matplotlib.figure.Figure): figure to which plots are added
        transform (matplotlib.transform.Transform): transform for locating axes
        cnv (pandas.DataFrame): copy number table

    """

    box = matplotlib.transforms.Bbox([[0.05, 0.05], [0.65, 0.95]])
    ax = fig.add_axes(transform.transform_bbox(box))

    remixt.cn_plot.plot_cnv_genome(ax, cnv, mincopies=-1, maxcopies=6, major_col='major_raw', minor_col='minor_raw')
    ax.set_ylabel('Raw copy number')
    ylim = ax.get_ylim()

    box = matplotlib.transforms.Bbox([[0.7, 0.05], [0.95, 0.95]])
    ax = fig.add_axes(transform.transform_bbox(box))

    cov = 0.05
    data = cnv[['minor_raw', 'major_raw', 'length']].replace(np.inf, np.nan).dropna()
    filled_density_weighted(
        ax, data['minor_raw'].values, data['length'].values,
        'blue', 0.5, ylim[0], ylim[1], cov, rotate=True)
    filled_density_weighted(
        ax, data['major_raw'].values, data['length'].values,
        'red', 0.5, ylim[0], ylim[1], cov, rotate=True)
    ax.set_ylim(ylim)
    ax.set_xlabel('Density')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.yaxis.grid(True)

    return fig


def create_chromosome_color_map(chromosomes):
    """ Create a map of colors per chromosome.

    Args:
        chromosomes (list): list of chromosome names

    Returns:
        pandas.DataFrame: chromosome color table
        
    """

    color_map = plt.get_cmap('Dark2')

    chromosome_colors = list()
    for i in range(len(chromosomes)):
        if len(chromosomes) == 1:
            f = 0.
        else:
            f = float(i)/float(len(chromosomes)-1)
        rgb_color = color_map(f)
        hex_color = matplotlib.colors.rgb2hex(rgb_color)
        chromosome_colors.append(hex_color)

    chromosome_colors = pd.DataFrame({'chromosome':chromosomes, 'color':chromosome_colors})

    return chromosome_colors


def plot_cnv_scatter(ax, cnv, major_col='major', minor_col='minor', highlight_col=None, chromosome_colors=None):
    """ Scatter plot segments major by minor.

    Args:
        ax (matplotlib.axes.Axes): plot axes
        cnv (pandas.DataFrame): copy number table

    KwArgs:
        major_col (str): name of major copies column
        minor_col (str): name of minor copies column
        highlight_col (str): name of boolean column for highlighting specific segments
        chromosome_colors (pandas.DataFrame): chromosome color table

    """

    cnv = cnv[['chromosome', 'start', 'end', 'length', major_col, minor_col]].replace(np.inf, np.nan).dropna()

    # Create color map for chromosomes
    chromosomes = remixt.utils.sort_chromosome_names(cnv['chromosome'].unique())

    # Create chromosome color map if not given
    if chromosome_colors is None:
        chromosome_colors = create_chromosome_color_map(chromosomes)

    # Scatter size scaled by segment length
    cnv['scatter_size'] = 10. * np.sqrt(cnv['length'] / 1e6)

    # Scatter color
    cnv = cnv.merge(chromosome_colors)

    major_samples = remixt.utils.weighted_resample(cnv[major_col].values, cnv['length'].values)
    major_min = np.percentile(major_samples, 1)
    major_max = np.percentile(major_samples, 99)
    major_margin = 0.25 * (major_max - major_min)
    xlim = (major_min - major_margin, major_max + major_margin)

    minor_samples = remixt.utils.weighted_resample(cnv[minor_col].values, cnv['length'].values)
    minor_min = np.percentile(minor_samples, 1)
    minor_max = np.percentile(minor_samples, 99)
    minor_margin = 0.25 * (minor_max - minor_min)
    ylim = (minor_min - minor_margin, minor_max + minor_margin)

    if highlight_col is not None:
        cnv_greyed = cnv[~cnv[highlight_col]]
        cnv = cnv[cnv[highlight_col]]
        points = ax.scatter(cnv_greyed[major_col], cnv_greyed[minor_col],
            s=cnv_greyed['scatter_size'], facecolor='#d0d0e0', edgecolor='#d0d0e0',
            linewidth=0.0, zorder=2)

    points = ax.scatter(cnv[major_col], cnv[minor_col],
        s=cnv['scatter_size'], facecolor=cnv['color'], edgecolor=cnv['color'],
        linewidth=0.0, zorder=2)
    
    ax.set_xlim(xlim)
    ax.set_xlabel('major')
    ax.set_ylim(ylim)
    ax.set_ylabel('minor')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    ax.grid(True)

    lgnd_artists = [plt.Circle((0, 0), color=c) for c in chromosome_colors['color']]
    lgnd = ax.legend(lgnd_artists, chromosomes,
        loc=2, markerscale=0.5, fontsize=6, ncol=2,
        title='Chromosome', frameon=True)
    lgnd.get_frame().set_edgecolor('w')


def plot_cnv_scatter_density(fig, transform, data, major_col='major', minor_col='minor', annotate=(), info=''):
    """ Plot CNV Scatter with major minor densities on axes.

    Args:
        fig (matplotlib.figure.Figure): figure to which plots are added
        transform (matplotlib.transform.Transform): transform for locating axes
        data (pandas.DataFrame): copy number data

    KwArgs:
        major_col (str): name of major copies column
        minor_col (str): name of minor copies column
        annotate (iterable): copy values to annotate with dashed line
        info (str): information to add to empty axis

    Returns:
    """

    box = matplotlib.transforms.Bbox([[0.05, 0.05], [0.65, 0.65]])
    ax = fig.add_axes(transform.transform_bbox(box))

    remixt.cn_plot.plot_cnv_scatter(ax, data)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for a in annotate:
        ax.plot(xlim, [a, a], '--k')
        ax.plot([a, a], ylim, '--k')

    box = matplotlib.transforms.Bbox([[0.05, 0.7], [0.65, 0.95]])
    ax = fig.add_axes(transform.transform_bbox(box))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.xaxis.grid(True)
    filled_density_weighted(
        ax, data[major_col].values, data['length'].values,
        '0.75', 0.5, xlim[0], xlim[1], 1e-7)
    ax.set_xlim(xlim)

    for a in annotate:
        ax.plot([a, a], ax.get_ylim(), '--k')

    box = matplotlib.transforms.Bbox([[0.7, 0.05], [0.95, 0.65]])
    ax = fig.add_axes(transform.transform_bbox(box))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.yaxis.grid(True)
    filled_density_weighted(
        ax, data[minor_col].values, data['length'].values,
        '0.75', 0.5, ylim[0], ylim[1], 1e-7, rotate=True)
    ax.set_ylim(ylim)

    for a in annotate:
        ax.plot(ax.get_xlim(), [a, a], '--k')

    box = matplotlib.transforms.Bbox([[0.7, 0.7], [0.95, 0.95]])
    ax = fig.add_axes(transform.transform_bbox(box))

    ax.axis('off')
    ax.text(0.0, 0.0, info)

    return fig


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
        patch = patches.PathPatch(path, facecolor='none', edgecolor='#2eb036', lw=2, zorder=100)
        ax.add_patch(patch)


def experiment_plot(experiment, cn, h, chromosome=None, start=None, end=None, maxcopies=4):
    """ Plot a sequencing experiment

    Args:
        experiment (Experiment): experiment object containing simulation information
        cn (numpy.array): segment copy number
        h (numpy.array): haploid depths

    KwArgs:
        chromosome (str): name of chromosome to plot, None for all chromosomes
        start (int): start of region in chromosome, None for beginning
        end (int): end of region in chromosome, None for end of chromosome
        maxcopies (int): max copy number for y axis

    Returns:
        matplotlib.Figure: figure object of plots

    """

    data = remixt.analysis.experiment.create_cn_table(experiment, cn, h)

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

    plot_cnv_genome(ax, data, maxcopies=maxcopies, major_col='major_raw', minor_col='minor_raw',
                    chromosome=chromosome, start=start, end=end)

    ax.set_xlabel('')
    ax.set_ylabel('raw')

    ax = plt.subplot(num_plots, 1, plot_idx)
    plot_idx += 1

    plot_cnv_genome(ax, data, maxcopies=maxcopies, major_col='major_raw_e', minor_col='minor_raw_e',
                    chromosome=chromosome, start=start, end=end)

    ax.set_xlabel('')
    ax.set_ylabel('expected')

    ax = plt.subplot(num_plots, 1, plot_idx)
    plot_idx += 1

    plot_cnv_genome(ax, data, maxcopies=maxcopies, major_col='major_1', minor_col='minor_1',
                    chromosome=chromosome, start=start, end=end)

    ax.set_xlabel('')
    ax.set_ylabel('clone 1')

    if 'major_2' in data:

        ax = plt.subplot(num_plots, 1, plot_idx)
        plot_idx += 1

        plot_cnv_genome(ax, data, maxcopies=maxcopies, major_col='major_2', minor_col='minor_2',
                        chromosome=chromosome, start=start, end=end)

        ax.set_xlabel('')
        ax.set_ylabel('clone 2')

        ax = plt.subplot(num_plots, 1, plot_idx)
        plot_idx += 1

        plot_cnv_genome(ax, data, maxcopies=2, major_col='major_diff', minor_col='minor_diff',
                        chromosome=chromosome, start=start, end=end)

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


def gc_plot(gc_table_filename, plot_filename):
    """ Plot the probability distribution of GC content for sampled reads

    Args:
        gc_table_filename (str): table of binned gc values
        plot_filename (str): plot PDF filename

    """
    gc_binned = pd.read_csv(gc_table_filename, sep='\t')

    fig = plt.figure(figsize=(4,4))

    plt.scatter(gc_binned['gc_bin'].values, gc_binned['mean'].values, c='k', s=4)
    plt.plot(gc_binned['gc_bin'].values, gc_binned['smoothed'].values, c='r')

    plt.xlabel('gc %')
    plt.ylabel('density')
    plt.xlim((-0.5, 100.5))
    plt.ylim((-0.01, gc_binned['mean'].max() * 1.1))

    plt.tight_layout()

    fig.savefig(plot_filename, format='pdf', bbox_inches='tight')


def plot_depth(ax, read_depth, minor_modes=None):
    """ Plot read depth of major minor and total as a density

    Args:
        ax (matplotlib.axis): optional axis for plotting major/minor/total read depth
        read_depth (pandas.DataFrame): observed major, minor, and total read depth and lengths

    KwArgs:
        minor_modes (list): annotate minor modes with verticle lines

    """

    total_depth_samples = remixt.utils.weighted_resample(read_depth['total'].values, read_depth['length'].values)

    depth_max = np.percentile(total_depth_samples, 95)
    cov = 0.0000001

    filled_density_weighted(ax, read_depth['minor'].values, read_depth['length'].values, 'blue', 0.5, 0.0, depth_max, cov)
    filled_density_weighted(ax, read_depth['major'].values, read_depth['length'].values, 'red', 0.5, 0.0, depth_max, cov)
    filled_density_weighted(ax, read_depth['total'].values, read_depth['length'].values, 'grey', 0.5, 0.0, depth_max, cov)

    if minor_modes is not None:
        init_h_mono = np.array(remixt.analysis.readdepth.calculate_candidate_h_monoclonal(minor_modes))

        h_normal = init_h_mono[0, 0]
        h_tumour = init_h_mono[:, 1] + h_normal

        plt.axvline(h_normal, lw=1, color='g')
        for x in h_tumour:
            plt.axvline(x, lw=1, color='g', ls=':')

    ax.set_xlabel('Read Depth')
    ax.set_ylabel('Density')


def plot_experiment(experiment_plot_filename, experiment_filename):
    """ Plot an experiment

    Args:
        experiment_plot_filename (str): plot PDF filename
        experiment_filename (str): filename of experiment pickle

    """
    with open(experiment_filename, 'r') as experiment_file:
        exp = pickle.load(experiment_file)

    fig = experiment_plot(exp, exp.cn, exp.h)

    fig.savefig(experiment_plot_filename, format='pdf', bbox_inches='tight', dpi=300)


def plot_mixture(mixture_plot_filename, mixture_filename):
    """ Plot a mixture

    Args:
        mixture_plot_filename (str): plot PDF filename
        mixture_filename (str): filename of mixture pickle

    """
    with open(mixture_filename, 'r') as mixture_file:
        mixture = pickle.load(mixture_file)

    fig = mixture_plot(mixture)

    fig.savefig(mixture_plot_filename, format='pdf', bbox_inches='tight', dpi=300)


def ploidy_analysis_plots(experiment_filename, plots_filename):
    """ Generate ploidy analysis plots

    Args:
        experiment_filename (str): experiment pickle filename
        plots_filename (str): ploidy analysis plots filename

    """

    with open(experiment_filename, 'rb') as experiment_file:
        experiment = pickle.load(experiment_file)

    read_depth = remixt.analysis.readdepth.calculate_depth(experiment)
    minor_modes = remixt.analysis.readdepth.calculate_minor_modes(read_depth)
    init_h_mono = remixt.analysis.readdepth.calculate_candidate_h_monoclonal(minor_modes)

    pdf = matplotlib.backends.backend_pdf.PdfPages(plots_filename)

    for h in init_h_mono:
        cn_modes = h[0] + np.arange(0, 5) * h[1]

        read_depth['major_raw'] = (read_depth['major'] - h[0]) / h[1]
        read_depth['minor_raw'] = (read_depth['minor'] - h[0]) / h[1]

        f = h / h.sum()

        major, minor, length = read_depth.replace(np.inf, np.nan).dropna()[['major_raw', 'minor_raw', 'length']].values.T
        ploidy = ((major + minor) * length).sum() / length.sum()

        info = 'Statistics:\n\n'
        info += ' normal = {:.3f}\n\n'.format(f[0])
        info += ' tumour = {:.3f}\n\n'.format(f[1])
        info += ' ploidy = {:.3f}'.format(ploidy)

        fig = plt.figure(figsize=(12,12))

        box = matplotlib.transforms.Bbox([[0., 0.25], [1., 1.]])
        transform = matplotlib.transforms.BboxTransformTo(box)
        remixt.cn_plot.plot_cnv_scatter_density(fig, transform, read_depth, annotate=cn_modes, info=info)

        box = matplotlib.transforms.Bbox([[0., 0.0], [1., 0.25]])
        transform = matplotlib.transforms.BboxTransformTo(box)
        remixt.cn_plot.plot_cnv_genome_density(fig, transform, read_depth.query('high_quality'))

        pdf.savefig(bbox_inches='tight')
        plt.close()

    pdf.close()

