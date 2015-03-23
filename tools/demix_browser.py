import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import seaborn
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('preds_filename', help='deMix Predictions Filename')
argparser.add_argument('--positions', help='annotate positions')
argparser.add_argument('--breakpoints', help='annotate breakpoints')
argparser.add_argument('--max_copies', help='maximum copies to display', type=float, default=5.0)
args = argparser.parse_args()

chromosomes = [str(a) for a in range(1, 23)] + ['X']
chromosome_indices = dict([(chromosome, idx) for idx, chromosome in enumerate(chromosomes)])

if args.preds_filename.endswith('.gz'):
    compression = 'gzip'
else:
    compression = None

cnv = pd.read_csv(args.preds_filename, sep='\t', converters={'chromosome':str}, compression=compression)
cnv.dropna(inplace=True)

cnv = cnv.loc[(cnv['chromosome'].isin(chromosomes))]

cnv['chr_index'] = cnv['chromosome'].apply(lambda a: chromosome_indices[a])

cnv = cnv.sort(['chr_index', 'start'])

chromosome_length = cnv.groupby('chromosome', sort=False)['end'].max()
chromosome_end = np.cumsum(chromosome_length)
chromosome_start = chromosome_end.shift(1)
chromosome_start[0] = 0

cnv.set_index('chromosome', inplace=True)
cnv['chromosome_start'] = chromosome_start
cnv['chromosome_end'] = chromosome_end
cnv.reset_index(inplace=True)

cnv['chromosome_mid'] = 0.5 * (cnv['chromosome_start'] + cnv['chromosome_end'])

cnv['plot_start'] = cnv['start'] + cnv['chromosome_start']
cnv['plot_end'] = cnv['end'] + cnv['chromosome_start']

mingap = 1000

fig = plt.figure(figsize=(12,9))

gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=(4, 1))

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

color_set = plt.get_cmap('Set1')
color_set = [color_set(float(i)/len(chromosomes)) for i in range(len(chromosomes))]
chromosome_color = lambda c: color_set[chromosomes.index(c)]
scatter_colors = [chromosome_color(c) for c in cnv['chromosome'].values]
            
major_minor_scatter = ax1.scatter(cnv['major_raw'], cnv['minor_raw'],
                                  s=cnv['length']/20000.0, 
                                  facecolor=scatter_colors, edgecolor=scatter_colors, linewidth=0.0,
                                  picker=True)

major_minor_scatter_highlight = ax1.scatter(cnv['major_raw'], cnv['minor_raw'],
                                            s=cnv['length']/20000.0, 
                                            facecolor=(0,0,0,0), edgecolor=(0,0,0,0), linewidth=0.0,
                                            picker=True)

ax1.set_xlim((-0.5, args.max_copies))
ax1.set_ylim((-0.5, 0.8*args.max_copies))

lgnd = ax1.legend([plt.Circle((0, 0), radius=1, color=chromosome_color(c), picker=True) for c in chromosomes], chromosomes, loc=2)
lgnd_patches = list(lgnd.get_patches())

for patch in lgnd_patches:
    patch.set_picker(True)

major_segments = list()
minor_segments = list()
major_connectors = list()
minor_connectors = list()

for (idx, row), (next_idx, next_row) in itertools.izip_longest(cnv.iterrows(), cnv.iloc[1:].iterrows(), fillvalue=(None, None)):
    major_segments.append([(row['plot_start'], row['major_raw']), (row['plot_end'], row['major_raw'])])
    minor_segments.append([(row['plot_start'], row['minor_raw']), (row['plot_end'], row['minor_raw'])])
    if next_row is not None and next_row['plot_start'] - row['plot_end'] < mingap and next_row['chromosome'] == row['chromosome']:
        major_connectors.append([(row['plot_end'], row['major_raw']), (next_row['plot_start'], next_row['major_raw'])])
        minor_connectors.append([(row['plot_end'], row['minor_raw']), (next_row['plot_start'], next_row['minor_raw'])])

major_segments = matplotlib.collections.LineCollection(major_segments, colors='r')
minor_segments = matplotlib.collections.LineCollection(minor_segments, colors='b')
major_connectors = matplotlib.collections.LineCollection(major_connectors, colors='r')
minor_connectors = matplotlib.collections.LineCollection(minor_connectors, colors='b')

major_segments.set_picker(True)
minor_segments.set_picker(True)

ax2.add_collection(major_segments)
ax2.add_collection(minor_segments)
ax2.add_collection(major_connectors)
ax2.add_collection(minor_connectors)

if args.positions is not None:

    if args.positions.endswith('.gz'):
        compression = 'gzip'
    else:
        compression = None

    positions = pd.read_csv(args.positions, sep='\t', converters={'chrom':str}, compression=compression)

    for idx, row in positions.iterrows():
        pos = chromosome_start[row['chrom']] + row['coord']
        markerline, stemlines, baseline = ax2.stem([pos, pos], [-10, 0.5], linefmt='-', markerfmt='-o', color='k')
        plt.setp(markerline, 'markerfacecolor', 'orange', 'markeredgecolor', 'k', 'zorder', 2)

breakpoint_markers = list()
breakpoint_infos = list()

if args.breakpoints is not None:

    if args.breakpoints.endswith('.gz'):
        compression = 'gzip'
    else:
        compression = None

    breakpoints = pd.read_csv(args.breakpoints, sep='\t', converters={'chromosome_1':str, 'chromosome_2':str}, compression=compression)

    # breakpoints = breakpoints.loc[(breakpoints[args.library_id+'_count'] > 0)]
    # breakpoints = breakpoints.loc[(breakpoints['normal_blood_count'] == 0)]

    for idx, row in breakpoints.iterrows():
        info = repr((row['chromosome_1'], row['strand_1'], row['position_1'],
                     row['chromosome_2'], row['strand_2'], row['position_2']))
        for side in ('1', '2'):
            if row['chromosome_'+side] not in chromosomes:
                continue
            pos = chromosome_start[row['chromosome_'+side]] + row['position_'+side]
            marker = None
            if row['strand_'+side] == '+':
                marker = '>'
            elif row['strand_'+side] == '-':
                marker = '<'
            paths = ax2.scatter([pos], [0.], marker=marker, color='orange', s=50, picker=True, zorder=3)
            breakpoint_markers.append(paths)
            breakpoint_infos.append(info)

ax2.set_xlim((cnv['plot_start'].min(), cnv['plot_end'].max()))
ax2.set_ylim((-0.2, args.max_copies + 0.2))

ax2.set_xticks([0] + sorted(cnv['chromosome_end'].unique()))
ax2.set_xticklabels([])

ax2.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(sorted(cnv['chromosome_mid'].unique())))
ax2.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(chromosomes))

ax2.grid(False, which="minor")

class Picker(object):
    def __init__(self):
        self.selected_chromosome = None
    def __call__(self, event):
        if isinstance(event.artist, matplotlib.collections.PathCollection) and event.artist.zorder == 3:
            try:
                ind = breakpoint_markers.index(event.artist)
            except ValueError:
                return

            print breakpoint_infos[ind]

        elif isinstance(event.artist, matplotlib.patches.Rectangle):
            try:
                ind = lgnd_patches.index(event.artist)
            except ValueError:
                return
            chromosome = chromosomes[ind]
            
            # Unhighlight currently selected chromosome if necessary
            if self.selected_chromosome is not None:
                lgnd_patches[chromosomes.index(self.selected_chromosome)].set_edgecolor((0, 0, 0, 0))

            # Clicking on the chromosome again unselects it
            if chromosome == self.selected_chromosome:
                self.unselect_chromosome()
            else:
                self.select_chromosome(chromosome)

        elif isinstance(event.artist, matplotlib.collections.PathCollection) or isinstance(event.artist, matplotlib.collections.LineCollection):

            # Print the segment to the terminal
            cnv_region = cnv.iloc[event.ind[0]]
            print 'selected: {0}:{1}-{2} {3} {4}'.format(cnv_region['chromosome'], int(cnv_region['start']), int(cnv_region['end']), cnv_region['major_raw'], cnv_region['minor_raw'])

            self.select_segment(event.ind)

        event.canvas.draw()

    def select_segment(self, ind):

        # Convert indices to mask
        mask = np.array([False] * len(cnv.index))
        mask[ind] = True
        ind = mask

        # Mask index if we have a selected chromosome
        if self.selected_chromosome is not None:
            ind[(cnv['chromosome'] != self.selected_chromosome).values] = False

        # Highlight scatter points
        scatter_linewidths = np.array([0] * len(cnv.index))
        scatter_linewidths[ind] = 4
        scatter_edgecolors = np.array(['k'] * len(cnv.index))
        scatter_edgecolors[ind] = 'yellow'
        scatter_facecolors = np.array(scatter_colors)
        scatter_facecolors[~ind] = (0, 0, 0, 0)
        major_minor_scatter_highlight.set_linewidths(scatter_linewidths)
        major_minor_scatter_highlight.set_edgecolors(scatter_edgecolors)
        major_minor_scatter_highlight.set_facecolors(scatter_facecolors)

        # Highlight segment lines
        lines_linewidths = np.array([1] * len(cnv.index))
        lines_linewidths[ind] = 4
        major_segments.set_linewidths(lines_linewidths)
        minor_segments.set_linewidths(lines_linewidths)

    def select_chromosome(self, chromosome):

        self.select_segment([])

        ind = chromosomes.index(chromosome)

        # Highlight currently selected chromosome 
        lgnd_patches[ind].set_edgecolor('yellow')
        lgnd_patches[ind].set_linewidth(2)

        # Make other chromosomes invisible
        selected_colors = np.array(scatter_colors)
        selected_colors[(cnv['chromosome'] != chromosome).values] = (0, 0, 0, 0)
        major_minor_scatter.set_edgecolors(selected_colors)
        major_minor_scatter.set_facecolors(selected_colors)

        # Restrict x axis to current chromosome view
        ax2.set_xlim((chromosome_start[chromosome], chromosome_end[chromosome]))
        ticks = np.arange(0, chromosome_length[chromosome], 10000000)
        ax2.set_xticks(ticks + chromosome_start[chromosome])
        ax2.set_xticklabels([str(a/1000000) for a in ticks])

        self.selected_chromosome = chromosome

    def unselect_chromosome(self):

        self.select_segment([])

        # Make all chromosomes visible
        major_minor_scatter.set_edgecolors(scatter_colors)
        major_minor_scatter.set_facecolors(scatter_colors)

        # Redo xaxis for full view
        ax2.set_xlim((cnv['plot_start'].min(), cnv['plot_end'].max()))
        ax2.set_xticks([0] + sorted(cnv['chromosome_end'].unique()))
        ax2.set_xticklabels([])
        ax2.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(sorted(cnv['chromosome_mid'].unique())))
        ax2.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(chromosomes))
        ax2.grid(False, which="minor")

        self.selected_chromosome = None


fig.canvas.mpl_connect('pick_event', Picker())


plt.show()

