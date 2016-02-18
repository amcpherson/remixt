import logging
import warnings

# warnings.filterwarnings('error')
logging.basicConfig(level=logging.DEBUG)

import collections
import glob
import math
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot
import matplotlib.colors

from bokeh.models import *
from bokeh.plotting import Figure, curdoc
from bokeh.core.properties import value


chromosomes = [str(a) for a in range(1, 23)] + ['X']

color_map = matplotlib.pyplot.get_cmap('Set1')
chromosome_colors = list()
for i in xrange(len(chromosomes)):
    rgb_color = color_map(float(i)/float(len(chromosomes)))
    hex_color = matplotlib.colors.rgb2hex(rgb_color)
    chromosome_colors.append(hex_color)
chromosome_colors = pd.DataFrame({'chromosome':chromosomes, 'scatter_color':chromosome_colors})

chromosome_indices = dict([(chromosome, idx) for idx, chromosome in enumerate(chromosomes)])
del chromosome

patient_samples = collections.defaultdict(list)
sample_stores = dict()


for data_filename in glob.glob('./patient_*/sample_*.h5'):
    patient = data_filename.split('/')[-2][len('patient_'):]
    sample = data_filename.split('/')[-1][len('sample_'):-len('.h5')]

    patient_samples[patient].append(sample)
    sample_stores[(patient, sample)] = pd.HDFStore(data_filename, 'r')


def major_minor_scatter_plot(source):
    """
    """
    p = Figure(
        title='raw major vs minor',
        plot_width=1000, plot_height=500,
        tools='pan,wheel_zoom,box_select,reset,lasso_select',
        logo=None,
        title_text_font_size=value('10pt'),
        x_range=[-0.5, 6.5],
        y_range=[-0.5, 4.5],
    )

    p.circle(x='major_raw', y='minor_raw',
        size='scatter_size', color='scatter_color', alpha=0.5,
        source=source,
    )

    return p


def major_minor_segment_plot(source, major_column, minor_column, x_range, name, width=1000):
    """
    """
    hover = HoverTool(
        tooltips=[
            ('chromosome', '@chromosome'),
            ('start', '@start'),
            ('end', '@end'),
            ('major_raw', '@major_raw'),
            ('minor_raw', '@minor_raw'),
        ]
    )

    tools = [
        PanTool(dimensions=['x']),
        WheelZoomTool(dimensions=['x']),
        BoxZoomTool(),
        BoxSelectTool(),
        ResetTool(),
        TapTool(),
        hover,
    ]

    p = Figure(
        title=name+' chromosome major/minor',
        plot_width=width, plot_height=200,
        tools=tools,
        logo=None,
        title_text_font_size=value('10pt'),
        x_range=x_range,
        y_range=[-0.5, 6.5],
    )

    p.quad(top=major_column, bottom=0, left='plot_start', right='plot_end',
        source=source, color='red', alpha=0.05, line_width=0)

    p.quad(top=minor_column, bottom=0, left='plot_start', right='plot_end',
        source=source, color='blue', alpha=0.05, line_width=0)

    p.segment(y0=major_column, y1=major_column, x0='plot_start', x1='plot_end',
        source=source, color='red', alpha=1.0, line_width=4)

    p.segment(y0=minor_column, y1=minor_column, x0='plot_start', x1='plot_end',
        source=source, color='blue', alpha=1.0, line_width=2)

    return p


def breakpoints_plot(source, x_range, width=1000):
    """
    """
    hover = HoverTool(
        tooltips=[
            ('prediction_id', '@prediction_id'),
            ('chromosome', '@chromosome'),
            ('position', '@position'),
            ('strand', '@strand'),
            ('other_chromosome', '@other_chromosome'),
            ('other_position', '@other_position'),
            ('other_strand', '@other_strand'),
            ('type', '@type'),
        ]
    )

    tools = [
        PanTool(dimensions=['x']),
        WheelZoomTool(dimensions=['x']),
        BoxSelectTool(),
        ResetTool(),
        TapTool(),
        hover,
    ]

    p = Figure(
        title='break ends',
        plot_width=width, plot_height=150,
        tools=tools,
        logo=None,
        title_text_font_size=value('10pt'),
        x_range=x_range,
        y_range=['+', '-'],
    )

    p.triangle(x='plot_position', y='strand', size=10, angle='strand_angle',
        line_color='grey', fill_color='clonality_color', alpha=1.0,
        source=source)

    return p


def setup_chromosome_plot_axes(p):
    """
    """
    p.xaxis[0].formatter = NumeralTickFormatter(format='0.00a')


def setup_genome_plot_axes(p, chromosome_plot_info):
    """
    """
    chromosomes = list(chromosome_plot_info['chromosome'].values)
    chromosome_bounds = [0] + list(chromosome_plot_info['chromosome_plot_end'].values)
    chromosome_mids = list(chromosome_plot_info['chromosome_plot_mid'].values)

    p.xgrid.ticker = FixedTicker(ticks=[-1] + chromosome_bounds + [chromosome_bounds[-1] + 1])
    p.xgrid.band_fill_alpha = 0.1
    p.xgrid.band_fill_color = "navy"

    p.xaxis[0].ticker = FixedTicker(ticks=chromosome_bounds)
    p.xaxis[0].major_label_text_font_size = value('0pt')

    p.text(x=chromosome_mids, y=-0.5, text=chromosomes, text_font_size=value('0.5em'), text_align='center')


def retrieve_solution_data(patient, sample):
    """
    """
    store = sample_stores[(patient, sample)]

    solutions_df = store['stats']

    for idx, row in solutions_df.iterrows():

        # Calculate ploidy
        cnv = retrieve_cnv_data(patient, sample, row['idx'])
        cnv = cnv.replace([np.inf, -np.inf], np.nan).dropna()
        ploidy = (cnv['length'] * (cnv['major_raw_e'] + cnv['minor_raw_e'])).sum() / cnv['length'].sum()
        solutions_df.loc[idx, 'ploidy'] = ploidy

        # Add haploid normal/tumour depth and clone fraction
        h = store['/solutions/solution_{0}/h'.format(row['idx'])]
        solutions_df.loc[idx, 'haploid_normal'] = h.values[0]
        solutions_df.loc[idx, 'haploid_tumour'] = h.values[1:].sum()
        solutions_df.loc[idx, 'haploid_tumour_mode'] = h.values.sum()
        solutions_df.loc[idx, 'clone_1_fraction'] = h.values[1] / h.values[1:].sum()
        solutions_df.loc[idx, 'clone_2_fraction'] = 1. - solutions_df.loc[idx, 'clone_1_fraction']

    return solutions_df


def retrieve_solutions(patient, sample):
    """
    """
    store = sample_stores[(patient, sample)]

    return list(store['stats']['idx'].astype(str).values)


def retrieve_cnv_data(patient, sample, solution, chromosome=''):
    """
    """
    store = sample_stores[(patient, sample)]

    cnv = store['solutions/solution_{0}/cn'.format(solution)]

    if chromosome != '':
        cnv = cnv[cnv['chromosome'] == chromosome].copy()

    return cnv


def retrieve_chromosome_plot_info(patient, sample, solution, chromosome=''):
    """
    """
    cnv = retrieve_cnv_data(patient, sample, solution, chromosome)

    cnv['chromosome_index'] = cnv['chromosome'].apply(lambda a: chromosome_indices[a])
    cnv.sort_values(['chromosome_index', 'start'], inplace=True)

    info = (
        cnv.groupby('chromosome', sort=False)['end']
        .max().reset_index().rename(columns={'end':'chromosome_length'}))

    info['chromosome_plot_end'] = np.cumsum(info['chromosome_length'])
    info['chromosome_plot_start'] = info['chromosome_plot_end'].shift(1)
    info.loc[info.index[0], 'chromosome_plot_start'] = 0
    info['chromosome_plot_mid'] = 0.5 * (info['chromosome_plot_start'] + info['chromosome_plot_end'])

    return info


def prepare_cnv_data(cnv, chromosome_plot_info, smooth_segments=False):
    """
    """
    # Group segments with same state
    if smooth_segments:
        cnv['chromosome_index'] = cnv['chromosome'].apply(lambda a: chromosome_indices[a])
        cnv['diff'] = cnv[['chromosome_index', 'major_1', 'major_2', 'minor_1', 'minor_2']].diff().abs().sum(axis=1)
        cnv['is_diff'] = (cnv['diff'] != 0)
        cnv['cn_group'] = cnv['is_diff'].cumsum()

        def agg_segments(df):

            stable_cols = [
                'chromosome',
                'major_1',
                'major_2',
                'minor_1',
                'minor_2',
                'major_raw_e',
                'minor_raw_e',
            ]

            a = df[stable_cols].iloc[0]

            a['start'] = df['start'].min()
            a['end'] = df['end'].max()
            a['length'] = df['length'].sum()

            length_normalized_cols = [
                'major_raw',
                'minor_raw',
            ]

            for col in length_normalized_cols:
                a[col] = (df[col] * df['length']).sum() / (df['length'].sum() + 1e-16)

            return a

        cnv = cnv.groupby('cn_group').apply(agg_segments)

    # Scatter size scaled by segment length
    cnv['scatter_size'] = 2. * np.sqrt(cnv['length'] / 1e6)

    # Scatter color by chromosome
    cnv = cnv.merge(chromosome_colors)

    # Calculate plot start and end
    cnv = cnv.merge(chromosome_plot_info[['chromosome', 'chromosome_plot_start']])
    cnv['plot_start'] = cnv['start'] + cnv['chromosome_plot_start']
    cnv['plot_end'] = cnv['end'] + cnv['chromosome_plot_start']

    # Drop nan values
    cnv = cnv.replace(np.inf, np.nan).fillna(0)

    return cnv


def retrieve_brk_data(patient, sample, solution, chromosome_plot_info):
    """
    """
    store = sample_stores[(patient, sample)]

    brk = store['breakpoints']

    def calculate_breakpoint_type(row):
        if row['chromosome_1'] != row['chromosome_2']:
            return 'translocation'
        if row['strand_1'] == row['strand_2']:
            return 'inversion'
        positions = sorted([(row['position_{0}'.format(side)], row['strand_{0}'.format(side)]) for side in (1, 2)])
        if positions[0][1] == '+':
            return 'deletion'
        else:
            return 'duplication'
    brk['type'] = brk.apply(calculate_breakpoint_type, axis=1)

    # Duplicate required columns before stack
    brk['type_1'] = brk['type']
    brk['type_2'] = brk['type']

    # Stack break ends
    brk.set_index(['prediction_id'], inplace=True)
    brk = brk.filter(regex='(_1|_2)')
    def split_col_name(col):
        parts = col.split('_')
        return '_'.join(parts[:-1]), parts[-1]
    brk.columns = pd.MultiIndex.from_tuples([split_col_name(col) for col in brk.columns])
    brk.columns.names = 'value', 'side'
    brk = brk.stack()
    brk.reset_index(inplace=True)

    # Add columns for other side
    brk2 = brk[['prediction_id', 'side', 'chromosome', 'strand', 'position']].copy()
    def swap_side(side):
        if side == '1':
            return '2'
        elif side == '2':
            return '1'
        else:
            raise ValueError()
    brk2['side'] = brk2['side'].apply(swap_side)
    brk2.rename(
        columns={
            'chromosome':'other_chromosome',
            'strand':'other_strand',
            'position':'other_position',
        },
        inplace=True
    )
    brk = brk.merge(brk2)

    # Annotate with copy number
    brk_cn = store['/solutions/solution_{0}/brk_cn'.format(solution)]
    brk_cn = brk_cn.groupby('prediction_id')[['cn_1', 'cn_2']].sum().reset_index()
    brk = brk.merge(brk_cn, on='prediction_id', how='left').fillna(0.0)

    # Annotate with strand related appearance    
    strand_angle = pd.DataFrame({'strand':['+', '-'], 'strand_angle':[math.pi/6., -math.pi/6.]})
    brk = brk.merge(strand_angle)

    # Calculate plot start and end
    brk = brk.merge(chromosome_plot_info[['chromosome', 'chromosome_plot_start']])
    brk['plot_position'] = brk['position'] + brk['chromosome_plot_start']

    # Annotate with clonal information
    brk['clone_1_color'] = np.where(brk['cn_1'] > 0, '00', 'ff')
    brk['clone_2_color'] = np.where(brk['cn_2'] > 0, '00', 'ff')
    brk['clonality_color'] = '#ff' + brk['clone_1_color'] + brk['clone_2_color']

    brk.sort_values(['prediction_id', 'side'], inplace=True)

    return brk


class gaussian_kde_set_covariance(scipy.stats.gaussian_kde):
    def __init__(self, dataset, covariance):
        self.covariance = covariance
        scipy.stats.gaussian_kde.__init__(self, dataset)
    def _compute_covariance(self):
        self.inv_cov = 1.0 / self.covariance
        self._norm_factor = np.sqrt(2*np.pi*self.covariance) * self.n


def weighted_density(xs, data, weights, cov):
    weights = weights.astype(float)
    resample_prob = weights / weights.sum()
    samples = np.random.choice(data, size=10000, replace=True, p=resample_prob)
    density = gaussian_kde_set_covariance(samples, cov)
    ys = density(xs)
    ys[0] = 0.0
    ys[-1] = 0.0
    return ys


def prepare_read_depth_data(patient, sample, solution):
    """
    """
    store = sample_stores[(patient, sample)]

    # Create read depth plot
    read_depth_df = store['read_depth']

    cov = 0.0000001

    read_depth_min = 0.0
    read_depth_max = np.percentile(read_depth_df['total'], 95)
    read_depths = [read_depth_min] + list(np.linspace(read_depth_min, read_depth_max, 2000)) + [read_depth_max]

    minor_density = weighted_density(read_depths, read_depth_df['minor'], read_depth_df['length'], cov)
    major_density = weighted_density(read_depths, read_depth_df['major'], read_depth_df['length'], cov)
    total_density = weighted_density(read_depths, read_depth_df['total'], read_depth_df['length'], cov)

    data = pd.DataFrame({
        'read_depth': read_depths,
        'minor_density': minor_density,
        'major_density': major_density,
        'total_density': total_density,
    })

    return data


def build_genome_panel(cnv_source, brk_source, chromosome_plot_info, width=1000):
    """
    """
    init_x_range = [0, chromosome_plot_info['chromosome_plot_end'].max()]

    scatter_plot = major_minor_scatter_plot(cnv_source)
    line_plot1 = major_minor_segment_plot(cnv_source, 'major_raw', 'minor_raw', init_x_range, 'raw', width)
    line_plot2 = major_minor_segment_plot(cnv_source, 'major_raw_e', 'minor_raw_e', line_plot1.x_range, 'expected', width)
    line_plot3 = major_minor_segment_plot(cnv_source, 'major_1', 'minor_1', line_plot1.x_range, 'clone 1', width)
    line_plot4 = major_minor_segment_plot(cnv_source, 'major_2', 'minor_2', line_plot1.x_range, 'clone 2', width)
    line_plot5 = major_minor_segment_plot(cnv_source, 'major_diff', 'minor_diff', line_plot1.x_range, 'clone diff', width)
    brk_plot = breakpoints_plot(brk_source, line_plot1.x_range, width)

    for p in [line_plot1, line_plot2, line_plot3, line_plot4, line_plot5, brk_plot]:
        setup_genome_plot_axes(p, chromosome_plot_info)

    columns = ['prediction_id',
        'chromosome', 'position', 'strand',
        'cn_1', 'cn_2']
    columns = [TableColumn(field=a, title=a, width=10) for a in columns]
    data_table = DataTable(source=brk_source, columns=columns, width=1000, height=1000)

    panel = Panel(title='Genome View', closable=False)
    panel.child = VBox(scatter_plot, line_plot1, line_plot2, line_plot3, line_plot4, line_plot5, brk_plot, data_table)

    return panel


def build_split_plots(cnv_source, brk_source, chromosome_plot_info, brk_view, width=500):
    """
    """
    init_x_range = [0, chromosome_plot_info['chromosome_plot_end'].max()]

    line_plot1 = major_minor_segment_plot(cnv_source, 'major_raw', 'minor_raw', init_x_range, 'raw', width)
    line_plot2 = major_minor_segment_plot(cnv_source, 'major_raw_e', 'minor_raw_e', line_plot1.x_range, 'expected', width)
    line_plot3 = major_minor_segment_plot(cnv_source, 'major_1', 'minor_1', line_plot1.x_range, 'clone 1', width)
    line_plot4 = major_minor_segment_plot(cnv_source, 'major_2', 'minor_2', line_plot1.x_range, 'clone 2', width)

    brk_plot = breakpoints_plot(brk_source, brk_view, line_plot1.x_range, width)

    for p in [line_plot1, line_plot2, line_plot3, line_plot4]:
        setup_chromosome_plot_axes(p)

    return VBox(line_plot1, line_plot2, line_plot3, line_plot4, brk_plot)


def build_split_panel(cnv_source_left, cnv_source_right, brk_source, chromosome_plot_info):
    """
    """
    plots1 = build_split_plots(cnv_source_left, brk_source, chromosome_plot_info['left'], 'left')
    plots2 = build_split_plots(cnv_source_right, brk_source, chromosome_plot_info['right'], 'right')

    plots = hplot(*[plots1, plots2])

    columns = ['prediction_id',
        'chromosome_1', 'position_1', 'strand_1',
        'chromosome_2', 'position_2', 'strand_2',
        'cn_1', 'cn_2']
    columns = [TableColumn(field=a, title=a, width=10) for a in columns]
    data_table = DataTable(source=brk_source, columns=columns, width=1000, height=1000)

    panel = Panel(title='Split View', closable=False)
    panel.child = VBox(plots, data_table)

    return panel


def build_solutions_panel(patient, sample, solutions_source, read_depth_source):
    # Create solutions table
    solutions_columns = ['decreased_log_posterior', 'graph_opt_iter', 'h_converged',
       'h_em_iter', 'log_posterior', 'log_posterior_graph', 'num_clones',
       'num_segments', 'idx', 'bic', 'bic_optimal', 'ploidy',
       'haploid_normal', 'haploid_tumour', 'clone_1_fraction', 'clone_2_fraction']
    columns = [TableColumn(field=a, title=a) for a in solutions_columns]
    solutions_table = DataTable(source=solutions_source, columns=columns, width=1000, height=500)

    readdepth_plot = Figure(
        title='major/minor/total read depth',
        plot_width=1000, plot_height=300,
        tools='pan,wheel_zoom,reset',
        logo=None,
        title_text_font_size=value('10pt'),
    )

    readdepth_plot.patch('read_depth', 'minor_density', color='blue', alpha=0.5, source=read_depth_source)
    readdepth_plot.patch('read_depth', 'major_density', color='red', alpha=0.5, source=read_depth_source)
    readdepth_plot.patch('read_depth', 'total_density', color='grey', alpha=0.5, source=read_depth_source)

    readdepth_plot.circle(x='haploid_normal', y=0, size=10, source=solutions_source, color='orange')
    readdepth_plot.circle(x='haploid_tumour_mode', y=0, size=10, source=solutions_source, color='green')

    panel = Panel(title='Solutions View', closable=False)
    panel.child = VBox(solutions_table, readdepth_plot)

    return panel


# Make inputs
patient_select = Select(
    title="Patient:",
    name='patients',
)

patient_select.options = patient_samples.keys()
patient_select.value = patient_select.options[0]
patient = patient_select.value

sample_select = Select(
    title="Sample:",
    name='patients',
)

sample_select.options = patient_samples[patient]
sample_select.value = sample_select.options[0]
sample = sample_select.value

solution_select = Select(
    title="Solution:",
    name='solutions',
)

solution_select.options = retrieve_solutions(patient, sample)
solution_select.value = solution_select.options[0]
solution = solution_select.value

# Make data sources
cnv_source = ColumnDataSource()
brk_source = ColumnDataSource()
solutions_source = ColumnDataSource()
read_depth_source = ColumnDataSource()


def reset_sample():
    global sample
    sample_select.options = patient_samples[patient]
    sample_select.value = sample_select.options[0]
    sample = sample_select.value


def reset_solution():
    global solution
    solution_select.options = retrieve_solutions(patient, sample)
    solution_select.value = solution_select.options[0]
    solution = solution_select.value


# Chromosome plot info will be based on first sample loaded
chromosome_plot_info = retrieve_chromosome_plot_info(patient, sample, solution)


def update_data():
    cnv = retrieve_cnv_data(patient, sample, solution)
    cnv_data = prepare_cnv_data(cnv, chromosome_plot_info)

    brk_data = retrieve_brk_data(patient, sample, solution, chromosome_plot_info)

    solutions_data = retrieve_solution_data(patient, sample)

    read_depth_data = prepare_read_depth_data(patient, sample, solution)

    assert cnv_data.notnull().all().all()
    assert brk_data.notnull().all().all()
    assert solutions_data.notnull().all().all()
    assert read_depth_data.notnull().all().all()

    cnv_source.data = cnv_data.to_dict(orient='list')
    brk_source.data = brk_data.to_dict(orient='list')
    solutions_source.data = solutions_data.to_dict(orient='list')
    read_depth_source.data = read_depth_data.to_dict(orient='list')
    

def update_patient(attr, old, new):
    global patient
    patient = new
    reset_sample()
    reset_solution()
    update_data()


def update_sample(attr, old, new):
    global sample
    sample = new
    reset_solution()
    update_data()


def update_solution(attr, old, new):
    global solution
    solution = new
    update_data()


# Add callbacks
patient_select.on_change('value', update_patient)
sample_select.on_change('value', update_sample)
solution_select.on_change('value', update_solution)

# Create main interface
tabs = Tabs()
tabs.tabs.append(build_solutions_panel(patient, sample, solutions_source, read_depth_source))
tabs.tabs.append(build_genome_panel(cnv_source, brk_source, chromosome_plot_info))
# self.tabs.tabs.append(build_split_panel(self.cnv_source_left, self.cnv_source_right, self.brk_source, self._chromosome_plot_info))

input_box = VBoxForm(patient_select, sample_select, solution_select)
main_box = HBox(input_box, tabs)

update_data()

curdoc().add_root(main_box)

