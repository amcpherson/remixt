import os
import numpy as np
import pandas as pd
import math
import matplotlib.colors
import matplotlib.pyplot
import bokeh.models
import bokeh.plotting
import bokeh.core.properties
import scipy.stats

import remixt.utils


chromosomes = [str(a) for a in range(1, 23)] + ['X']


def _create_chromosome_colors(chromosomes):
    color_map = matplotlib.pyplot.get_cmap('Set1')
    chromosome_colors = list()
    for i in range(len(chromosomes)):
        rgb_color = color_map(float(i) / float(len(chromosomes)))
        hex_color = matplotlib.colors.rgb2hex(rgb_color)
        chromosome_colors.append(hex_color)
    chromosome_colors = pd.DataFrame({'chromosome': chromosomes, 'scatter_color': chromosome_colors})
    return chromosome_colors

chromosome_colors = _create_chromosome_colors(chromosomes)


def _create_chromosome_indices(chromosomes):
    chromosome_indices = dict([(chromosome, idx) for idx, chromosome in enumerate(chromosomes)])
    return chromosome_indices

chromosome_indices = _create_chromosome_indices(chromosomes)


def major_minor_scatter_plot(source):
    """ Plot a major / minor scatter plot from a copy number data source
    """
    p = bokeh.plotting.Figure(
        title='raw major vs minor',
        plot_width=1000, plot_height=500,
        tools='pan,wheel_zoom,box_select,reset,lasso_select',
        logo=None,
        x_range=[-0.5, 6.5],
        y_range=[-0.5, 4.5],
    )

    p.title.text_font_size=bokeh.core.properties.value('10pt')

    p.circle(
        x='major_raw', y='minor_raw',
        size='scatter_size', color='scatter_color', alpha=0.5,
        source=source,
    )

    return p


def major_minor_segment_plot(source, major_column, minor_column, x_range, name, width=1000):
    """ Plot a major / minor line plot from a copy number data source
    """
    hover = bokeh.models.HoverTool(
        tooltips=[
            ('segment_idx', '@segment_idx'),
            ('chromosome', '@chromosome'),
            ('start', '@start'),
            ('end', '@end'),
            ('major_raw', '@major_raw'),
            ('minor_raw', '@minor_raw'),
        ]
    )

    tools = [
        bokeh.models.PanTool(dimensions='width'),
        bokeh.models.WheelZoomTool(dimensions='width'),
        bokeh.models.BoxZoomTool(),
        bokeh.models.BoxSelectTool(),
        bokeh.models.ResetTool(),
        bokeh.models.TapTool(),
        hover,
    ]

    p = bokeh.plotting.Figure(
        title=name + ' chromosome major/minor',
        plot_width=width, plot_height=200,
        tools=tools,
        toolbar_location='above',
        x_range=x_range,
        y_range=[-0.5, 6.5],
    )

    p.title.text_font_size = bokeh.core.properties.value('10pt')

    p.quad(
        top=major_column, bottom=0, left='plot_start', right='plot_end',
        source=source, color='red', alpha=0.05, line_width=0)

    p.quad(
        top=minor_column, bottom=0, left='plot_start', right='plot_end',
        source=source, color='blue', alpha=0.05, line_width=0)

    p.segment(
        y0=major_column, y1=major_column, x0='plot_start', x1='plot_end',
        source=source, color='red', alpha=1.0, line_width=4)

    p.segment(
        y0=minor_column, y1=minor_column, x0='plot_start', x1='plot_end',
        source=source, color='blue', alpha=1.0, line_width=2)

    return p


def breakpoints_plot(source, x_range, width=1000):
    """ Plot break ends from a breakpoint source
    """
    hover = bokeh.models.HoverTool(
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
        bokeh.models.PanTool(dimensions='width'),
        bokeh.models.WheelZoomTool(dimensions='width'),
        bokeh.models.BoxSelectTool(),
        bokeh.models.ResetTool(),
        bokeh.models.TapTool(),
        hover,
    ]

    p = bokeh.plotting.Figure(
        title='break ends',
        plot_width=width, plot_height=150,
        tools=tools,
        logo=None,
        x_range=x_range,
        y_range=['+', '-'],
    )

    p.title.text_font_size = bokeh.core.properties.value('10pt')

    p.triangle(
        x='plot_position', y='strand', size=10, angle='strand_angle',
        line_color='grey', fill_color='clonality_color', alpha=1.0,
        source=source)

    return p


def setup_genome_plot_axes(p, chromosome_plot_info):
    """ Configure axes of a genome view
    """
    chromosomes = list(chromosome_plot_info['chromosome'].values)
    chromosome_bounds = [0] + list(chromosome_plot_info['chromosome_plot_end'].values)
    chromosome_mids = list(chromosome_plot_info['chromosome_plot_mid'].values)

    p.xgrid.ticker = bokeh.models.FixedTicker(ticks=[-1] + chromosome_bounds + [chromosome_bounds[-1] + 1])
    p.xgrid.band_fill_alpha = 0.1
    p.xgrid.band_fill_color = "navy"

    p.xaxis[0].ticker = bokeh.models.FixedTicker(ticks=chromosome_bounds)
    p.xaxis[0].major_label_text_font_size = bokeh.core.properties.value('0pt')

    p.text(x=chromosome_mids, y=-0.5, text=chromosomes, text_font_size=bokeh.core.properties.value('0.5em'), text_align='center')


def create_chromosome_plot_info(cnv, chromosome=''):
    """ Create information about chromosome start ends for genome view
    """
    cnv['chromosome_index'] = cnv['chromosome'].apply(lambda a: chromosome_indices[a])
    cnv.sort_values(['chromosome_index', 'start'], inplace=True)

    info = (
        cnv.groupby('chromosome', sort=False)['end']
        .max().reset_index().rename(columns={'end': 'chromosome_length'}))

    info['chromosome_plot_end'] = np.cumsum(info['chromosome_length'])
    info['chromosome_plot_start'] = info['chromosome_plot_end'].shift(1)
    info.loc[info.index[0], 'chromosome_plot_start'] = 0
    info['chromosome_plot_mid'] = 0.5 * (info['chromosome_plot_start'] + info['chromosome_plot_end'])

    return info


def prepare_cnv_data(cnv, chromosome_plot_info, smooth_segments=False):
    """ Prepare copy number data for loading in the a data source
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


def prepare_brk_data(brk_cn, chromosome_plot_info):
    """ Prepare breakpoint data for loading into a breakpoint source
    """
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
    brk_cn['type'] = brk_cn.apply(calculate_breakpoint_type, axis=1)

    # Duplicate required columns before stack
    brk_cn['type_1'] = brk_cn['type']
    brk_cn['type_2'] = brk_cn['type']

    # Stack break ends
    brk_ends = brk_cn[[
        'prediction_id',
        'chromosome_1', 'strand_1', 'position_1',
        'chromosome_2', 'strand_2', 'position_2',
    ]]
    brk_ends.set_index(['prediction_id'], inplace=True)
    brk_ends = brk_ends.filter(regex='(_1|_2)')
    def split_col_name(col):
        parts = col.split('_')
        return '_'.join(parts[:-1]), parts[-1]
    brk_ends.columns = pd.MultiIndex.from_tuples([split_col_name(col) for col in brk_ends.columns])
    brk_ends.columns.names = 'value', 'side'
    brk_ends = brk_ends.stack()
    brk_ends.reset_index(inplace=True)

    # Add columns for other side
    brk_ends_2 = brk_ends[['prediction_id', 'side', 'chromosome', 'strand', 'position']].copy()
    def swap_side(side):
        if side == '1':
            return '2'
        elif side == '2':
            return '1'
        else:
            raise ValueError()
    brk_ends_2['side'] = brk_ends_2['side'].apply(swap_side)
    brk_ends_2.rename(
        columns={
            'chromosome': 'other_chromosome',
            'strand': 'other_strand',
            'position': 'other_position',
        },
        inplace=True
    )
    brk_ends = brk_ends.merge(brk_ends_2)

    # Annotate with copy number
    brk_ends = brk_ends.merge(brk_cn[['prediction_id', 'cn_1', 'cn_2']], on='prediction_id')

    # Annotate with strand related appearance
    strand_angle = pd.DataFrame({'strand': ['+', '-'], 'strand_angle': [math.pi / 6., -math.pi / 6.]})
    brk_ends = brk_ends.merge(strand_angle)

    # Calculate plot start and end
    brk_ends = brk_ends.merge(chromosome_plot_info[['chromosome', 'chromosome_plot_start']])
    brk_ends['plot_position'] = brk_ends['position'] + brk_ends['chromosome_plot_start']

    # Annotate with clonal information
    brk_ends['clone_1_color'] = np.where(brk_ends['cn_1'] > 0, '00', 'ff')
    brk_ends['clone_2_color'] = np.where(brk_ends['cn_2'] > 0, '00', 'ff')
    brk_ends['clonality_color'] = '#ff' + brk_ends['clone_1_color'] + brk_ends['clone_2_color']

    brk_ends.sort_values(['prediction_id', 'side'], inplace=True)

    return brk_ends


def build_genome_panel(cnv_source, brk_source, chromosome_plot_info, width=1000):
    """ Build the genome pannel with scatter, line and break end plots and breakpoint table
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

    columns = [
        'prediction_id',
        'chromosome', 'position', 'strand',
        'cn_1', 'cn_2']
    columns = [bokeh.models.TableColumn(field=a, title=a, width=10) for a in columns]
    data_table = bokeh.models.DataTable(source=brk_source, columns=columns, width=1000, height=1000)

    panel = bokeh.models.Panel(title='Genome View', closable=False)
    panel.child = bokeh.models.VBox(scatter_plot, line_plot1, line_plot2, line_plot3, line_plot4, line_plot5, brk_plot, data_table)

    return panel


def create_genome_visualization(cn, brk_cn, html_filename):
    """ Create a genome visualization and output to an html file
    """
    try:
        os.remove(html_filename)
    except OSError:
        pass
    bokeh.plotting.output_file(html_filename)

    chromosome_plot_info = create_chromosome_plot_info(cn)
    cnv_data = prepare_cnv_data(cn, chromosome_plot_info)
    brk_data = prepare_brk_data(brk_cn, chromosome_plot_info)

    cnv_source = bokeh.models.ColumnDataSource(cnv_data)
    brk_source = bokeh.models.ColumnDataSource(brk_data)

    tabs = bokeh.models.Tabs()
    tabs.tabs.append(build_genome_panel(cnv_source, brk_source, chromosome_plot_info))
    main_box = bokeh.models.HBox(tabs)

    bokeh.plotting.save(main_box)


def retrieve_solutions(store):
    """ Retrieve a list of solutions from the data store
    """
    return list(store['stats']['init_id'].astype(str).values)


def retrieve_cnv_data(store, solution, chromosome=''):
    """ Retrieve copy number data for a specific solution
    """
    cnv = store['solutions/solution_{0}/cn'.format(solution)]

    if chromosome != '':
        cnv = cnv[cnv['chromosome'] == chromosome].copy()

    cnv['segment_idx'] = cnv.index

    return cnv


def retrieve_brk_data(store, solution, chromosome_plot_info):
    """ Retrieve breakpoint copy number data for a specific solution
    """
    brk_cn = store['/solutions/solution_{0}/brk_cn'.format(solution)]

    return prepare_brk_data(brk_cn, chromosome_plot_info)


def retrieve_chromosome_plot_info(store, solution, chromosome=''):
    """ Retrieve chromosome plot info for a specific solution
    """
    cnv = retrieve_cnv_data(store, solution, chromosome)

    return create_chromosome_plot_info(cnv, chromosome=chromosome)


def create_cnv_brk_sources(store, solution, chromosome_plot_info):
    """ Create ColumnDataSource for copy number and breakpoints given a specific solution
    """
    cnv = retrieve_cnv_data(store, solution)
    cnv_data = prepare_cnv_data(cnv, chromosome_plot_info)
    brk_data = retrieve_brk_data(store, solution, chromosome_plot_info)

    assert cnv_data.notnull().all().all()
    assert brk_data.notnull().all().all()

    cnv_source = bokeh.models.ColumnDataSource(cnv_data)
    brk_source = bokeh.models.ColumnDataSource(brk_data)

    return cnv_source, brk_source


def retrieve_solution_data(store):
    """ Retrieve solution data from the data store
    """
    solutions_df = store['stats']

    for idx, row in solutions_df.iterrows():

        # Calculate ploidy
        cnv = retrieve_cnv_data(store, row['init_id'])
        cnv = cnv.replace([np.inf, -np.inf], np.nan).dropna()
        ploidy = (cnv['length'] * (cnv['major_raw_e'] + cnv['minor_raw_e'])).sum() / cnv['length'].sum()
        solutions_df.loc[idx, 'ploidy'] = ploidy

        # Calculate proportion subclonal
        subclonal = (
            ((cnv['major_1'] != cnv['major_2']) * 1) +
            ((cnv['minor_1'] != cnv['minor_2']) * 1))
        prop_subclonal = (subclonal * cnv['length']).sum() / (2. * cnv['length'].sum())
        solutions_df.loc[idx, 'prop_subclonal'] = prop_subclonal

        # Add haploid normal/tumour depth and clone fraction
        h = store['/solutions/solution_{0}/h'.format(row['init_id'])]
        solutions_df.loc[idx, 'haploid_normal'] = h.values[0]
        solutions_df.loc[idx, 'haploid_tumour'] = h.values[1:].sum()
        solutions_df.loc[idx, 'haploid_tumour_mode'] = h.values.sum()
        solutions_df.loc[idx, 'clone_1_fraction'] = h.values[1] / h.values[1:].sum()
        solutions_df.loc[idx, 'clone_2_fraction'] = 1. - solutions_df.loc[idx, 'clone_1_fraction']

    return solutions_df


class gaussian_kde_set_covariance(scipy.stats.gaussian_kde):
    def __init__(self, dataset, covariance):
        self.covariance = covariance
        scipy.stats.gaussian_kde.__init__(self, dataset)

    def _compute_covariance(self):
        self.inv_cov = 1.0 / self.covariance
        self._norm_factor = np.sqrt(2 * np.pi * self.covariance) * self.n


def _weighted_density(xs, data, weights, cov):
    weights = weights.astype(float)
    resample_prob = weights / weights.sum()
    samples = np.random.choice(data, size=10000, replace=True, p=resample_prob)
    density = gaussian_kde_set_covariance(samples, cov)
    ys = density(xs)
    ys[0] = 0.0
    ys[-1] = 0.0
    return ys


def prepare_read_depth_data(store, solution):
    """ Prepare read depth data for plotting
    """
    read_depth_df = store['read_depth']

    cov = 0.0000001

    read_depth_min = 0.0
    read_depth_max = remixt.utils.weighted_percentile(read_depth_df['total'].values, read_depth_df['length'].values, 95)
    read_depths = [read_depth_min] + list(np.linspace(read_depth_min, read_depth_max, 2000)) + [read_depth_max]

    minor_density = _weighted_density(read_depths, read_depth_df['minor'], read_depth_df['length'], cov)
    major_density = _weighted_density(read_depths, read_depth_df['major'], read_depth_df['length'], cov)
    total_density = _weighted_density(read_depths, read_depth_df['total'], read_depth_df['length'], cov)

    data = pd.DataFrame({
        'read_depth': read_depths,
        'minor_density': minor_density,
        'major_density': major_density,
        'total_density': total_density,
    })

    return data


def build_solutions_panel(solutions_source, read_depth_source):
    """ Build an overview of solutions including a read depth plot
    """
    solutions_columns = [
        ('init_id', bokeh.models.NumberFormatter(format='0')),
        ('elbo', bokeh.models.NumberFormatter(format='0.000')),
        ('ploidy', bokeh.models.NumberFormatter(format='0.000')),
        ('prop_subclonal', bokeh.models.NumberFormatter(format='0.000')),
        ('haploid_normal', bokeh.models.NumberFormatter(format='0.000')),
        ('haploid_tumour', bokeh.models.NumberFormatter(format='0.000')),
        ('clone_1_fraction', bokeh.models.NumberFormatter(format='0.000')),
        ('clone_2_fraction', bokeh.models.NumberFormatter(format='0.000')),
        ('divergence_weight', None),
    ]
    columns = [bokeh.models.TableColumn(field=a, title=a, formatter=f) for a, f in solutions_columns]
    solutions_table = bokeh.models.DataTable(source=solutions_source, columns=columns, width=1000, height=500)

    readdepth_plot = bokeh.plotting.Figure(
        title='major/minor/total read depth',
        plot_width=1000, plot_height=300,
        tools='pan,wheel_zoom,reset',
        logo=None,
    )

    readdepth_plot.title.text_font_size = bokeh.core.properties.value('10pt')

    readdepth_plot.patch('read_depth', 'minor_density', color='blue', alpha=0.5, source=read_depth_source)
    readdepth_plot.patch('read_depth', 'major_density', color='red', alpha=0.5, source=read_depth_source)
    readdepth_plot.patch('read_depth', 'total_density', color='grey', alpha=0.5, source=read_depth_source)

    readdepth_plot.circle(x='haploid_normal', y=0, size=10, source=solutions_source, color='orange')
    readdepth_plot.circle(x='haploid_tumour_mode', y=0, size=10, source=solutions_source, color='green')

    panel = bokeh.models.Panel(title='Solutions View', closable=False)
    panel.child = bokeh.models.VBox(solutions_table, readdepth_plot)

    return panel


def create_source_select(sources, title, name):
    """ Create a general data source selection widget

    Args:
        sources(list): selected and selectable sources
        title(str): title of widget
        name(str): name of widget

    Returns:
        bokeh.models.Select: selection widget
    
    The sources are provided as a list of tuples, where each tuple has length 2.  The first item in
    the tuple is the datasource the user has selected, and the second item is a dictionary of possible data
    sources keyed by an id for the source.
    """
    names = sources[0][1].keys()
    initial_value = names[0]
    
    callback_code = "var t = cb_obj.get('value');\n"
    callback_args = {}

    for idx, (to_source, from_sources) in enumerate(sources):
        to_source.data = from_sources[initial_value].data

        for s_name in names:
            callback_code += "if (t == '{}') {{\n".format(s_name)
            callback_code += "  var d = source_{}_{}.get('data');\n".format(idx, s_name)
            callback_code += "}\n"

        callback_code += "source_{}.set('data', d);\n".format(idx)
        callback_code += "source_{}.trigger('change');\n".format(idx)

        callback_args["source_{}".format(idx)] = to_source
        for s_name, s_data in from_sources.items():
            callback_args['source_{}_{}'.format(idx, s_name)] = s_data

    callback = bokeh.models.CustomJS(args=callback_args, code=callback_code)

    source_select = bokeh.models.Select(
        title=title,
        name=name,
    )
    
    source_select.options = from_sources.keys()
    source_select.value = initial_value
    source_select.callback = callback
    
    return source_select


def create_solutions_visualization(results_filename, html_filename):
    """ Create a multi-tab visualization of remixt solutions
    """
    try:
        os.remove(html_filename)
    except OSError:
        pass
    bokeh.plotting.output_file(html_filename)

    with pd.HDFStore(results_filename, 'r') as store:
        solutions = list(retrieve_solutions(store))

        chromosome_plot_info = retrieve_chromosome_plot_info(store, solutions[0])
        cnv_selected_source, brk_selected_source = create_cnv_brk_sources(store, solutions[0], chromosome_plot_info)

        cnv_solution_sources = {}
        brk_solution_sources = {}
        for solution in solutions:
            cnv_source, brk_source = create_cnv_brk_sources(store, solution, chromosome_plot_info)

            cnv_solution_sources[solution] = cnv_source
            brk_solution_sources[solution] = brk_source

        solutions_data = retrieve_solution_data(store)
        read_depth_data = prepare_read_depth_data(store, solution)

    assert solutions_data.notnull().all().all()
    assert read_depth_data.notnull().all().all()

    solutions_source = bokeh.models.ColumnDataSource(solutions_data)
    read_depth_source = bokeh.models.ColumnDataSource(read_depth_data)

    solution_select = create_source_select(
        [
            (cnv_selected_source, cnv_solution_sources),
            (brk_selected_source, brk_solution_sources),
        ],
        "Solution:",
        'solutions',
    )

    # Create main interface
    tabs = bokeh.models.Tabs()
    tabs.tabs.append(build_solutions_panel(solutions_source, read_depth_source))
    tabs.tabs.append(build_genome_panel(cnv_selected_source, brk_selected_source, chromosome_plot_info))
    input_box = bokeh.models.WidgetBox(solution_select)
    main_box = bokeh.models.HBox(input_box, tabs)

    bokeh.plotting.save(main_box)



