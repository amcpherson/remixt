import argparse
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

import remixt.visualize


def setup_chromosome_plot_axes(p):
    """
    """
    p.xaxis[0].formatter = NumeralTickFormatter(format='0.00a')


def retrieve_solution_data(store):
    """
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


def retrieve_solutions(store):
    """
    """
    return list(store['stats']['init_id'].astype(str).values)


def retrieve_cnv_data(store, solution, chromosome=''):
    """
    """
    cnv = store['solutions/solution_{0}/cn'.format(solution)]

    if chromosome != '':
        cnv = cnv[cnv['chromosome'] == chromosome].copy()

    cnv['segment_idx'] = cnv.index

    return cnv


def retrieve_chromosome_plot_info(store, solution, chromosome=''):
    """
    """
    cnv = retrieve_cnv_data(store, solution, chromosome)

    return remixt.visualize.create_chromosome_plot_info(cnv, chromosome=chromosome)


def retrieve_brk_data(store, solution, chromosome_plot_info):
    """
    """
    brk = store['breakpoints']
    brk_cn = store['/solutions/solution_{0}/brk_cn'.format(solution)]

    return remixt.visualize.prepare_brk_data(brk, brk_cn, chromosome_plot_info)


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


def prepare_read_depth_data(store, solution):
    """
    """
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


def build_solutions_panel(solutions_source, read_depth_source):
    # Create solutions table
    solutions_columns = [
        ('init_id', NumberFormatter(format='0')),
        ('log_likelihood', NumberFormatter(format='0.000')),
        ('ploidy', NumberFormatter(format='0.000')),
        ('prop_subclonal', NumberFormatter(format='0.000')),
        ('haploid_normal', NumberFormatter(format='0.000')),
        ('haploid_tumour', NumberFormatter(format='0.000')),
        ('clone_1_fraction', NumberFormatter(format='0.000')),
        ('clone_2_fraction', NumberFormatter(format='0.000')),
        ('divergence_weight', None),
    ]
    columns = [TableColumn(field=a, title=a, formatter=f) for a, f in solutions_columns]
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


def create_source_select(sources, title, name):
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
        for s_name, s_data in from_sources.iteritems():
            callback_args['source_{}_{}'.format(idx, s_name)] = s_data

    callback = CustomJS(args=callback_args, code=callback_code)

    source_select = Select(
        title=title,
        name=name,
    )
    
    source_select.options = from_sources.keys()
    source_select.value = initial_value
    source_select.callback = callback
    
    return source_select


def create_cnv_brk_sources(store, solution, chromosome_plot_info):
    cnv = retrieve_cnv_data(store, solution)
    cnv_data = remixt.visualize.prepare_cnv_data(cnv, chromosome_plot_info)
    brk_data = retrieve_brk_data(store, solution, chromosome_plot_info)

    assert cnv_data.notnull().all().all()
    assert brk_data.notnull().all().all()

    cnv_source = ColumnDataSource(cnv_data)
    brk_source = ColumnDataSource(brk_data)

    return cnv_source, brk_source


from bokeh.plotting import hplot, figure, output_file, show

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('results',
        help='Results to visualize')

    argparser.add_argument('html',
        help='HTML output visualization')

    args = vars(argparser.parse_args())

    with pd.HDFStore(args['results'], 'r') as store:

        output_file(args['html'])

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

        solutions_source = ColumnDataSource(solutions_data)
        read_depth_source = ColumnDataSource(read_depth_data)

        solution_select = create_source_select(
            [
                (cnv_selected_source, cnv_solution_sources),
                (brk_selected_source, brk_solution_sources),
            ],
            "Solution:",
            'solutions',
        )

        # Create main interface
        tabs = Tabs()
        tabs.tabs.append(build_solutions_panel(solutions_source, read_depth_source))
        tabs.tabs.append(remixt.visualize.build_genome_panel(cnv_selected_source, brk_selected_source, chromosome_plot_info))
        input_box = VBoxForm(solution_select)
        main_box = HBox(input_box, tabs)

        show(main_box)


