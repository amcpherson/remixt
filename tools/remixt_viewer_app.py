"""
This file demonstrates a bokeh applet, which can either be viewed
directly on a bokeh-server, or embedded into a flask application.
See the README.md file in this directory for instructions on running.
"""

import logging

logging.basicConfig(level=logging.DEBUG)

import collections
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot
import matplotlib.colors

from bokeh.models import ColumnDataSource, Plot, FixedTicker, NumeralTickFormatter, BasicTicker
from bokeh.plotting import figure, curdoc, gridplot
from bokeh.charts import Scatter
from bokeh.io import vplot, hplot
from bokeh.properties import String, Instance, Dict
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import HBox, VBox, VBoxForm, Paragraph, Select, DataTable, TableColumn


chromosomes = [str(a) for a in range(1, 23)] + ['X']

color_map = matplotlib.pyplot.get_cmap('Set1')
chromosome_colors = list()
for i in xrange(len(chromosomes)):
    rgb_color = color_map(float(i)/float(len(chromosomes)))
    hex_color = matplotlib.colors.rgb2hex(rgb_color)
    chromosome_colors.append(hex_color)
chromosome_colors = pd.DataFrame({'chromosome':chromosomes, 'scatter_color':chromosome_colors})

chromosome_indices = dict([(chromosome, idx) for idx, chromosome in enumerate(chromosomes)])


patient_samples = collections.defaultdict(list)
sample_stores = dict()


for data_filename in glob.glob('./patient_*/sample_*.h5'):
    patient = data_filename.split('/')[1][len('patient_'):]
    sample = data_filename.split('/')[2][len('sample_'):-len('.h5')]

    patient_samples[patient].append(sample)
    sample_stores[(patient, sample)] = pd.HDFStore(data_filename, 'r')


def major_minor_scatter_plot(source):
    """
    """
    p = figure(
        title='major vs minor',
        plot_width=1000, plot_height=500,
        tools='pan,wheel_zoom,box_select,reset',
        logo=None,
        title_text_font_size='10pt',
        x_range=[-0.5, 3],
        y_range=[-0.5, 3],
    )

    p.circle(x='major_raw', y='minor_raw',
        size='scatter_size', color='scatter_color', source=source
    )

    return p


def major_minor_segment_plot(source, major_column, minor_column, x_range, width=1000):
    """
    """
    p = figure(
        title='chromosome major/minor',
        plot_width=width, plot_height=200,
        tools='xpan,xwheel_zoom,box_select,reset',
        logo=None,
        title_text_font_size='10pt',
        x_range=x_range,
        y_range=[-0.5, 3],
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


def breakpoints_plot(source, view, x_range, width=1000):
    """
    """
    p = figure(
        title='major vs minor',
        plot_width=width, plot_height=150,
        tools='xpan,xwheel_zoom,box_select,reset,tap',
        logo=None,
        title_text_font_size='10pt',
        x_range=x_range,
        y_range=['+', '-'],
    )

    p.triangle(x='plot_position_'+view, y='plot_strand_'+view, size=10, angle='plot_strand_angle_'+view,
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
    p.xaxis[0].major_label_text_font_size = '0pt'

    p.text(x=chromosome_mids, y=-0.5, text=chromosomes, text_font_size='0.5em', text_align='center')


def retrieve_solutions(patient, sample):
    """
    """
    store = sample_stores[(patient, sample)]

    return list(store['stats']['idx'].astype(str).values)


def retrieve_cnv_data(patient, sample, solution, chromosome=''):
    """
    """
    store = sample_stores[(patient, sample)]

    cnv = store['solutions/{0}/cn'.format(solution)]

    if chromosome != '':
        cnv = cnv[cnv['chromosome'] == chromosome].copy()

    return cnv


def retrieve_chromosome_plot_info(patient, sample, solution, chromosome=''):
    """
    """
    cnv = retrieve_cnv_data(patient, sample, solution, chromosome)

    cnv['chromosome_index'] = cnv['chromosome'].apply(lambda a: chromosome_indices[a])
    cnv = cnv.sort(['chromosome_index', 'start'])

    info = (
        cnv.groupby('chromosome', sort=False)['end']
        .max().reset_index().rename(columns={'end':'chromosome_length'}))

    info['chromosome_plot_end'] = np.cumsum(info['chromosome_length'])
    info['chromosome_plot_start'] = info['chromosome_plot_end'].shift(1)
    info['chromosome_plot_start'].iloc[0] = 0
    info['chromosome_plot_mid'] = 0.5 * (info['chromosome_plot_start'] + info['chromosome_plot_end'])

    return info


def prepare_cnv_data(cnv, chromosome_plot_info):
    """
    """
    # Scatter size scaled by segment length        
    cnv['scatter_size'] = 5. * cnv['length'] / 3e6

    # Scatter color by chromosome
    cnv = cnv.merge(chromosome_colors)

    # Calculate plot start and end
    cnv = cnv.merge(chromosome_plot_info[['chromosome', 'chromosome_plot_start']])
    cnv['plot_start'] = cnv['start'] + cnv['chromosome_plot_start']
    cnv['plot_end'] = cnv['end'] + cnv['chromosome_plot_start']

    return cnv


def retrieve_brk_data(patient, sample, solution, left_chromosome, right_chromosome):
    """
    """
    store = sample_stores[(patient, sample)]

    brk = store['breakpoints']

    brk_cn = store['/solutions/{0}/brk_cn'.format(solution)]
    brk_cn = brk_cn.groupby('prediction_id')[['cn_1', 'cn_2']].sum().reset_index()
    brk = brk.merge(brk_cn, on='prediction_id', how='left').fillna(0.0)

    chromosomes = [left_chromosome, right_chromosome]
    brk = brk[
        (brk['chromosome_1'].isin(chromosomes)) |
        (brk['chromosome_2'].isin(chromosomes))]

    sides = ('1', '2')

    for side in sides:
        strand_angle = pd.DataFrame({'strand_'+side:['+', '-'], 'strand_angle_'+side:[math.pi/6., -math.pi/6.]})
        brk = brk.merge(strand_angle)

    plot_variables = ('chromosome', 'position', 'strand', 'strand_angle')
    for var in plot_variables:
        brk['plot_{0}_left'.format(var)] = brk['{0}_1'.format(var)]
        brk['plot_{0}_right'.format(var)] = brk['{0}_2'.format(var)]

    if left_chromosome == right_chromosome:
        flip = (brk['plot_position_left'] > brk['plot_position_right'])
    else:
        flip = (brk['plot_chromosome_left'] != left_chromosome)

    for var in plot_variables:
        cols = ['plot_{0}_left'.format(var), 'plot_{0}_right'.format(var)]
        brk.loc[flip, cols] = brk.loc[flip, cols[::-1]]

    brk.loc[(~brk['plot_chromosome_left'].isin(chromosomes)), 'plot_position_left'] = np.inf
    brk.loc[(~brk['plot_chromosome_right'].isin(chromosomes)), 'plot_position_right'] = np.inf

    brk['clone_1_color'] = np.where(brk['cn_1'] > 0, '00', 'ff')
    brk['clone_2_color'] = np.where(brk['cn_2'] > 0, '00', 'ff')
    brk['clonality_color'] = '#ff' + brk['clone_1_color'] + brk['clone_2_color']

    return brk


class RemixtApp(VBox):
    extra_generated_classes = [["RemixtApp", "RemixtApp", "VBox"]]
    jsmodel = "VBox"

    # plots
    scatter_plot = Instance(Plot)
    line_plot1 = Instance(Plot)
    line_plot2 = Instance(Plot)
    line_plot3 = Instance(Plot)

    # tables
    data_table = Instance(DataTable)

    # data sources
    cnv_source_full = Instance(ColumnDataSource)
    cnv_source_left = Instance(ColumnDataSource)
    cnv_source_right = Instance(ColumnDataSource)
    brk_source = Instance(ColumnDataSource)

    # layout boxes
    plot_column = Instance(VBox)
    split_panel = Instance(HBox)

    # inputs
    patient = String()
    patient_select = Instance(Select)
    sample = String
    sample_select = Instance(Select)
    solution = String(default="0")
    solution_select = Instance(Select)
    chromosome_left = String(default="")
    chromosome_left_select = Instance(Select)
    chromosome_right = String(default="")
    chromosome_right_select = Instance(Select)
    input_box = Instance(VBoxForm)

    def __init__(self, *args, **kwargs):
        super(RemixtApp, self).__init__(*args, **kwargs)
        self._dfs = {}
        self._chromosome_plot_mids = []
        self._chromosome_plot_bounds = [0]

    @classmethod
    def create(cls):
        """
        This function is called once, and is responsible for
        creating all objects (plots, datasources, etc)
        """
        # create layout widgets
        obj = cls()
        obj.plot_column = VBox()
        obj.split_panel = HBox()
        obj.input_box = VBoxForm()

        # create input widgets
        obj.make_inputs()

        # outputs
        obj.make_source()
        obj.make_plots()
        obj.make_tables()

        # layout
        obj.set_children()

        return obj

    def make_inputs(self):
        self.patient_select = Select(
            title="Patient:",
            name='patients',
        )

        self.patient_select.options = patient_samples.keys()
        self.patient_select.value = self.patient_select.options[0]
        self.patient = self.patient_select.value

        self.sample_select = Select(
            title="Sample:",
            name='patients',
        )

        self.sample_select.options = patient_samples[self.patient]
        self.sample_select.value = self.sample_select.options[0]
        self.sample = self.sample_select.value

        self.solution_select = Select(
            title="Solution:",
            name='solutions',
        )

        self.solution_select.options = retrieve_solutions(self.patient, self.sample)
        self.solution_select.value = self.solution_select.options[0]
        self.solution = self.solution_select.value

        self.chromosome_left_select = Select(
            title="Left chromosome:",
            name='chromosome',
            value=chromosomes[0],
            options=chromosomes,
        )
        self.chromosome_left = self.chromosome_left_select.value

        self.chromosome_right_select = Select(
            title="Right chromosome:",
            name='chromosome',
            value=chromosomes[0],
            options=chromosomes,
        )
        self.chromosome_right = self.chromosome_right_select.value


    def make_cnv_source(self, chromosome=''):
        """
        """
        if self.patient is None or self.sample is None:
            return

        chromosome_plot_info = retrieve_chromosome_plot_info(self.patient, self.sample, self.solution, chromosome)

        cnv = retrieve_cnv_data(self.patient, self.sample, self.solution, chromosome)
        cnv = prepare_cnv_data(cnv, chromosome_plot_info)

        return cnv, chromosome_plot_info


    def make_source(self):
        if self.patient is None or self.sample is None:
            return

        view_info = [
            ('', 'full'),
            (self.chromosome_left, 'left'),
            (self.chromosome_right, 'right'),
        ]

        self._chromosome_plot_info = dict()

        for view_chrom, view_name in view_info:
            cnv, chromosome_plot_info = self.make_cnv_source(view_chrom)
            self._chromosome_plot_info[view_name] = chromosome_plot_info
            if view_name == 'full':
                self.cnv_source_full = ColumnDataSource(cnv)
            elif view_name == 'left':
                self.cnv_source_left = ColumnDataSource(cnv)
            elif view_name == 'right':
                self.cnv_source_right = ColumnDataSource(cnv)

        brk = retrieve_brk_data(self.patient, self.sample, self.solution,
            self.chromosome_left, self.chromosome_right)

        self.brk_source = ColumnDataSource(brk)


    def make_plots(self):
        """
        """
        init_x_range = [0, self._chromosome_plot_info['full']['chromosome_plot_end'].max()]

        self.line_plot1 = major_minor_segment_plot(self.cnv_source_full, 'major_raw', 'minor_raw', init_x_range)

        self.line_plot2 = major_minor_segment_plot(self.cnv_source_full, 'major_1', 'minor_1', self.line_plot1.x_range)

        self.line_plot3 = major_minor_segment_plot(self.cnv_source_full, 'major_2', 'minor_2', self.line_plot1.x_range)

        setup_genome_plot_axes(self.line_plot1, self._chromosome_plot_info['full'])
        setup_genome_plot_axes(self.line_plot2, self._chromosome_plot_info['full'])
        setup_genome_plot_axes(self.line_plot3, self._chromosome_plot_info['full'])

        self.split_panel.children = [
            self.make_split_panel(self.cnv_source_left, 'left'),
            self.make_split_panel(self.cnv_source_right, 'right'),
        ]


    def make_split_panel(self, source, view, width=500):
        """
        """
        init_x_range = [0, self._chromosome_plot_info[view]['chromosome_plot_end'].max()]

        plots = []
        plots.append(major_minor_segment_plot(source, 'major_raw', 'minor_raw', init_x_range, width))
        plots.append(major_minor_segment_plot(source, 'major_1', 'minor_1', plots[0].x_range, width))
        plots.append(major_minor_segment_plot(source, 'major_2', 'minor_2', plots[0].x_range, width))

        plots.append(breakpoints_plot(self.brk_source, view, plots[0].x_range, width))

        for p in plots:
            setup_chromosome_plot_axes(p)

        return vplot(*plots)


    def make_tables(self):
        columns = ['prediction_id',
            'chromosome_1', 'position_1', 'strand_1',
            'chromosome_2', 'position_2', 'strand_2',
            'cn_1', 'cn_2']
        columns = [TableColumn(field=a, title=a, width=10) for a in columns]
        self.data_table = DataTable(source=self.brk_source, columns=columns, width=1000, height=1000)


    def set_children(self):
        self.children = [self.input_box, self.plot_column, self.data_table]
        self.plot_column.children = [self.line_plot1, self.line_plot2, self.line_plot3, self.split_panel]
        self.input_box.children = [self.patient_select, self.sample_select, self.solution_select, self.chromosome_left_select, self.chromosome_right_select]


    def input_change(self, obj, attrname, old, new):
        update_samples = False
        update_solutions = False
        if obj == self.patient_select:
            self.patient = new
            update_samples = True
            update_solutions = True
        if obj == self.sample_select:
            self.sample = new
            update_solutions = True
        if obj == self.solution_select:
            self.solution = new

        if update_samples:
            self.sample_select.options = patient_samples[self.patient]
            self.sample_select.value = self.sample_select.options[0]
            self.sample = self.sample_select.value

        if update_solutions:
            self.solution_select.options = retrieve_solutions(self.patient, self.sample)
            self.solution_select.value = self.solution_select.options[0]
            self.solution = self.solution_select.value

        self.make_source()
        self.make_plots()
        self.make_tables()
        self.set_children()
        curdoc().add(self)


    def chromosome_left_change(self, obj, attrname, old, new):
        self.chromosome_left = new
        self.make_source()
        self.make_plots()
        self.make_tables()
        self.set_children()
        curdoc().add(self)


    def chromosome_right_change(self, obj, attrname, old, new):
        self.chromosome_right = new
        self.make_source()
        self.make_plots()
        self.make_tables()
        self.set_children()
        curdoc().add(self)


    def setup_events(self):
        super(RemixtApp, self).setup_events()
        if self.patient_select:
            self.patient_select.on_change('value', self, 'input_change')
        if self.sample_select:
            self.sample_select.on_change('value', self, 'input_change')
        if self.solution_select:
            self.solution_select.on_change('value', self, 'input_change')
        if self.chromosome_left_select:
            self.chromosome_left_select.on_change('value', self, 'chromosome_left_change')
        if self.chromosome_right_select:
            self.chromosome_right_select.on_change('value', self, 'chromosome_right_change')


@bokeh_app.route("/remixt")
@object_page("remixt")
def make_stocks():
    app = RemixtApp.create()
    return app
