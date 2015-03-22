import os
import itertools
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pypeliner
import pypeliner.managed as mgd

import demix.simulations.pipeline
import demix.analysis.haplotype
import demix.wrappers
import demix.utils


demix_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
default_config_filename = os.path.join(demix_directory, 'defaultconfig.py')


if __name__ == '__main__':

    import run_comparison

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('sim_params',
        help='Simulation Parameters Filename')

    argparser.add_argument('install_dir',
        help='Tool installation directory')

    argparser.add_argument('results_table',
        help='Output Table Filename')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    config = {'ref_data_directory':args['ref_data_dir']}
    execfile(default_config_filename, {}, config)

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([demix, run_comparison], config)


    sim_params = dict()
    sim_params['chromosomes'] = ['20']
    sim_params['germline_seeds'] = range(10, 11)
    sim_params['genome_params'] = {'1':{}, '2':{}}
    sim_params['mixture_params'] = {'1':{}, '2':{}}


    # For each of n patients:
    #     * simulate germline alleles

    germline_axis = ('bygermline',)

    pyp.sch.setobj(mgd.TempOutputObj('germline_seed', *germline_axis), sim_params['germline_seeds'])
    pyp.sch.setobj(mgd.TempOutputObj('bychromosome'), sim_params['chromosomes'])

    pyp.sch.transform('simulate_germline_alleles', germline_axis, {'mem':8},
        demix.simulations.pipeline.simulate_germline_alleles,
        None,
        mgd.TempOutputFile('germline_alleles', *germline_axis),
        mgd.TempInputObj('germline_seed', *germline_axis),
        mgd.TempInputObj('bychromosome'),
        config,
    )

    # For each of n patients:
    #     For each of m genomes:
    #         * clonal divergence
    #         * normal dataset

    genome_axis = germline_axis + ('bygenome',)

    pyp.sch.setobj(mgd.TempOutputObj('genome_params', *genome_axis), sim_params['genome_params'], genome_axis[:-1])

    pyp.sch.transform('simulate_genomes', genome_axis, {'mem':4},
        run_comparison.simulate_genomes,
        None,
        mgd.TempOutputFile('genomes', *genome_axis),
        mgd.TempOutputFile('segment.tsv', *genome_axis),
        mgd.TempOutputFile('perfect_segment.tsv', *genome_axis),
        mgd.TempOutputFile('breakpoint.tsv', *genome_axis),
        mgd.TempInputObj('genome_params', *genome_axis),
    )

    pyp.sch.transform('simulate_normal_data', genome_axis, {'mem':24},
        demix.simulations.pipeline.simulate_normal_data,
        None,
        mgd.TempOutputFile('normal', *genome_axis),
        mgd.TempInputFile('mixture', *genome_axis),
        mgd.TempInputFile('germline_alleles', *germline_axis),
        mgd.TempFile('normal_tmp', *genome_axis),
        mgd.TempInputObj('genome_params', *genome_axis),
    )

    # For each of n patients:
    #     For each of m genomes:
    #         For each of l mixtures:
    #             * tumour dataset

    mixture_axis = genome_axis + ('bymixture',)

    pyp.sch.setobj(mgd.TempOutputObj('mixture_params', *genome_axis), sim_params['mixture_params'], genome_axis[:-1])

    pyp.sch.transform('simulate_mixture', mixture_axis, {'mem':1},
        demix.simulations.pipeline.simulate_mixture,
        None,
        mgd.TempOutputFile('mixture', *mixture_axis),
        mgd.TempOutputFile('mixture_plot.pdf', *mixture_axis),
        mgd.TempInputFile('genomes', *genome_axis),
        mgd.TempInputObj('mixture_params', *genome_axis),
    )

    pyp.sch.transform('simulate_tumour_data', mixture_axis, {'mem':24},
        demix.simulations.pipeline.simulate_tumour_data,
        None,
        mgd.TempOutputFile('tumour', *mixture_axis),
        mgd.TempInputFile('mixture', *mixture_axis),
        mgd.TempInputFile('germline_alleles', *germline_axis),
        mgd.TempFile('tumour_tmp', *mixture_axis),
        mgd.TempInputObj('mixture_params', *genome_axis),
    )

    haps_axis = genome_axis + ('bychromosome',)

    pyp.sch.setobj(mgd.OutputChunks('chromosomes', *haps_axis), sim_params['chromosomes'], haps_axis[:-1])

    pyp.sch.transform('infer_haps', haps_axis, {'mem':16},
        demix.analysis.haplotype.infer_haps,
        None,
        mgd.TempOutputFile('haps.tsv', *haps_axis),
        mgd.TempInputFile('normal', *genome_axis),
        mgd.InputInstance('bychromosome'),
        mgd.TempFile('haplotyping', *haps_axis),
        config,
    )

    pyp.sch.transform('merge_haps', genome_axis, {'mem':1},
        demix.utils.merge_tables,
        None,
        mgd.TempOutputFile('haps.tsv', *genome_axis),
        mgd.TempInputFile('haps.tsv', *haps_axis),
    )

    tool_axis = mixture_axis + ('bytool',)

    pyp.sch.transform('create_tool_analyses', mixture_axis, {'local':True},
        run_comparison.create_tool_analysis,
        mgd.TempOutputObj('tool_analysis', *tool_axis),
        args['install_dir'],
        mgd.TempFile('tool_tmp', *tool_axis),
    )

    init_axis = tool_axis + ('bytool',)

    pyp.sch.transform('tool_prepare', tool_axis, {'mem':8},
        run_comparison.tool_prepare,
        mgd.TempOutputObj('init_idx', *init_axis),
        mgd.TempInputObj('tool_analysis', *tool_axis),
        mgd.TempInputFile('normal', *genome_axis),
        mgd.TempInputFile('tumour', *mixture_axis),
        mgd.TempInputFile('segment.tsv', *genome_axis),
        mgd.TempInputFile('perfect_segment.tsv', *genome_axis),
        mgd.TempInputFile('breakpoint.tsv', *genome_axis),
        mgd.TempInputFile('haps.tsv', *genome_axis),
    )

    pyp.sch.transform('tool_run', init_axis, {'mem':8},
        run_comparison.tool_run,
        mgd.TempOutputObj('run_result', *init_axis),
        mgd.TempInputObj('tool_analysis', *tool_axis),
        mgd.TempInputObj('init_idx', *init_axis),
    )

    pyp.sch.transform('tool_report', tool_axis, {'mem':4},
        run_comparison.tool_report,
        None,
        mgd.TempInputObj('tool_analysis', *tool_axis),
        mgd.TempInputObj('run_result', *init_axis),
        mgd.TempOutputFile('cn.tsv', *tool_axis),
        mgd.TempOutputFile('mix.tsv', *tool_axis),
    )

    pyp.sch.transform('tabulate_results', tool_axis, {'mem':1},
        run_comparison.tabulate_results,
        None,
        mgd.OutputFile(args['results_table']),
        mgd.TempInputFile('cn.tsv', *tool_axis),
        mgd.TempInputFile('mix.tsv', *tool_axis),
    )

    pyp.run()

else:

    def read_sim_defs(params_filename, config):

        params = dict()
        execfile(params_filename, {}, params)

        params['chromosome_lengths'] = dict()
        for seq_id, sequence in demix.utils.read_sequences(config['genome_fasta']):
            if seq_id not in params['chromosomes']:
                continue
            params['chromosome_lengths'][seq_id] = len(sequence)

        return params


    def simulate_genomes(genomes_filename, segment_filename, perfect_segment_filename, breakpoint_filename, params):

        demix.simulations.pipeline.simulate_genomes(genomes_filename, params)
        demix.simulations.pipeline.write_segments(segment_filename, genomes_filename)
        demix.simulations.pipeline.write_perfect_segments(perfect_segment_filename, genomes_filename)
        demix.simulations.pipeline.write_breakpoints(breakpoint_filename, genomes_filename)


    def simulate_mixture(mixture_filename, plot_filename, genome_filename, params):

        demix.simulations.pipeline.simulate_mixture(mixture_filename, genome_filename, params)
        demix.simulations.pipeline.plot_mixture(plot_filename, mixture_filename)

    def set_chromosomes(sim_defs):

        return sim_defs['chromosomes']


    def create_tool_analysis(install_directory, analysis_directory):

        tool_analyses = dict()
        for tool_name, Tool in demix.wrappers.catalog.iteritems():
            tool = Tool(os.path.join(install_directory, tool_name))
            tool_analyses[tool_name] = tool.create_analysis(analysis_directory)

        return tool_analyses


    def tool_prepare(tool_analysis, normal_filename, tumour_filename, segment_filename, perfect_segment_filename, breakpoint_filename, haps_filename):

        num_inits = tool_analysis.prepare(
            normal_filename,
            tumour_filename,
            segment_filename=segment_filename,
            perfect_segment_filename=perfect_segment_filename,
            breakpoint_filename=breakpoint_filename,
            haplotype_filename=haps_filename
        )

        return dict(zip(xrange(num_inits), xrange(num_inits)))


    def tool_run(tool_analysis, init_idx):

        tool_analysis.run(init_idx)

        return True


    def tool_report(tool_analysis, run_results, cn_filename, mix_filename):

        tool_analysis.report(cn_filename, mix_filename)


    def tabulate_results(table_filename, cn_filenames, mix_filenames):

        with open(table_filename, 'w') as f:
            pass


