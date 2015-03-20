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

    import run_inference_read_sim

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('sim_defs', 
        help='Simulation Definition Filename')

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

    pyp = pypeliner.app.Pypeline([demix, run_inference_read_sim], config)

    pyp.sch.transform('read_sim_defs', (), {'mem':1,'local':True},
        run_inference_read_sim.read_sim_defs,
        mgd.TempOutputObj('sim_defs'),
        mgd.InputFile(args['sim_defs']),
        config)

    pyp.sch.transform('simulate_genomes', (), {'mem':4},
        demix.simulations.pipeline.simulate_genomes,
        None,
        mgd.TempOutputFile('genomes'),
        mgd.TempInputObj('sim_defs'))

    pyp.sch.transform('simulate_mixture', (), {'mem':1},
        demix.simulations.pipeline.simulate_mixture,
        None,
        mgd.TempOutputFile('mixture'),
        mgd.TempInputFile('genomes'),
        mgd.TempInputObj('sim_defs'))

    pyp.sch.transform('plot_mixture', (), {'mem':4},
        demix.simulations.pipeline.plot_mixture,
        None,
        mgd.TempOutputFile('mixture_plot.pdf'),
        mgd.TempInputFile('mixture'))

    pyp.sch.transform('simulate_germline_alleles', (), {'mem':8},
        demix.simulations.pipeline.simulate_germline_alleles,
        None,
        mgd.TempOutputFile('germline_alleles'),
        mgd.TempInputObj('sim_defs'),
        config)

    pyp.sch.transform('simulate_normal_data', (), {'mem':24},
        demix.simulations.pipeline.simulate_normal_data,
        None,
        mgd.TempOutputFile('normal'),
        mgd.TempInputFile('mixture'),
        mgd.TempInputFile('germline_alleles'),
        mgd.TempFile('normal_tmp'),
        mgd.TempInputObj('sim_defs'))

    pyp.sch.transform('simulate_tumour_data', (), {'mem':24},
        demix.simulations.pipeline.simulate_tumour_data,
        None,
        mgd.TempOutputFile('tumour'),
        mgd.TempInputFile('mixture'),
        mgd.TempInputFile('germline_alleles'),
        mgd.TempFile('tumour_tmp'),
        mgd.TempInputObj('sim_defs'))

    pyp.sch.transform('write_segments', (), {'mem':1},
        demix.simulations.pipeline.write_segments,
        None,
        mgd.TempOutputFile('segment.tsv'),
        mgd.TempInputFile('genomes'))

    pyp.sch.transform('write_perfect_segments', (), {'mem':1},
        demix.simulations.pipeline.write_perfect_segments,
        None,
        mgd.TempOutputFile('perfect_segment.tsv'),
        mgd.TempInputFile('genomes'))

    pyp.sch.transform('write_breakpoints', (), {'mem':1},
        demix.simulations.pipeline.write_breakpoints,
        None,
        mgd.TempOutputFile('breakpoint.tsv'),
        mgd.TempInputFile('genomes'))

    pyp.sch.transform('set_chromosomes', (), {'local':True},
        run_inference_read_sim.set_chromosomes,
        mgd.OutputChunks('bychromosome'),
        mgd.TempInputObj('sim_defs'))

    pyp.sch.transform('infer_haps', ('bychromosome',), {'mem':16},
        demix.analysis.haplotype.infer_haps,
        None,
        mgd.TempOutputFile('haps.tsv', 'bychromosome'),
        mgd.TempInputFile('normal'),
        mgd.InputInstance('bychromosome'),
        mgd.TempFile('haplotyping', 'bychromosome'),
        config)

    pyp.sch.transform('merge_haps', (), {'mem':16},
        demix.utils.merge_tables,
        None,
        mgd.TempOutputFile('haps.tsv'),
        mgd.TempInputFile('haps.tsv', 'bychromosome'))

    pyp.sch.transform('create_tools', (), {'local':True},
        run_inference_read_sim.create_tools,
        mgd.TempOutputObj('tool', 'bytool'),
        args['install_dir'])

    pyp.sch.transform('create_analysis', ('bytool',), {'local':True},
        run_inference_read_sim.create_analysis,
        mgd.TempOutputObj('tool_analysis', 'bytool'),
        mgd.TempInputObj('tool', 'bytool'),
        mgd.TempFile('tool_tmp', 'bytool'))

    pyp.sch.transform('tool_prepare', ('bytool',), {'mem':8},
        run_inference_read_sim.tool_prepare,
        mgd.TempOutputObj('init_idx', 'bytool', 'byinit'),
        mgd.TempInputObj('tool_analysis', 'bytool'),
        mgd.TempInputFile('normal'),
        mgd.TempInputFile('tumour'),
        mgd.TempInputFile('segment.tsv'),
        mgd.TempInputFile('perfect_segment.tsv'),
        mgd.TempInputFile('breakpoint.tsv'),
        mgd.TempInputFile('haps.tsv'))

    pyp.sch.transform('tool_run', ('bytool', 'byinit'), {'mem':8},
        run_inference_read_sim.tool_run,
        mgd.TempOutputObj('run_result', 'bytool', 'byinit'),
        mgd.TempInputObj('tool_analysis', 'bytool'),
        mgd.TempInputObj('init_idx', 'bytool', 'byinit'))

    pyp.sch.transform('tool_report', ('bytool',), {'mem':4},
        run_inference_read_sim.tool_report,
        None,
        mgd.TempInputObj('tool_analysis', 'bytool'),
        mgd.TempInputObj('run_result', 'bytool', 'byinit'),
        mgd.TempOutputFile('cn.tsv', 'bytool'),
        mgd.TempOutputFile('mix.tsv', 'bytool'))

    pyp.sch.transform('tabulate_results', (), {'mem':1},
        run_inference_read_sim.tabulate_results,
        None,
        mgd.OutputFile(args['results_table']),
        mgd.TempInputFile('cn.tsv', 'bytool'),
        mgd.TempInputFile('mix.tsv', 'bytool'))

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


    def set_chromosomes(sim_defs):

        return sim_defs['chromosomes']


    def create_tools(install_directory):

        tools = dict()
        for tool_name, Tool in demix.wrappers.catalog.iteritems():
            tools[tool_name] = Tool(os.path.join(install_directory, tool_name))

        return tools


    def create_analysis(tool, analysis_directory):

        return tool.create_analysis(analysis_directory)


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



