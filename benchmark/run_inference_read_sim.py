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
        mgd.TempInputObj('sim_defs').extract(lambda a: a['chromosomes']),
        config)

    pyp.sch.transform('simulate_normal_data', (), {'mem':32},
        demix.simulations.pipeline.simulate_normal_data,
        None,
        mgd.TempOutputFile('normal'),
        mgd.TempInputFile('genomes'),
        mgd.TempInputFile('germline_alleles'),
        mgd.TempFile('normal_tmp'),
        mgd.TempInputObj('sim_defs'))

    pyp.sch.transform('simulate_tumour_data', (), {'mem':32},
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

    pyp.sch.transform('merge_haps', (), {'mem':4},
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

    pyp.sch.transform('tool_prepare', ('bytool',), {'mem':16},
        run_inference_read_sim.tool_prepare,
        mgd.TempOutputObj('init_idx', 'bytool', 'byinit'),
        mgd.TempInputObj('tool_analysis', 'bytool'),
        mgd.TempInputFile('normal'),
        mgd.TempInputFile('tumour'),
        mgd.TempInputFile('segment.tsv'),
        mgd.TempInputFile('perfect_segment.tsv'),
        mgd.TempInputFile('breakpoint.tsv'),
        mgd.TempInputFile('haps.tsv'))

    pyp.sch.transform('tool_run', ('bytool', 'byinit'), {'mem':16},
        run_inference_read_sim.tool_run,
        mgd.TempOutputObj('run_result', 'bytool', 'byinit'),
        mgd.TempInputObj('tool_analysis', 'bytool'),
        mgd.TempInputObj('init_idx', 'bytool', 'byinit'))

    pyp.sch.transform('tool_report', ('bytool',), {'mem':16},
        run_inference_read_sim.tool_report,
        None,
        mgd.TempInputObj('tool_analysis', 'bytool'),
        mgd.TempInputObj('run_result', 'bytool', 'byinit'),
        mgd.TempOutputFile('cn.tsv', 'bytool'),
        mgd.TempOutputFile('mix.tsv', 'bytool'))

    pyp.sch.transform('evaluate_results', ('bytool',), {'mem':1},
        run_inference_read_sim.evaluate_results,
        None,
        mgd.TempInputFile('results.tsv'),
        mgd.TempInputFile('mixture'),
        mgd.TempInputFile('cn.tsv', 'bytool'),
        mgd.TempInputFile('mix.tsv', 'bytool'),
        mgd.InputInstance('bytool'))

    pyp.sch.transform('merge_results', (), {'mem':1},
        demix.utils.merge_tables,
        None,
        mgd.OutputFile(args['results_table']),
        mgd.TempInputFile('results.tsv', 'bytool'))

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


    def evaluate_results(results_filename, mixture_filename, cn_filename, mix_filename, tool_name):

        with open(mixture_filename, 'r') as mixture_file:
            gm = pickle.load(mixture_file)

        cn_data = pd.read_csv(cn_filename, sep='\t', converters={'chromosome':str})

        sim_segments = pd.DataFrame({
            'chromosome':gm.segment_chromosome_id,
            'start':gm.segment_start,
            'end':gm.segment_end,
        })

        if 'major_1' in cn_data:

            cn_true = gm.cn[:,1:,:]

            cn_pred = np.array(
                [
                    [cn_data['major_1'], cn_data['minor_1']],
                    [cn_data['major_2'], cn_data['minor_2']],
                ]
            ).swapaxes(0, 2).swapaxes(1, 2)

        else:

            cn_true = np.zeros((gm.cn.shape[0], gm.cn.shape[1]-1, 1))

            cn_true[:,:,0] = gm.cn[:,1:,:].sum(axis=2)

            cn_pred = np.array(
                [
                    [cn_data['total_1']],
                    [cn_data['total_2']],
                ]
            ).swapaxes(0, 2).swapaxes(1, 2)
            
        cn_data_index = demix.simulations.pipeline.reindex_segments(sim_segments, cn_data)

        cn_true = cn_true[cn_data_index['idx_1'].values,:,:]
        cn_pred = cn_pred[cn_data_index['idx_2'].values,:,:]
        segment_lengths = (cn_data_index['end'] - cn_data_index['start']).values

        mix_true = gm.frac
        with open(mix_filename, 'r') as mix_file:
            mix_pred = np.array(mix_file.readline().split()).astype(float)

        proportion_cn_correct = demix.simulations.pipeline.compare_cn(
            mix_true[1:], mix_pred[1:], cn_true, cn_pred, segment_lengths)

        results = pd.DataFrame({
            'tool':tool_name,
            'proportion_cn_correct':[proportion_cn_correct],
        })

        for idx, f in enumerate(mix_true):
            results['mix_true_'+str(idx)] = f

        for idx, f in enumerate(mix_pred):
            results['mix_pred_'+str(idx)] = f

        results.to_csv(results_filename, sep='\t', index=False)


