import os
import argparse
import pandas as pd

import pypeliner
import pypeliner.managed as mgd

import remixt.simulations.pipeline
import remixt.analysis.haplotype
import remixt.wrappers
import remixt.utils


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

    config = {}
    if args['config'] is not None:
        execfile(args['config'], {}, config)

    pyp = pypeliner.app.Pypeline(modules=[remixt, run_comparison], config=args)

    params = dict()
    execfile(args['sim_params'], {}, params)

    chromosomes = params['chromosomes']

    chromosome_lengths = dict()
    for chromosome, length in remixt.utils.read_chromosome_lengths(config['genome_fai']).iteritems():
        if chromosome not in chromosomes:
            continue
        chromosome_lengths[chromosome] = length

    defaults = params['defaults']

    defaults['chromosomes'] = chromosomes
    defaults['chromosome_lengths'] = chromosome_lengths

    def merge_params(params, defaults):
        merged = dict()
        for idx, values in enumerate(zip(*params.values())):
            merged[idx] = dict(zip(params.keys(), values))
            for key, value in defaults.iteritems():
                merged[idx][key] = value
        return merged

    germline_params = params['germline_params']
    germline_params = merge_params(germline_params, defaults)

    genome_params = params['genome_params']
    genome_params = merge_params(genome_params, defaults)

    mixture_params = params['mixture_params']
    mixture_params = merge_params(mixture_params, defaults)

    workflow = pypeliner.workflow.Workflow()

    # For each of n patients:
    #     * simulate germline alleles

    germline_axis = ('bygermline',)

    workflow.setobj(mgd.TempOutputObj('germline_params', *germline_axis), germline_params)
    workflow.setobj(mgd.TempOutputObj('chromosomes'), chromosomes)

    workflow.transform(
        name='simulate_germline_alleles',
        axes=germline_axis,
        ctx={'mem':8},
        func=remixt.simulations.pipeline.simulate_germline_alleles,
        args=(
            mgd.TempOutputFile('germline_alleles', *germline_axis),
            mgd.TempInputObj('germline_params', *germline_axis),
            mgd.TempInputObj('chromosomes'),
            config,
        ),
    )

    # For each of n patients:
    #     For each of m genomes:
    #         * clonal divergence
    #         * normal dataset

    genome_axis = germline_axis + ('bygenome',)

    workflow.setobj(mgd.TempOutputObj('genome_params', *genome_axis), genome_params, genome_axis[:-1])

    workflow.transform(
        name='simulate_genomes',
        axes=genome_axis,
        ctx={'mem':4},
        func=run_comparison.simulate_genomes,
        args=(
            mgd.TempOutputFile('genomes', *genome_axis),
            mgd.TempOutputFile('segment.tsv', *genome_axis),
            mgd.TempOutputFile('perfect_segment.tsv', *genome_axis),
            mgd.TempOutputFile('breakpoint.tsv', *genome_axis),
            mgd.TempInputObj('genome_params', *genome_axis),
        ),
    )

    workflow.transform(
        name='simulate_normal_data',
        axes=genome_axis,
        ctx={'mem':16},
        func=remixt.simulations.pipeline.simulate_normal_data,
        args=(
            mgd.TempOutputFile('normal', *genome_axis),
            mgd.TempInputFile('genomes', *genome_axis),
            mgd.TempInputFile('germline_alleles', *germline_axis),
            mgd.TempFile('normal_tmp', *genome_axis),
            mgd.TempInputObj('genome_params', *genome_axis),
        ),
    )

    # For each of n patients:
    #     For each of m genomes:
    #         For each of l mixtures:
    #             * tumour dataset

    mixture_axis = genome_axis + ('bymixture',)

    workflow.setobj(mgd.TempOutputObj('mixture_params', *mixture_axis), mixture_params, mixture_axis[:-1])

    workflow.transform(
        name='simulate_mixture',
        axes=mixture_axis,
        ctx={'mem':2},
        func=run_comparison.simulate_mixture,
        args=(
            mgd.TempOutputFile('mixture', *mixture_axis),
            mgd.TempOutputFile('mixture_plot.pdf', *mixture_axis),
            mgd.TempInputFile('genomes', *genome_axis),
            mgd.TempInputObj('mixture_params', *mixture_axis),
        ),
    )

    workflow.transform(
        name='simulate_tumour_data',
        axes=mixture_axis,
        ctx={'mem':16},
        func=remixt.simulations.pipeline.simulate_tumour_data,
        args=(
            mgd.TempOutputFile('tumour', *mixture_axis),
            mgd.TempInputFile('mixture', *mixture_axis),
            mgd.TempInputFile('germline_alleles', *germline_axis),
            mgd.TempFile('tumour_tmp', *mixture_axis),
            mgd.TempInputObj('mixture_params', *mixture_axis),
        ),
    )

    haps_axis = genome_axis + ('bychromosome',)

    workflow.setobj(mgd.OutputChunks(*haps_axis), chromosomes, haps_axis[:-1])

    workflow.transform(
        name='infer_haps',
        axes=haps_axis,
        ctx={'mem':16},
        func=remixt.analysis.haplotype.infer_haps,
        args=(
            mgd.TempOutputFile('haps.tsv', *haps_axis),
            mgd.TempInputFile('normal', *genome_axis),
            mgd.InputInstance('bychromosome'),
            mgd.TempFile('haplotyping', *haps_axis),
            config,
        ),
    )

    workflow.transform(
        name='merge_haps',
        axes=genome_axis,
        ctx={'mem':8},
        func=remixt.utils.merge_tables,
        args=(
            mgd.TempOutputFile('haps.tsv', *genome_axis),
            mgd.TempInputFile('haps.tsv', *haps_axis),
        ),
    )

    tool_axis = mixture_axis + ('bytool',)

    workflow.transform(
        name='create_tools',
        axes=mixture_axis,
        ctx={'local':True},
        func=run_comparison.create_tools,
        ret=mgd.TempOutputObj('tool', *tool_axis),
        args=(args['install_dir'],),
    )

    workflow.transform(
        name='create_analysis',
        axes=tool_axis,
        ctx={'local':True},
        func=run_comparison.create_analysis,
        ret=mgd.TempOutputObj('tool_analysis', *tool_axis),
        args=(
            mgd.TempInputObj('tool', *tool_axis),
            mgd.TempFile('tool_tmp', *tool_axis),
        ),
    )

    init_axis = tool_axis + ('bytool',)

    workflow.transform(
        name='tool_prepare',
        axes=tool_axis,
        ctx={'mem':16},
        func=run_comparison.tool_prepare,
        ret=mgd.TempOutputObj('init_idx', *init_axis),
        args=(
            mgd.TempInputObj('tool_analysis', *tool_axis),
            mgd.TempInputFile('normal', *genome_axis),
            mgd.TempInputFile('tumour', *mixture_axis),
            mgd.TempInputFile('segment.tsv', *genome_axis),
            mgd.TempInputFile('perfect_segment.tsv', *genome_axis),
            mgd.TempInputFile('breakpoint.tsv', *genome_axis),
            mgd.TempInputFile('haps.tsv', *genome_axis),
        ),
    )

    workflow.transform(
        name='tool_run',
        axes=init_axis,
        ctx={'mem':16},
        func=run_comparison.tool_run,
        ret=mgd.TempOutputObj('run_result', *init_axis),
        args=(
            mgd.TempInputObj('tool_analysis', *tool_axis),
            mgd.TempInputObj('init_idx', *init_axis),
        ),
    )

    workflow.transform(
        name='tool_report',
        axes=tool_axis,
        ctx={'mem':16},
        func=run_comparison.tool_report,
        args=(
            mgd.TempInputObj('tool_analysis', *tool_axis),
            mgd.TempInputObj('run_result', *init_axis),
            mgd.TempOutputFile('cn.tsv', *tool_axis),
            mgd.TempOutputFile('mix.tsv', *tool_axis),
        ),
    )

    workflow.transform(
        name='evaluate_results',
        axes=tool_axis,
        ctx={'mem':1},
        func=run_comparison.evaluate_results,
        args=(
            mgd.TempOutputFile('results.tsv', *tool_axis),
            mgd.TempInputFile('mixture', *mixture_axis),
            mgd.TempInputFile('cn.tsv', *tool_axis),
            mgd.TempInputFile('mix.tsv', *tool_axis),
            mgd.InputInstance('bytool'),
        ),
        germline=mgd.TempInputObj('germline_params', *germline_axis),
        mixture=mgd.TempInputObj('mixture_params', *mixture_axis),
        genome=mgd.TempInputObj('genome_params', *genome_axis),
    )

    workflow.transform(
        name='merge_tool_results',
        axes=mixture_axis,
        ctx={'mem':1},
        func=remixt.utils.merge_tables,
        args=(
            mgd.TempOutputFile('results.tsv', *mixture_axis),
            mgd.TempInputFile('results.tsv', *tool_axis),
        ),
    )

    workflow.transform(
        name='merge_mixture_results',
        axes=genome_axis,
        ctx={'mem':1},
        func=remixt.utils.merge_tables,
        args=(
            mgd.TempOutputFile('results.tsv', *genome_axis),
            mgd.TempInputFile('results.tsv', *mixture_axis),
        ),
    )

    workflow.transform(
        name='merge_genome_results',
        axes=germline_axis,
        ctx={'mem':1},
        func=remixt.utils.merge_tables,
        args=(
            mgd.TempOutputFile('results.tsv', *germline_axis),
            mgd.TempInputFile('results.tsv', *genome_axis),
        ),
    )

    workflow.transform(
        name='merge_germline_results',
        axes=(),
        ctx={'mem':1},
        func=remixt.utils.merge_tables,
        args=(
            mgd.OutputFile(args['results_table']),
            mgd.TempInputFile('results.tsv', *germline_axis),
        ),
    )

    pyp.run(workflow)

else:

    def read_sim_defs(params_filename, config):

        params = dict()
        execfile(params_filename, {}, params)

        params['chromosome_lengths'] = dict()
        for seq_id, sequence in remixt.utils.read_sequences(config['genome_fasta']):
            if seq_id not in params['chromosomes']:
                continue
            params['chromosome_lengths'][seq_id] = len(sequence)

        return params


    def simulate_genomes(genomes_filename, segment_filename, perfect_segment_filename, breakpoint_filename, params):

        remixt.simulations.pipeline.simulate_genomes(genomes_filename, params)
        remixt.simulations.pipeline.write_segments(segment_filename, genomes_filename)
        remixt.simulations.pipeline.write_perfect_segments(perfect_segment_filename, genomes_filename)
        remixt.simulations.pipeline.write_breakpoints(breakpoint_filename, genomes_filename)


    def simulate_mixture(mixture_filename, plot_filename, genome_filename, params):

        remixt.simulations.pipeline.simulate_mixture(mixture_filename, genome_filename, params)
        remixt.simulations.pipeline.plot_mixture(plot_filename, mixture_filename)

    def set_chromosomes(sim_defs):

        return sim_defs['chromosomes']


    def create_tools(install_directory):

        tools = dict()
        for tool_name, Tool in remixt.wrappers.catalog.iteritems():
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


    def evaluate_results(results_filename, mixture_filename, cn_filename, mix_filename, tool_name, **params):

        results = remixt.simulations.pipeline.evaluate_results(
            mixture_filename, cn_filename, mix_filename)

        results['tool'] = tool_name

        for param_group, param_values in params.iteritems():
            for key, value in param_values.iteritems():
                results[param_group + '_' + key] = [value]

        results = pd.DataFrame(results, index=[0])

        results.to_csv(results_filename, sep='\t', index=False)


