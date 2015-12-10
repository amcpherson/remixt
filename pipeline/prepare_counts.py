import sys
import logging
import os
import itertools
import argparse
import gzip
import collections
import numpy as np
import pandas as pd
import scipy.stats

import pypeliner
import pypeliner.workflow
import pypeliner.managed as mgd

import remixt
import remixt.seqdataio
import remixt.segalg
import remixt.utils
import remixt.analysis.haplotype
import remixt.analysis.segment
import remixt.analysis.gcbias
import remixt.analysis.stats


remixt_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
bin_directory = os.path.join(remixt_directory, 'bin')
default_config_filename = os.path.join(remixt_directory, 'defaultconfig.py')


if __name__ == '__main__':

    import prepare_counts
    
    argparser = argparse.ArgumentParser()

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('segment_file',
        help='Input segments file')

    argparser.add_argument('normal_file',
        help='Input normal sequence data filename')

    argparser.add_argument('--tumour_files', nargs='+', required=True,
        help='Input tumour sequence data filenames')

    argparser.add_argument('--count_files', nargs='+', required=True,
        help='Output count TSV filenames')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    if len(args['tumour_files']) != len(args['count_files']):
        raise Exception('--count_files must correspond one to one with --tumour_files')

    config = {'ref_data_directory':args['ref_data_dir']}
    execfile(default_config_filename, {}, config)

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([remixt, prepare_counts], config)

    workflow = pypeliner.workflow.Workflow()

    tumour_fnames = dict(enumerate(args['tumour_files']))
    count_fnames = dict(enumerate(args['count_files']))

    workflow.setobj(obj=mgd.OutputChunks('bytumour'), value=tumour_fnames.keys())

    workflow.transform(
        name='calc_fragment_stats',
        axes=('bytumour',),
        ctx={'mem':16},
        func=remixt.analysis.stats.calculate_fragment_stats,
        ret=mgd.TempOutputObj('fragstats', 'bytumour'),
        args=(mgd.InputFile('tumour_file', 'bytumour', fnames=tumour_fnames),),
    )

    workflow.setobj(obj=mgd.OutputChunks('bychromosome'), value=config['chromosomes'])

    workflow.transform(
        name='infer_haps',
        axes=('bychromosome',),
        ctx={'mem':16},
        func=remixt.analysis.haplotype.infer_haps,
        args=(
            mgd.TempOutputFile('haps.tsv', 'bychromosome'),
            mgd.InputFile(args['normal_file']),
            mgd.InputInstance('bychromosome'),
            mgd.TempFile('haplotyping', 'bychromosome'),
            config,
        )
    )

    workflow.transform(
        name='merge_haps',
        ctx={'mem':16},
        func=remixt.utils.merge_tables,
        args=(
            mgd.TempOutputFile('haps.tsv'),
            mgd.TempInputFile('haps.tsv', 'bychromosome'),
        )
    )

    workflow.transform(
        name='create_readcounts',
        axes=('bytumour',),
        ctx={'mem':16},
        func=prepare_counts.create_counts,
        args=(
            mgd.TempOutputFile('segment_counts.tsv', 'bytumour'),
            mgd.TempOutputFile('allele_counts.tsv', 'bytumour'),
            mgd.InputFile(args['segment_file']),
            mgd.InputFile('tumour_file', 'bytumour', fnames=tumour_fnames),
            mgd.TempInputFile('haps.tsv'),
        )
    )

    workflow.transform(
        name='phase_segments',
        ctx={'mem':16},
        func=prepare_counts.phase_segments,
        args=(
            mgd.TempInputFile('allele_counts.tsv', 'bytumour'),
            mgd.TempOutputFile('phased_allele_counts.tsv', 'bytumour', axes_origin=[]),
        )
    )

    workflow.transform(
        name='sample_gc',
        axes=('bytumour',),
        ctx={'mem':16},
        func=remixt.analysis.gcbias.sample_gc,
        args=(
            mgd.TempOutputFile('gcsamples.tsv', 'bytumour'),
            mgd.InputFile('tumour_file', 'bytumour', fnames=tumour_fnames),
            mgd.TempInputObj('fragstats', 'bytumour').prop('fragment_mean'),
            config,
        )
    )

    workflow.transform(
        name='gc_lowess',
        axes=('bytumour',),
        ctx={'mem':16},
        func=remixt.analysis.gcbias.gc_lowess,
        args=(
            mgd.TempInputFile('gcsamples.tsv', 'bytumour'),
            mgd.TempOutputFile('gcloess.tsv', 'bytumour'),
            mgd.TempOutputFile('gctable.tsv', 'bytumour'),
        )
    )

    workflow.commandline(
        name='gc_segment',
        axes=('bytumour',),
        ctx={'mem':16},
        args=(
            os.path.join(bin_directory, 'estimategc'),
            '-m', config['mappability_filename'],
            '-g', config['genome_fasta'],
            '-c', mgd.TempInputFile('segment_counts.tsv', 'bytumour'),
            '-i',
            '-o', '4',
            '-u', mgd.TempInputObj('fragstats', 'bytumour').prop('fragment_mean'),
            '-s', mgd.TempInputObj('fragstats', 'bytumour').prop('fragment_stddev'),
            '-a', config['mappability_length'],
            '-l', mgd.TempInputFile('gcloess.tsv', 'bytumour'),
            '>', mgd.TempOutputFile('segment_counts_lengths.tsv', 'bytumour'),
        )
    )

    workflow.transform(
        name='prepare_counts',
        axes=('bytumour',),
        ctx={'mem':16},
        func=prepare_counts.prepare_counts,
        args=(
            mgd.TempInputFile('segment_counts_lengths.tsv', 'bytumour'),
            mgd.TempInputFile('phased_allele_counts.tsv', 'bytumour'),
            mgd.OutputFile('count_file', 'bytumour', fnames=count_fnames),
        )
    )

    pyp.run(workflow)


def create_counts(segment_counts_filename, allele_counts_filename, segment_filename, seqdata_filename, haps_filename):

    segments = pd.read_csv(segment_filename, sep='\t', converters={'chromosome':str})

    segment_counts = remixt.analysis.segment.create_segment_counts(
        segments,
        seqdata_filename,
    )

    segment_counts.to_csv(segment_counts_filename, sep='\t', index=False)

    allele_counts = remixt.analysis.haplotype.create_allele_counts(
        segments,
        seqdata_filename,
        haps_filename,
    )

    allele_counts.to_csv(allele_counts_filename, sep='\t', index=False)


def phase_segments(allele_counts_filenames, phased_allele_counts_filename_callback):

    tumour_ids = allele_counts_filenames.keys()

    allele_count_tables = list()
    for allele_counts_filename in allele_counts_filenames.itervalues():
        allele_count_tables.append(pd.read_csv(allele_counts_filename, sep='\t', converters={'chromosome':str}))

    phased_allele_counts_tables = remixt.analysis.haplotype.phase_segments(*allele_count_tables)

    for tumour_id, phased_allele_counts in zip(tumour_ids, phased_allele_counts_tables):
        phased_allele_counts_filename = phased_allele_counts_filename_callback(tumour_id)
        phased_allele_counts.to_csv(phased_allele_counts_filename, sep='\t', index=False)


def prepare_counts(segments_filename, alleles_filename, count_filename):

    segment_data = pd.read_csv(segments_filename, sep='\t', converters={'chromosome':str})
    allele_data = pd.read_csv(alleles_filename, sep='\t', converters={'chromosome':str})

    segment_allele_counts = remixt.analysis.segment.create_segment_allele_counts(segment_data, allele_data)

    segment_allele_counts.to_csv(count_filename, sep='\t', index=False)



