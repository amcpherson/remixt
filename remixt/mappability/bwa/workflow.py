import pypeliner
import pypeliner.workflow
import pypeliner.managed as mgd

import remixt
import remixt.config
import remixt.utils
import remixt.mappability.tasks


def create_bwa_mappability_workflow(config, ref_data_dir, **kwargs):
    workflow = pypeliner.workflow.Workflow(default_ctx={'mem': 8})

    mappability_length = remixt.config.get_param(config, 'mappability_length')
    genome_fasta = remixt.config.get_filename(config, ref_data_dir, 'genome_fasta')
    mappability_filename = remixt.config.get_filename(config, ref_data_dir, 'mappability')

    workflow.transform(
        name='create_kmers',
        func=remixt.mappability.tasks.create_kmers,
        args=(
            mgd.InputFile(genome_fasta),
            mappability_length,
            mgd.TempOutputFile('kmers'),
        ),
    )

    workflow.transform(
        name='split_kmers',
        func=remixt.mappability.tasks.split_file_byline,
        args=(
            mgd.TempInputFile('kmers'),
            4000000,
            mgd.TempOutputFile('kmers', 'bykmer'),
        ),
    )

    workflow.commandline(
        name='bwa_aln_kmers',
        axes=('bykmer',),
        args=(
            'bwa',
            'aln',
            mgd.InputFile(genome_fasta),
            mgd.TempInputFile('kmers', 'bykmer'),
            '>',
            mgd.TempOutputFile('sai', 'bykmer'),
        ),
    )

    workflow.commandline(
        name='bwa_samse_kmers',
        axes=('bykmer',),
        args=(
            'bwa',
            'samse',
            mgd.InputFile(genome_fasta),
            mgd.TempInputFile('sai', 'bykmer'),
            mgd.TempInputFile('kmers', 'bykmer'),
            '>',
            mgd.TempOutputFile('alignments', 'bykmer'),
        ),
    )

    workflow.transform(
        name='create_bedgraph',
        axes=('bykmer',),
        func=remixt.mappability.tasks.create_bedgraph,
        args=(
            mgd.TempInputFile('alignments', 'bykmer'),
            mgd.TempOutputFile('bedgraph', 'bykmer'),
        ),
    )

    workflow.transform(
        name='merge_bedgraph',
        func=remixt.mappability.tasks.merge_files_by_line,
        args=(
            mgd.TempInputFile('bedgraph', 'bykmer'),
            mgd.OutputFile(mappability_filename),
        ),
    )

    return workflow
    
