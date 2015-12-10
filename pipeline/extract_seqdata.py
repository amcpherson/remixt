import sys
import logging
import os
import itertools
import argparse
import gzip
import tarfile

import pypeliner
import pypeliner.workflow
import pypeliner.managed as mgd

import remixt.seqdataio as seqdataio


remixt_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
bin_directory = os.path.join(remixt_directory, 'bin')
default_config_filename = os.path.join(remixt_directory, 'defaultconfig.py')


if __name__ == '__main__':

    import extract_seqdata
    
    argparser = argparse.ArgumentParser()

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('bam_file',
        help='Input bam filename')

    argparser.add_argument('seqdata_file',
        help='Output sequence data filenames')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    config = {'ref_data_directory':args['ref_data_dir']}
    execfile(default_config_filename, {}, config)

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([extract_seqdata], config)

    workflow = pypeliner.workflow.Workflow()

    workflow.setobj(obj=mgd.OutputChunks('chromosome'), value=config['chromosomes'])

    workflow.commandline(
        name='read_concordant',
        axes=('chromosome',),
        ctx={'mem':16},
        args=(
            os.path.join(bin_directory, 'bamconcordantreads'),
            '--clipmax', '8',
            '--flen', '1000',
            '--chr', mgd.InputInstance('chromosome'),
            '-b', mgd.InputFile(args['bam_file']),
            '-s', mgd.InputFile(config['snp_positions']),
            '-r', mgd.TempOutputFile('reads', 'chromosome'),
            '-a', mgd.TempOutputFile('alleles', 'chromosome'),
        )
    )

    workflow.transform(
        name='create_seqdata',
        ctx={'mem':4},
        func=seqdataio.create_seqdata,
        args=(
            mgd.OutputFile(args['seqdata_file']),
            mgd.TempInputFile('reads', 'chromosome'),
            mgd.TempInputFile('alleles', 'chromosome'),
        )
    )

    pyp.run(workflow)


