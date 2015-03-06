import sys
import logging
import os
import itertools
import argparse
import gzip
import tarfile

import pypeliner
import pypeliner.managed as mgd


demix_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))

bin_directory = os.path.join(demix_directory, 'bin')

sys.path.append(demix_directory)

import demix.seqdataio as seqdataio


if __name__ == '__main__':

    import extract_reads
    
    argparser = argparse.ArgumentParser()

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('bam_file',
                           help='Input bam filename')

    argparser.add_argument('snp_positions',
                           help='Input SNP positions filename')

    argparser.add_argument('seqdata_file',
                           help='Output sequence data filenames')

    argparser.add_argument('--chromosomes', nargs='+', required=True,
                           help='Chromosomes to extract')

    args = vars(argparser.parse_args())

    pyp = pypeliner.app.Pypeline([extract_reads], args)

    ctx = {'mem':4}

    pyp.sch.setobj(mgd.OutputChunks('chromosome'), args['chromosomes'])

    pyp.sch.commandline('read_concordant', ('chromosome',), ctx_general,
        os.path.join(bin_directory, 'bamconcordantreads'),
        '--clipmax', '8',
        '--flen', '1000',
        '--chr', mgd.InputInstance('chromosome'),
        '-b', mgd.InputFile(args['bam_file']),
        '-s', mgd.InputFile(args['snp_positions']),
        '-r', mgd.TempOutputFile('reads', 'chromosome'),
        '-a', mgd.TempOutputFile('alleles', 'chromosome'))

    pyp.sch.transform('create_seqdata', (), ctx,
        seqdataio.create_seqdata,
        mgd.OutputFile(args['seqdata_file']),
        mgd.TempInputFile('reads', 'chromosome'),
        mgd.TempInputFile('alleles', 'chromosome'))

