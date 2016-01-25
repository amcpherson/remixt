import argparse

import pypeliner

import remixt
import remixt.workflow


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('segment_file',
        help='Input segments file')

    argparser.add_argument('haplotypes_file',
        help='Input haplotypes file')

    argparser.add_argument('--tumour_files', nargs='+', required=True,
        help='Input tumour sequence data filenames')

    argparser.add_argument('--count_files', nargs='+', required=True,
        help='Output count TSV filenames')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    if len(args['tumour_files']) != len(args['count_files']):
        raise Exception('--count_files must correspond one to one with --tumour_files')

    config = {'ref_data_directory': args['ref_data_dir']}

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([remixt], config)

    tumour_fnames = dict(enumerate(args['tumour_files']))
    count_fnames = dict(enumerate(args['count_files']))

    workflow = remixt.workflow.create_prepare_counts_workflow(
        args['segment_file'],
        args['haplotypes_file'],
        tumour_fnames,
        count_fnames,
        config,
    )

    pyp.run(workflow)

