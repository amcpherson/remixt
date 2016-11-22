import argparse
import yaml

import pypeliner

import remixt
import remixt.workflow


def run(**args):
    if len(args['tumour_bam_files']) != len(args['tumour_sample_ids']):
        raise Exception('--tumour_bam_files must correspond one to one with --tumour_sample_ids')

    if len(args['results_files']) != len(args['tumour_sample_ids']):
        raise Exception('--results_files must correspond one to one with --tumour_sample_ids')

    config = yaml.load(open(args['config']))

    tumour_bam_filenames = dict(zip(args['tumour_sample_ids'], args['tumour_bam_files']))
    results_filenames = dict(zip(args['tumour_sample_ids'], args['results_files']))

    pypeliner_config = config.copy()
    pypeliner_config.update(args)
    pyp = pypeliner.app.Pypeline([remixt], pypeliner_config)

    workflow = remixt.workflow.create_remixt_bam_workflow(
        args['segment_file'],
        args['breakpoint_file'],
        tumour_bam_filenames,
        args['normal_bam_file'],
        args['normal_sample_id'],
        results_filenames,
        args['raw_data_dir'],
        config,
        args['ref_data_dir'],
    )

    pyp.run(workflow)


def add_arguments(argparser):
    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('raw_data_dir',
        help='Output raw data directory')

    argparser.add_argument('segment_file',
        help='Input segments file')

    argparser.add_argument('breakpoint_file',
        help='Input breakpoints filename')

    argparser.add_argument('normal_bam_file',
        help='Input normal bam filenames')

    argparser.add_argument('--tumour_sample_ids', nargs='+', required=True,
        help='Identifiers for tumour samples')

    argparser.add_argument('--tumour_bam_files', nargs='+', required=True,
        help='Input tumour bam filenames')

    argparser.add_argument('--results_files', nargs='+', required=True,
        help='Output results filenames')

    argparser.add_argument('--normal_sample_id', default='normal',
        help='Normal sample id')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    argparser.set_defaults(func=run)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    add_arguments(argparser)

    args = vars(argparser.parse_args())
    func = args.pop('func')
    func(**args)
