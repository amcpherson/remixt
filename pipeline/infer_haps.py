import argparse

import pypeliner
import pypeliner.workflow

import remixt.workflow


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('normal_file',
        help='Input normal sequence data filename')

    argparser.add_argument('haplotypes_file',
        help='Output haplotypes file')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    config = {'ref_data_dir': args['ref_data_dir']}

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([remixt], config)

    workflow = remixt.workflow.create_infer_haps_workflow(
        args['normal_file'],
        args['haplotypes_file'],
        config,
    )

    pyp.run(workflow)










