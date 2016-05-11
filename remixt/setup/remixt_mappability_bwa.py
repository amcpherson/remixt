import argparse

import pypeliner

import remixt.mappability.bwa.workflow


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    pypeliner.app.add_arguments(argparser)

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    ref_data_dir = args['ref_data_dir']

    config = {}
    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline(config=config)

    workflow = remixt.mappability.bwa.workflow.create_bwa_mappability_workflow(config, ref_data_dir)

    pyp.run(workflow)
    

