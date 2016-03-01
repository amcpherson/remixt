import argparse

import pypeliner

import remixt.workflow

if __name__ == '__main__':
    
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

    config = {}
    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([remixt], config)

    workflow = remixt.workflow.create_extract_seqdata_workflow(
        args['bam_file'],
        args['seqdata_file'],
        config,
        args['ref_data_dir'],
    )

    pyp.run(workflow)


