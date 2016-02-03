import argparse

import pypeliner

import remixt
import remixt.workflow


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pypeliner.app.add_arguments(argparser)
    
    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('seqdata_file',
        help='Input sequence data filename')

    argparser.add_argument('segments',
        help='Input segments filename')

    argparser.add_argument('segment_lengths',
        help='Output segments with lengths filename')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    config = {'ref_data_dir': args['ref_data_dir']}

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    pyp = pypeliner.app.Pypeline([remixt], args)

    workflow = remixt.workflow.create_calc_bias_workflow(
        args['seqdata_file'],
        args['segments'],
        args['segment_lengths'],
        config,
    )

    pyp.run(workflow)

