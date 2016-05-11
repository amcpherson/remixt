import argparse

import remixt.ref_data


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('-c', '--config',
        help='Configuration filename')

    args = argparser.parse_args()

    args = vars(argparser.parse_args())

    ref_data_dir = args['ref_data_dir']

    config = {}
    if args['config'] is not None:
        execfile(args['config'], {}, config)

    remixt.ref_data.create_ref_data(config, ref_data_dir)
    
