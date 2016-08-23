import argparse
import os

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

    ref_data_sentinal = os.path.join(ref_data_dir, 'sentinal')

    remixt.ref_data.create_ref_data(config, ref_data_dir, ref_data_sentinal)
    
