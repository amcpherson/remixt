import argparse
import os
import yaml

import remixt.ref_data


def run(**args):
    ref_data_dir = args['ref_data_dir']

    config = {}
    if args['config'] is not None:
        config = yaml.load(open(args['config']))

    ref_data_sentinal = os.path.join(ref_data_dir, 'sentinal')

    remixt.ref_data.create_ref_data(config, ref_data_dir, ref_data_sentinal)


def add_arguments(argparser):
    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('-c', '--config',
        help='Configuration filename')

    argparser.add_argument('-b', '--bwa_index_genome',
        action='store_true',
        help='Index the genome for bwa, used for tests/benchmarking')

    argparser.set_defaults(func=run)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    add_arguments(argparser)

    args = vars(argparser.parse_args())
    func = args.pop('func')
    func(**args)
