import argparse
import logging
import os

import remixt.visualize


def parse_results_arg(arg):
    """ Parse a results argument, either tumour_id:path or a path named after the tumour.
    """
    if ':' in arg and not os.path.exists(arg):
        tumour_id, path = arg.split(':', 1)
        return tumour_id, path
    tumour_id = os.path.basename(arg)
    for suffix in ('.results.h5', '.h5'):
        if tumour_id.endswith(suffix):
            tumour_id = tumour_id[:-len(suffix)]
            break
    return tumour_id, arg


def create_visualization(**args):
    logging.basicConfig(level=logging.INFO)
    results_filenames = dict(parse_results_arg(a) for a in args['results'])
    remixt.visualize.create_report(
        results_filenames, args['html'], title=args['title'])


def add_arguments(argparser):
    argparser.add_argument('html',
        help='HTML output report')

    argparser.add_argument('results', nargs='+',
        help='Results HDF5 files to visualize, as tumour_id:path or a path named <tumour_id>.h5')

    argparser.add_argument('--title', default='ReMixT results',
        help='Title of the report')

    argparser.set_defaults(func=create_visualization)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    add_arguments(argparser)

    args = vars(argparser.parse_args())
    func = args.pop('func')
    func(**args)
