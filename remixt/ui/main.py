import argparse

import remixt.ui.run
import remixt.ui.create_ref_data
import remixt.ui.mappability_bwa


def main():
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = argparser.add_subparsers()
    
    remixt.ui.run.add_arguments(subparsers.add_parser('run'))
    remixt.ui.create_ref_data.add_arguments(subparsers.add_parser('create_ref_data'))
    remixt.ui.mappability_bwa.add_arguments(subparsers.add_parser('mappability_bwa'))

    args = vars(argparser.parse_args())
    func = args.pop('func')
    func(**args)
