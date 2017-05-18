import argparse

import remixt.ui.run
import remixt.ui.create_ref_data
import remixt.ui.mappability_bwa
import remixt.ui.write_results
import remixt.ui.plot_results
import remixt.ui.visualize_solutions


def main():
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = argparser.add_subparsers()
    
    remixt.ui.run.add_arguments(subparsers.add_parser('run'))
    remixt.ui.create_ref_data.add_arguments(subparsers.add_parser('create_ref_data'))
    remixt.ui.mappability_bwa.add_arguments(subparsers.add_parser('mappability_bwa'))
    remixt.ui.write_results.add_arguments(subparsers.add_parser('write_results'))
    remixt.ui.plot_results.add_arguments(subparsers.add_parser('plot_results'))
    remixt.ui.visualize_solutions.add_arguments(subparsers.add_parser('visualize_solutions'))

    args = vars(argparser.parse_args())
    func = args.pop('func')
    func(**args)
