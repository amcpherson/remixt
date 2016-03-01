import argparse
import os
import numpy as np
import pandas as pd

import remixt.utils
import remixt.analysis.segment


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()

    argparser.add_argument('ref_data_dir',
        help='Reference dataset directory')

    argparser.add_argument('breakpoint_file',
        help='Input breakpoint file')

    argparser.add_argument('segment_file',
        help='Output segment filename')

    argparser.add_argument('--config', required=False,
        help='Configuration Filename')

    args = vars(argparser.parse_args())

    config = {}
    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    remixt.analysis.segment.create_segments(
        args['segment_file'],
        config,
        args['ref_data_dir'],
        breakpoint_filename=args['breakpoint_file'],
    )



