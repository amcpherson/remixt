import argparse
import os
import numpy as np
import pandas as pd

import remixt.utils


def create_segments(segment_filename, breakpoint_filename, segment_length, chromosomes, genome_fai_filename):

    chromosome_lengths = remixt.utils.read_chromosome_lengths(genome_fai_filename)

    changepoints = list()

    # Add regular segments
    for chromosome in chromosomes:
        length = chromosome_lengths[chromosome]
        for position in np.arange(0, length, segment_length, dtype=int):
            changepoints.append((chromosome, position))
        changepoints.append((chromosome, length))

    # Add breakpoint segments
    breakpoints = pd.read_csv(
        breakpoint_filename, sep='\t',
        converters={'chromosome_1':str, 'chromosome_2':str, 'position_1':int, 'position_2':int}
    )

    for idx, row in breakpoints.iterrows():
        changepoints.append((row['chromosome_1'], row['position_1']))
        changepoints.append((row['chromosome_2'], row['position_2']))

    changepoints = pd.DataFrame(changepoints, columns=['chromosome', 'position'])
    changepoints.sort(['chromosome', 'position'], inplace=True)

    # Create segments from changepoints
    segments = list()
    for chromosome, chrom_changepoints in changepoints.groupby('chromosome'):
        chrom_segments = pd.DataFrame({
            'start':chrom_changepoints['position'].values[:-1],
            'end':chrom_changepoints['position'].values[1:],
        })
        chrom_segments['chromosome'] = chromosome
        segments.append(chrom_segments)
    segments = pd.concat(segments, ignore_index=True)

    # Sort segments by placement in chromosome list, and position
    segments = segments.merge(pd.DataFrame(list(enumerate(chromosomes)), columns=['chromosome_idx', 'chromosome']))
    segments.sort(['chromosome_idx', 'start'], inplace=True)

    segments.to_csv(segment_filename, sep='\t', index=False, columns=['chromosome', 'start', 'end'])


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

    remixt_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_config_filename = os.path.join(remixt_directory, 'defaultconfig.py')

    config = {'ref_data_directory':args['ref_data_dir']}
    execfile(default_config_filename, {}, config)

    if args['config'] is not None:
        execfile(args['config'], {}, config)

    config.update(args)

    create_segments(
        args['segment_file'],
        args['breakpoint_file'],
        config['segment_length'],
        config['chromosomes'],
        config['genome_fai'],
    )


