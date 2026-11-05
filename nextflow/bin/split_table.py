#!/usr/bin/env python
"""Split a TSV table into fixed-size row chunks."""

import click
import remixt.utils


@click.command()
@click.option('--input', 'input_file', required=True, help='Input TSV file')
@click.option('--num_rows', required=True, type=int, default=100, help='Rows per chunk')
@click.option('--output_prefix', required=True, help='Output prefix for chunk files')
def main(input_file, num_rows, output_prefix):
    # Read the table to determine number of chunks
    input_data = remixt.utils.read_table_raw(input_file)

    output_filenames = {}
    for idx, start_row in enumerate(range(0, len(input_data.index), num_rows)):
        output_filenames[idx] = f'{output_prefix}.{idx}.tsv'

    remixt.utils.split_table(output_filenames, input_file, num_rows)


if __name__ == '__main__':
    main()
