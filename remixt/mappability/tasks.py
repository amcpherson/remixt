import csv
import numpy as np
import pandas as pd

import remixt
import remixt.utils


def create_kmers(genome_fasta, k, kmers_filename):
    with open(kmers_filename, 'w') as kmers_file:
        genome_sequences = dict(remixt.utils.read_sequences(genome_fasta))
        for chromosome, sequence in genome_sequences.items():
            chromosome = chromosome.split()[0]
            for start in range(len(sequence)):
                kmer = sequence[start:start+k].upper()
                if len(kmer) < k:
                    continue
                if 'N' in kmer:
                    continue
                kmers_file.write('>{0}:{1}\n{2}\n'.format(chromosome, start, kmer))


def split_file_byline(in_filename, lines_per_file, out_filename_callback):
    with open(in_filename, 'r') as in_file:
        file_number = 0
        out_file = None
        out_file_lines = None
        try:
            for line in in_file:
                if out_file is None or out_file_lines == lines_per_file:
                    if out_file is not None:
                        out_file.close()
                    out_file = open(out_filename_callback(file_number), 'w')
                    out_file_lines = 0
                    file_number += 1
                out_file.write(line)
                out_file_lines += 1
        finally:
            if out_file is not None:
                out_file.close()


def create_bedgraph(alignment_filename, bedgraph_filename):
    mqual_table = list()
    with open(alignment_filename, 'r') as alignment_file:
        for row in csv.reader(alignment_file, delimiter='\t'):
            if row[0][0] == '@':
                continue
            origin_chromosome = row[0].split(':')[0]
            origin_position = int(row[0].split(':')[1])
            mapping_chromosome = row[2]
            mapping_position = int(row[3]) - 1   # 0-based positions
            mapping_quality = int(row[4])
            if origin_chromosome != mapping_chromosome:
                continue
            if origin_position != mapping_position:
                continue
            mqual_table.append((origin_chromosome, origin_position, mapping_quality))
        mqual_table = pd.DataFrame(mqual_table, columns=['chromosome', 'position', 'quality'])
        mqual_table['chromosome_index'] = np.searchsorted(np.unique(mqual_table['chromosome']), mqual_table['chromosome'])
        mqual_table.sort_values(['chromosome_index', 'position'], inplace=True)
        mqual_table['chromosome_diff'] = mqual_table['chromosome_index'].diff()
        mqual_table['position_diff'] = mqual_table['position'].diff() - 1
        mqual_table['quality_diff'] = mqual_table['quality'].diff()
        mqual_table['is_diff'] = (mqual_table[['chromosome_diff', 'position_diff', 'quality_diff']].sum(axis=1) != 0)
        mqual_table['group'] = mqual_table['is_diff'].cumsum()
        def agg_positions(data):
            return pd.Series({
                    'chromosome': data['chromosome'].iloc[0],
                    'start': data['position'].min(),
                    'end': data['position'].max() + 1,
                    'quality': data['quality'].iloc[0],
            })
        mqual_table = mqual_table.groupby('group').apply(agg_positions)
        mqual_table.to_csv(
            bedgraph_filename, sep='\t', index=False, header=False,
            columns=['chromosome', 'start', 'end', 'quality'])
                

def merge_files_by_line(in_filenames, out_filename):
    with pd.HDFStore(out_filename, 'w') as store:
        for in_filename in in_filenames.values():
            data = pd.read_csv(
                in_filename, sep='\t', header=None,
                names=['chromosome', 'start', 'end', 'quality'],
                converters={'chromosome':str})
            for chromosome, chrom_data in data.groupby('chromosome'):
                store.append('chromosome_'+chromosome, chrom_data[['start', 'end', 'quality']], data_columns=True)

