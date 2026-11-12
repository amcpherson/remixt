""" Mappability of a reference genome, from aligning each k-mer of the genome
back to the genome with bwa.

A position is mappable with the mapping quality of the k-mer starting at that
position, if the k-mer aligns back to that position.  Positions are stored as
runs of consecutive positions with the same mapping quality.
"""

import numpy as np
import pandas as pd
import pysam


def write_kmers(genome_fasta, chromosome, start, end, kmer_length, out_file):
    """ Write k-mers starting in a region of a chromosome as fasta.

    Args:
        genome_fasta (str): indexed genome fasta
        chromosome (str): chromosome name
        start (int): first k-mer start position, 0-based
        end (int): end of k-mer start positions, exclusive
        kmer_length (int): k-mer length
        out_file (file): output file object

    K-mers are named chromosome:position, and k-mers containing N are skipped.
    """
    with pysam.FastaFile(genome_fasta) as fasta:
        end = min(end, fasta.get_reference_length(chromosome) - kmer_length + 1)
        if end <= start:
            return
        sequence = fasta.fetch(chromosome, start, end + kmer_length - 1).upper()

    for offset in range(end - start):
        kmer = sequence[offset:offset + kmer_length]
        if 'N' in kmer:
            continue
        out_file.write('>{}:{}\n{}\n'.format(chromosome, start + offset, kmer))


def read_origin_mapping_quality(sam_file):
    """ Read the mapping quality of k-mers that align back to their origin.

    Args:
        sam_file (file): SAM alignments of k-mers written by `write_kmers`

    Returns:
        pandas.DataFrame: chromosome, position, quality

    Secondary and supplementary alignments are ignored.
    """
    chromosomes = []
    positions = []
    qualities = []

    for line in sam_file:
        if line.startswith('@'):
            continue
        fields = line.split('\t', 5)
        if int(fields[1]) & 0x900:
            continue
        origin_chromosome, origin_position = fields[0].rsplit(':', 1)
        origin_position = int(origin_position)
        if fields[2] != origin_chromosome or int(fields[3]) - 1 != origin_position:
            continue
        chromosomes.append(origin_chromosome)
        positions.append(origin_position)
        qualities.append(int(fields[4]))

    return pd.DataFrame({
        'chromosome': pd.Series(chromosomes, dtype=str),
        'position': np.array(positions, dtype=np.int64),
        'quality': np.array(qualities, dtype=np.int64),
    })


def calculate_mappability_runs(mapping_quality):
    """ Collapse per position mapping quality into runs of consecutive
    positions with the same mapping quality.

    Args:
        mapping_quality (pandas.DataFrame): chromosome, position, quality

    Returns:
        pandas.DataFrame: chromosome, start, end, quality, end exclusive
    """
    columns = ['chromosome', 'start', 'end', 'quality']

    if len(mapping_quality) == 0:
        return pd.DataFrame(columns=columns)

    data = mapping_quality.sort_values(['chromosome', 'position'], kind='mergesort')

    is_run_start = (
        (data['chromosome'] != data['chromosome'].shift()) |
        (data['position'] != data['position'].shift() + 1) |
        (data['quality'] != data['quality'].shift()))
    run = is_run_start.cumsum()

    runs = data.groupby(run, sort=True).agg(
        chromosome=('chromosome', 'first'),
        start=('position', 'min'),
        end=('position', 'max'),
        quality=('quality', 'first'),
    )
    runs['end'] += 1

    return runs[columns].reset_index(drop=True)


def merge_mappability(runs_filenames, mappability_filename):
    """ Merge mappability runs into an HDF5 mappability file.

    Args:
        runs_filenames (list of str): runs tables in genome order
        mappability_filename (str): output HDF5, one table per chromosome

    Tables are named chromosome_{chromosome}, with columns start, end and quality,
    as read by `remixt.analysis.binbias.read_mappability_indicator`.
    """
    with pd.HDFStore(mappability_filename, 'w') as store:
        for runs_filename in runs_filenames:
            runs = pd.read_csv(runs_filename, sep='\t', dtype={'chromosome': str})
            for chromosome, chrom_runs in runs.groupby('chromosome', sort=False):
                store.append(
                    'chromosome_' + chromosome,
                    chrom_runs[['start', 'end', 'quality']].astype(np.int64),
                    data_columns=True)
