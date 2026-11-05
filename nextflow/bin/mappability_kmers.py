#!/usr/bin/env python
"""Write the k-mers starting in a region of the genome as fasta, for alignment with bwa."""

import sys
import click
import remixt.mappability
import remixt.defaults


@click.command()
@click.option('--genome_fasta', required=True, help='Indexed genome fasta')
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--start', type=int, required=True, help='First k-mer start, 0-based')
@click.option('--end', type=int, required=True, help='End of k-mer starts, exclusive')
@click.option('--mappability_length', type=int, default=remixt.defaults.mappability_length, show_default=True,
              help='K-mer length')
def main(genome_fasta, chromosome, start, end, mappability_length):
    remixt.mappability.write_kmers(genome_fasta, chromosome, start, end, mappability_length, sys.stdout)


if __name__ == '__main__':
    main()
