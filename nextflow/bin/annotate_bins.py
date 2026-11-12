#!/usr/bin/env python
"""Regular bins over the selected chromosomes, with GC and mappable fraction."""

import click
import remixt.analysis.binbias
import remixt.defaults
import remixt.utils


@click.command()
@click.option('--genome_fasta', required=True, help='Reference genome fasta file')
@click.option('--genome_fai', required=True, help='Reference genome fasta index')
@click.option('--mappability', required=True, help='Mappability HDF5 file')
@click.option('--bin_size', type=int, required=True, help='Bin size')
@click.option('--output', required=True, help='Output bins TSV file')
@click.option('--chromosomes', default=','.join(remixt.defaults.chromosomes), show_default=True)
@click.option('--chr_name_prefix', default=remixt.defaults.chr_name_prefix, show_default=True)
@click.option('--map_qual_threshold', type=int, default=remixt.defaults.map_qual_threshold, show_default=True)
def main(genome_fasta, genome_fai, mappability, bin_size, output, chromosomes, chr_name_prefix, map_qual_threshold):
    chromosomes = chromosomes.split(',') if isinstance(chromosomes, str) else chromosomes
    lengths = remixt.utils.read_selected_chromosome_lengths(genome_fai, chromosomes, chr_name_prefix=chr_name_prefix)
    bins = remixt.analysis.binbias.make_bins(lengths, bin_size)
    bins = remixt.analysis.binbias.annotate_bins(bins, genome_fasta, mappability, map_qual_threshold=map_qual_threshold)
    bins.to_csv(output, sep='\t', index=False)


if __name__ == '__main__':
    main()
