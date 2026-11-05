#!/usr/bin/env python
"""Sample GC content and read counts at random genomic positions."""

import click
import remixt.analysis.gcbias
import remixt.defaults


@click.command()
@click.option('--seqdata', required=True, help='Input seqdata HDF5 file')
@click.option('--fragment_mean', required=True, type=float, help='Mean fragment length')
@click.option('--genome_fasta', required=True, help='Reference genome fasta file')
@click.option('--genome_fai', required=True, help='Reference genome .fai file')
@click.option('--mappability', required=True, help='Mappability HDF5 file')
@click.option('--output', required=True, help='Output GC samples TSV file')
@click.option('--chromosomes', default=','.join(remixt.defaults.chromosomes), show_default=True,
              callback=lambda ctx, param, value: value.split(','), help='Comma separated chromosomes')
@click.option('--chr_name_prefix', default=remixt.defaults.chr_name_prefix, show_default=True)
@click.option('--sample_gc_num_positions', type=int, default=remixt.defaults.sample_gc_num_positions, show_default=True)
@click.option('--gc_position_offset', type=int, default=remixt.defaults.gc_position_offset, show_default=True)
@click.option('--filter_duplicates', type=bool, default=remixt.defaults.filter_duplicates, show_default=True)
@click.option('--map_qual_threshold', type=int, default=remixt.defaults.map_qual_threshold, show_default=True)
@click.option('--random_seed', type=int, default=remixt.defaults.random_seed, show_default=True)
def main(seqdata, fragment_mean, genome_fasta, genome_fai, mappability, output, **kwargs):
    remixt.analysis.gcbias.sample_gc(
        output, seqdata, fragment_mean, genome_fasta, genome_fai, mappability, **kwargs,
    )


if __name__ == '__main__':
    main()
