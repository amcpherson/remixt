#!/usr/bin/env python
"""Infer haplotype phasing for a single GRCh38 chromosome using SHAPEIT4."""

import click
import os
import remixt.analysis.haplotype
import remixt.defaults


@click.command()
@click.option('--snp_genotype', required=True, help='Input SNP genotype TSV file')
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--snp_positions', required=True, help='Thousand genomes SNP positions file')
@click.option('--bcf_reference', required=True, help='Thousand genomes phased panel BCF for the chromosome')
@click.option('--genetic_map', required=True, help='Genetic map for the chromosome')
@click.option('--output', required=True, help='Output haplotypes TSV file')
@click.option('--temp_directory', required=True, help='Temp directory for SHAPEIT intermediate files')
@click.option('--chr_name_prefix', default=remixt.defaults.chr_name_prefix, show_default=True)
@click.option('--is_female', type=bool, default=remixt.defaults.is_female, show_default=True)
@click.option('--shapeit_num_samples', type=int, default=remixt.defaults.shapeit_num_samples, show_default=True)
@click.option('--shapeit_confidence_threshold', type=float, default=remixt.defaults.shapeit_confidence_threshold, show_default=True)
def main(snp_genotype, chromosome, snp_positions, bcf_reference, genetic_map, output, temp_directory, **kwargs):
    os.makedirs(temp_directory, exist_ok=True)

    remixt.analysis.haplotype.infer_haps_grch38_shapeit4(
        output, snp_genotype, chromosome, temp_directory,
        snp_positions, bcf_reference, genetic_map,
        **kwargs,
    )


if __name__ == '__main__':
    main()
