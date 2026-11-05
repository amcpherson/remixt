#!/usr/bin/env python
"""Infer haplotype phasing for a single GRCh37 chromosome using SHAPEIT2."""

import click
import os
import remixt.analysis.haplotype
import remixt.defaults


@click.command()
@click.option('--snp_genotype', required=True, help='Input SNP genotype TSV file')
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--genetic_map', required=True, help='Impute2 genetic map for the chromosome')
@click.option('--haplotypes', required=True, help='Impute2 haplotypes for the chromosome')
@click.option('--legend', required=True, help='Impute2 legend for the chromosome')
@click.option('--sample', required=True, help='Impute2 sample file')
@click.option('--output', required=True, help='Output haplotypes TSV file')
@click.option('--temp_directory', required=True, help='Temp directory for SHAPEIT intermediate files')
@click.option('--is_female', type=bool, default=remixt.defaults.is_female, show_default=True)
@click.option('--shapeit_num_samples', type=int, default=remixt.defaults.shapeit_num_samples, show_default=True)
@click.option('--shapeit_confidence_threshold', type=float, default=remixt.defaults.shapeit_confidence_threshold, show_default=True)
def main(snp_genotype, chromosome, genetic_map, haplotypes, legend, sample, output, temp_directory, **kwargs):
    os.makedirs(temp_directory, exist_ok=True)

    remixt.analysis.haplotype.infer_haps_grch37_shapeit2(
        output, snp_genotype, chromosome, temp_directory,
        genetic_map, haplotypes, legend, sample,
        **kwargs,
    )


if __name__ == '__main__':
    main()
