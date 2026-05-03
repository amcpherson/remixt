#!/usr/bin/env python
"""Infer haplotype phasing for a single chromosome using SHAPEIT."""

import click
import os
import remixt.analysis.haplotype


@click.command()
@click.option('--snp_genotype', required=True, help='Input SNP genotype TSV file')
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--output', required=True, help='Output haplotypes TSV file')
@click.option('--ensembl_genome_version', default='GRCh38', help='Genome version (GRCh38 or GRCh37)')
@click.option('--chr_name_prefix', default='', help='Chromosome name prefix')
@click.option('--is_female/--is_male', default=True, help='Sample is female')
@click.option('--shapeit_num_samples', type=int, default=100, help='SHAPEIT num samples')
@click.option('--shapeit_confidence_threshold', type=float, default=0.95, help='SHAPEIT confidence threshold')
@click.option('--temp_directory', required=True, help='Temp directory for SHAPEIT intermediate files')
def main(snp_genotype, chromosome, ref_data_dir, output,
         ensembl_genome_version, chr_name_prefix, is_female,
         shapeit_num_samples, shapeit_confidence_threshold, temp_directory):

    cfg = {
        'ensembl_genome_version': ensembl_genome_version,
        'chr_name_prefix': chr_name_prefix,
        'is_female': is_female,
        'shapeit_num_samples': shapeit_num_samples,
        'shapeit_confidence_threshold': shapeit_confidence_threshold,
    }

    os.makedirs(temp_directory, exist_ok=True)

    remixt.analysis.haplotype.infer_haps(
        output, snp_genotype, chromosome, temp_directory, cfg, ref_data_dir,
    )


if __name__ == '__main__':
    main()
