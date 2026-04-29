#!/usr/bin/env python
"""Infer haplotype phasing for a single chromosome using SHAPEIT."""

import click
import yaml
import os
import tempfile
import remixt.analysis.haplotype


@click.command()
@click.option('--snp_genotype', required=True, help='Input SNP genotype TSV file')
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--config', required=True, help='YAML config file')
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--output', required=True, help='Output haplotypes TSV file')
def main(snp_genotype, chromosome, config, ref_data_dir, output):
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    # infer_haps needs a temp directory for SHAPEIT intermediate files
    temp_directory = os.path.join(os.getcwd(), 'shapeit_temp')
    os.makedirs(temp_directory, exist_ok=True)

    remixt.analysis.haplotype.infer_haps(
        output, snp_genotype, chromosome, temp_directory, cfg, ref_data_dir,
    )


if __name__ == '__main__':
    main()
