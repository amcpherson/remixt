#!/usr/bin/env python
"""Infer haplotype phasing for a single chromosome using SHAPEIT."""

import json
import click
import os
import remixt.analysis.haplotype


@click.command()
@click.option('--snp_genotype', required=True, help='Input SNP genotype TSV file')
@click.option('--chromosome', required=True, help='Chromosome')
@click.option('--ref_data_dir', required=True, help='Reference data directory')
@click.option('--output', required=True, help='Output haplotypes TSV file')
@click.option('--temp_directory', required=True, help='Temp directory for SHAPEIT intermediate files')
@click.option('--config', type=click.File(), required=True, help='remixt config JSON file')
def main(snp_genotype, chromosome, ref_data_dir, output, temp_directory, config):
    cfg = json.load(config)

    os.makedirs(temp_directory, exist_ok=True)

    remixt.analysis.haplotype.infer_haps(
        output, snp_genotype, chromosome, temp_directory, cfg, ref_data_dir,
    )


if __name__ == '__main__':
    main()
