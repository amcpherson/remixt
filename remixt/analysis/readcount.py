import pandas as pd

import remixt.config
import remixt.analysis.segment
import remixt.analysis.haplotype


def segment_readcount(segment_counts_filename, segment_filename, seqdata_filename, config):

    segments = pd.read_csv(segment_filename, sep='\t', converters={'chromosome': str})

    filter_duplicates = remixt.config.get_param(config, 'filter_duplicates')
    map_qual_threshold = remixt.config.get_param(config, 'map_qual_threshold')

    segment_counts = remixt.analysis.segment.create_segment_counts(
        segments,
        seqdata_filename,
        filter_duplicates=filter_duplicates,
        map_qual_threshold=map_qual_threshold,
    )

    segment_counts.to_csv(segment_counts_filename, sep='\t', index=False)


def haplotype_allele_readcount(allele_counts_filename, segment_filename, seqdata_filename, haps_filename, config):
    
    segments = pd.read_csv(segment_filename, sep='\t', converters={'chromosome': str})

    filter_duplicates = remixt.config.get_param(config, 'filter_duplicates')
    map_qual_threshold = remixt.config.get_param(config, 'map_qual_threshold')

    allele_counts = remixt.analysis.haplotype.create_allele_counts(
        segments,
        seqdata_filename,
        haps_filename,
        filter_duplicates=filter_duplicates,
        map_qual_threshold=map_qual_threshold,
    )

    allele_counts.to_csv(allele_counts_filename, sep='\t', index=False)


def phase_segments(allele_counts_filenames, phased_allele_counts_filenames):

    tumour_ids = allele_counts_filenames.keys()

    allele_count_tables = list()
    for allele_counts_filename in allele_counts_filenames.values():
        allele_count_tables.append(pd.read_csv(allele_counts_filename, sep='\t', converters={'chromosome': str}))

    phased_allele_counts_tables = remixt.analysis.haplotype.phase_segments(*allele_count_tables)

    for tumour_id, phased_allele_counts in zip(tumour_ids, phased_allele_counts_tables):
        phased_allele_counts_filename = phased_allele_counts_filenames[tumour_id]
        phased_allele_counts.to_csv(phased_allele_counts_filename, sep='\t', index=False)


def prepare_readcount_table(segments_filename, alleles_filename, count_filename):

    segment_data = pd.read_csv(segments_filename, sep='\t', converters={'chromosome': str})
    allele_data = pd.read_csv(alleles_filename, sep='\t', converters={'chromosome': str})

    segment_allele_counts = remixt.analysis.segment.create_segment_allele_counts(segment_data, allele_data)

    segment_allele_counts.to_csv(count_filename, sep='\t', index=False)



