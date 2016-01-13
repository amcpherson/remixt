# distutils: language = c++

from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "BamReader.h":
    void ExtractReads(
        string bamFilename,
        string snpFilename,
        int maxFragmentLength,
        int maxSoftClipped,
        string chromosome,
        string readsFilename,
        string allelesFilename,
        bool removeDuplicates,
        int mapQualThreshold,
    ) except +

def extract_reads(
        str bam_filename,
        str snp_filename,
        int max_fragment_length,
        int max_soft_clipped,
        str chromosome,
        str reads_filename,
        str alleles_filename,
        bool remove_duplicates=False,
        int map_qual_threshold=1,
):
    """ Extract read data from a bam file.

    Args:
        bam_filename(str): bam from which to extract read information
        snp_filename(str): TSV chromosome, position file listing SNPs
        max_fragment_length(int): maximum length of fragments generating paired reads
        max_soft_clipped(int): maximum soft clipping for considering a read concordant
        chromosome(str): chromosome to extract
        reads_filename(str): compressed read data output
        alleles_filename(str): allele data output

    KwArgs:
        remove_duplicates(bool): remove reads marked as duplicate
        map_qual_threshold(int): threshold mapping quality

    """
    ExtractReads(
        bam_filename,
        snp_filename,
        max_fragment_length,
        max_soft_clipped,
        chromosome,
        reads_filename,
        alleles_filename,
        remove_duplicates,
        map_qual_threshold,
    )
