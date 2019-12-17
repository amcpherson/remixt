import numpy as np
import pandas as pd

import remixt.bamreader

import os


empty_data = {
    'fragments': remixt.bamreader.create_fragment_table(0),
    'alleles': remixt.bamreader.create_allele_table(0),
}


def _get_key(record_type, chromosome):
    return '/{}/chromosome_{}'.format(record_type, chromosome)


def _unique_index_append(store, key, data):
    try:
        nrows = store.get_storer(key).nrows
    except (AttributeError, KeyError):
        nrows = 0
    data.index = pd.Series(data.index) + nrows
    if nrows == 0:
        store.put(key, data, format='table')
    else:
        store.append(key, data)


def merge_overlapping_seqdata(outfile, infiles, chromosomes):
    out_store = pd.HDFStore(outfile, 'w', complevel=9, complib='blosc')

    index_offsets = pd.Series(0, index=chromosomes, dtype=np.int64)

    for _id, infile in infiles.items():
        store = pd.HDFStore(infile)
        tables = store.keys()

        for chromosome in chromosomes:
            allele_table = '/alleles/chromosome_{}'.format(chromosome)
            fragment_table = '/fragments/chromosome_{}'.format(chromosome)

            if allele_table not in tables:
                print("missing table {}".format(allele_table))
                continue

            if fragment_table not in tables:
                print("missing table {}".format(fragment_table))
                continue

            alleles = store[allele_table]
            fragments = store[fragment_table]

            alleles['fragment_id'] = alleles['fragment_id'].astype(np.int64)
            fragments['fragment_id'] = fragments['fragment_id'].astype(np.int64)

            alleles['fragment_id'] += index_offsets[chromosome]
            fragments['fragment_id'] += index_offsets[chromosome]

            index_offsets[chromosome] = max(alleles['fragment_id'].max(), fragments['fragment_id'].max()) + 1

            out_store.append('/alleles/chromosome_{}'.format(chromosome), alleles)
            out_store.append('/fragments/chromosome_{}'.format(chromosome), fragments)

        store.close()
    out_store.close()


def create_chromosome_seqdata(seqdata_filename, bam_filename, snp_filename, chromosome, max_fragment_length, max_soft_clipped, check_proper_pair):
    """ Create seqdata from bam for one chromosome.

    Args:
        seqdata_filename(str): seqdata hdf store to write to
        bam_filename(str): bam from which to extract read information
        snp_filename(str): TSV chromosome, position file listing SNPs
        chromosome(str): chromosome to extract
        max_fragment_length(int): maximum length of fragments generating paired reads
        max_soft_clipped(int): maximum soft clipping for considering a read concordant
        check_proper_pair(boo): check proper pair flag

    """

    reader = remixt.bamreader.AlleleReader(
        bam_filename,
        snp_filename,
        chromosome,
        max_fragment_length,
        max_soft_clipped,
        check_proper_pair,
    )

    with pd.HDFStore(seqdata_filename, 'w', complevel=9, complib='zlib') as store:
        while reader.ReadAlignments(10000000):
            _unique_index_append(store, _get_key('fragments', chromosome), reader.GetFragmentTable())
            _unique_index_append(store, _get_key('alleles', chromosome), reader.GetAlleleTable())



def create_seqdata(seqdata_filename, bam_filename, snp_filename, max_fragment_length, max_soft_clipped, check_proper_pair, tempdir, chromosomes):

    try:
        os.makedirs(tempdir)
    except:
        pass

    all_seqdata = {}

    for chrom in chromosomes:
        chrom_seqdata = os.path.join(tempdir, "{}_seqdata.h5".format(chrom))
        all_seqdata[chrom] = chrom_seqdata

        create_chromosome_seqdata(
            chrom_seqdata, bam_filename, snp_filename,
            chrom, max_fragment_length, max_soft_clipped,
            check_proper_pair
        )

    merge_seqdata(seqdata_filename, all_seqdata)


def merge_seqdata(out_filename, in_filenames):
    """ Merge seqdata files for non-overlapping sets of chromosomes

    Args:
        out_filename(str): seqdata hdf store to write to
        out_filename(dict): seqdata hdf store to read from

    """

    with pd.HDFStore(out_filename, 'w', complevel=9, complib='zlib') as out_store:
        for in_filename in in_filenames.values():
            with pd.HDFStore(in_filename, 'r') as in_store:
                for key in in_store.keys():
                    out_store.put(key, in_store[key], format='table')


class Writer(object):
    def __init__(self, seqdata_filename):
        """ Streaming writer of seq data hdf5 files 

        Args:
            seqdata_filename (str): name of seqdata hdf5 file

        """

        self.store = pd.HDFStore(seqdata_filename, 'w', complevel=9, complib='zlib')

    def write(self, chromosome, fragment_data, allele_data):
        """ Write a chunk of reads and alleles data

        Args:
            fragment_data (pandas.DataFrame): fragment data
            allele_data (pandas.DataFrame): allele data

        Input 'fragment_data' dataframe has columns 'fragment_id', 'start', 'end'.
        if columns 'is_duplicate', 'mapping_quality' are not provided they are
        given nominal values.

        Input 'allele_data' dataframe has columns 'position', 'fragment_id', 'is_alt'.

        """

        # Add nominal mapping quality
        if 'mapping_quality' not in fragment_data:
            fragment_data['mapping_quality'] = 60

        # Add nominal is_duplicate value
        if 'is_duplicate' not in fragment_data:
            fragment_data['is_duplicate'] = 0

        fragment_data = fragment_data[['fragment_id', 'start', 'end', 'is_duplicate', 'mapping_quality']]
        allele_data = allele_data[['position', 'fragment_id', 'is_alt']]

        _unique_index_append(self.store, _get_key('fragments', chromosome), fragment_data)
        _unique_index_append(self.store, _get_key('alleles', chromosome), allele_data)

    def close(self):
        """ Close seq data file
        
        """

        self.store.close()


_identity = lambda x: x


def _read_seq_data_full(seqdata_filename, record_type, chromosome, post=_identity):
    key = _get_key(record_type, chromosome)
    try:
        return post(pd.read_hdf(seqdata_filename, key))
    except KeyError:
        return empty_data[record_type]


def _get_seq_data_nrows(seqdata_filename, key):
    with pd.HDFStore(seqdata_filename, 'r') as store:
        try:
            return store.get_storer(key).nrows
        except (AttributeError, KeyError):
            return 0


def _read_seq_data_chunks(seqdata_filename, record_type, chromosome, chunksize, post=_identity):
    key = _get_key(record_type, chromosome)
    nrows = _get_seq_data_nrows(seqdata_filename, key)
    if nrows == 0:
        yield empty_data[record_type]
    else:
        for i in range(nrows//chunksize + 1):
            yield post(pd.read_hdf(seqdata_filename, key, start=i*chunksize, stop=(i+1)*chunksize))


def read_seq_data(seqdata_filename, record_type, chromosome, chunksize=None, post=_identity):
    """ Read sequence data from a HDF seqdata file.

    Args:
        seqdata_filename (str): name of seqdata file
        record_type (str): record type, can be 'alleles' or 'reads'
        chromosome (str): select specific chromosome

    KwArgs:
        chunksize (int): number of rows to stream at a time, None for the entire file
        post (callable): post processing function

    Yields:
        pandas.DataFrame

    """

    if chunksize is None:
        return _read_seq_data_full(seqdata_filename, record_type, chromosome, post=post)
    else:
        return _read_seq_data_chunks(seqdata_filename, record_type, chromosome, chunksize, post=post)


def read_fragment_data(seqdata_filename, chromosome, filter_duplicates=False, map_qual_threshold=1, chunksize=None):
    """ Read fragment data from a HDF seqdata file.

    Args:
        seqdata_filename (str): name of seqdata file
        chromosome (str): select specific chromosome, None for all chromosomes

    KwArgs:
        filter_duplicates (bool): filter reads marked as duplicate
        map_qual_threshold (int): filter reads with less than this mapping quality
        chunksize (int): number of rows to stream at a time, None for the entire file

    Yields:
        pandas.DataFrame

    Returned dataframe has columns 'fragment_id', 'start', 'end'

    """

    def filter_reads(reads):
        # Filter duplicates if necessary
        if 'is_duplicate' in reads and filter_duplicates is not None:
            if filter_duplicates:
                reads = reads[reads['is_duplicate'] == 0]
            reads.drop(['is_duplicate'], axis=1, inplace=True)

        # Filter poor quality reads
        if 'mapping_quality' in reads and map_qual_threshold is not None:
            reads = reads[reads['mapping_quality'] >= map_qual_threshold]
            reads.drop(['mapping_quality'], axis=1, inplace=True)

        return reads

    return read_seq_data(seqdata_filename, 'fragments', chromosome, chunksize=chunksize, post=filter_reads)


def read_allele_data(seqdata_filename, chromosome, chunksize=None):
    """ Read allele data from a HDF seqdata file.

    Args:
        seqdata_filename (str): name of seqdata file
        chromosome (str): select specific chromosome, None for all chromosomes

    KwArgs:
        chunksize (int): number of rows to stream at a time, None for the entire file

    Yields:
        pandas.DataFrame

    Returned dataframe has columns 'position', 'is_alt', 'fragment_id'

    """

    return read_seq_data(seqdata_filename, 'alleles', chromosome, chunksize=chunksize)


def read_chromosomes(seqdata_filename):
    """ Read chromosomes from a HDF seqdata file.

    Args:
        seqdata_filename (str): name of seqdata file

    Returns:
        list of chromsomes

    """

    with pd.HDFStore(seqdata_filename, 'r') as store:
        chromosomes = set()
        for key in store.keys():
            if 'chromosome_' in key:
                chromosomes.add(key[key.index('chromosome_') + len('chromosome_'):])

        return chromosomes


