import os
import contextlib
import tarfile
import gzip
import StringIO
import subprocess
import numpy as np
import pandas as pd

import remixt.bamreader


empty_data = {
    'fragments': remixt.bamreader.create_fragment_table(0),
    'alleles': remixt.bamreader.create_allele_table(0),
}


read_data_dtype = np.dtype([('start', np.uint32), ('length', np.uint16)])


allele_data_dtype = np.dtype([('fragment_id', np.uint32), ('position', np.uint32), ('is_alt', np.uint8)])


def _get_key(record_type, chromosome):
    return '/{}/chromosome_{}'.format(record_type, chromosome)


def _unique_index_append(store, key, data):
    try:
        nrows = store.get_storer(key).nrows
    except AttributeError:
        nrows = 0
    data.index = pd.Series(data.index) + nrows
    if nrows == 0:
        store.put(key, data, format='table')
    else:
        store.append(key, data)


def create_chromosome_seqdata(seqdata_filename, bam_filename, snp_filename, chromosome, max_fragment_length, max_soft_clipped):
    """ Create seqdata from bam for one chromosome.

    Args:
        seqdata_filename(str): seqdata hdf store to write to
        bam_filename(str): bam from which to extract read information
        snp_filename(str): TSV chromosome, position file listing SNPs
        chromosome(str): chromosome to extract
        max_fragment_length(int): maximum length of fragments generating paired reads
        max_soft_clipped(int): maximum soft clipping for considering a read concordant

    """

    reader = remixt.bamreader.AlleleReader(
        bam_filename,
        snp_filename,
        chromosome,
        max_fragment_length,
        max_soft_clipped,
    )

    with pd.HDFStore(seqdata_filename, 'w') as store:
        while reader.ReadAlignments(10000000):
            _unique_index_append(store, _get_key('fragments', chromosome), reader.GetFragmentTable())
            _unique_index_append(store, _get_key('alleles', chromosome), reader.GetAlleleTable())


def merge_seqdata(out_filename, in_filenames):
    """ Merge seqdata files for non-overlapping sets of chromosomes

    Args:
        out_filename(str): seqdata hdf store to write to
        out_filename(dict): seqdata hdf store to read from

    """

    with pd.HDFStore(out_filename, 'w') as out_store:
        for in_filename in in_filenames.itervalues():
            with pd.HDFStore(in_filename, 'r') as in_store:
                for key in in_store.keys():
                    out_store.put(key, in_store[key], format='table')


def write_read_data(reads_file, read_data):
    """ Write read data for a specific chromosome to a file

    Args:
        reads_file (str): tar to which data will be written
        read_data (pandas.DataFrame): read data

    Input 'read_data' dataframe has columns 'start', 'end'.

    """

    raw_data = np.zeros(len(read_data.index), dtype=read_data_dtype)

    raw_data['start'] = read_data['start']
    raw_data['length'] = read_data['end'] - read_data['start']

    raw_data.tofile(reads_file)


def write_allele_data(alleles_file, allele_data, fragment_id_offset=0):
    """ Write allele data for a specific chromosome to a file

    Args:
        alleles_file (str): tar to which data will be written
        allele_data (pandas.DataFrame): allele data

    Input 'allele_data' dataframe has columns 'position', 'fragment_id', 'is_alt'.

    """

    raw_data = np.zeros(len(allele_data.index), dtype=allele_data_dtype)

    raw_data['fragment_id'] = allele_data['fragment_id'] + fragment_id_offset
    raw_data['position'] = allele_data['position']
    raw_data['is_alt'] = allele_data['is_alt']

    raw_data.tofile(alleles_file)


class Writer(object):
    def __init__(self, seqdata_filename, temp_dir):
        """ Streaming writer of seq data files 

        Args:
            seqdata_filename (str): name of seqdata tar file
            temp_dir (str): temporary directory to write to

        """

        self.seqdata_filename = seqdata_filename
        self.temp_dir = temp_dir

        self.fragment_id_offset = dict()
        self.reads_filenames = dict()
        self.alleles_filenames = dict()

        try:
            os.makedirs(self.temp_dir)
        except OSError as e:
            if e.errno != 17:
                raise

    def get_reads_filename(self, chromosome):
        return os.path.join(self.temp_dir, 'reads.{0}'.format(chromosome))

    def get_alleles_filename(self, chromosome):
        return os.path.join(self.temp_dir, 'alleles.{0}'.format(chromosome))

    def write(self, chromosome, read_data, allele_data):
        """ Write a chunk of reads and alleles data

        Args:
            read_data (pandas.DataFrame): read data
            allele_data (pandas.DataFrame): allele data

        Input 'read_data' dataframe has columns 'chromosome', 'start', 'end'.
        Input 'allele_data' dataframe has columns 'chromosome', 'position', 'fragment_id', 'is_alt'.

        """

        if chromosome not in self.fragment_id_offset:

            with open(self.get_reads_filename(chromosome), 'w'):
                pass

            with open(self.get_alleles_filename(chromosome), 'w'):
                pass

            self.reads_filenames[chromosome] = self.get_reads_filename(chromosome)
            self.alleles_filenames[chromosome] = self.get_alleles_filename(chromosome)

            self.fragment_id_offset[chromosome] = 0

        with open(self.get_reads_filename(chromosome), 'ab') as f:
            write_read_data(f, read_data)

        with open(self.get_alleles_filename(chromosome), 'ab') as f:
            write_allele_data(f, allele_data, self.fragment_id_offset[chromosome])

        self.fragment_id_offset[chromosome] += len(read_data.index)

    def gzip_files(self, filenames):
        """ Gzip files

        Args:
            filenames (dict): files to gzip, keyed by chromosome

        Returns: 
            dict: gzipped files, keyed by chromosome

        """

        gzipped = dict()

        for chrom, filename in filenames.iteritems():

            try:
                os.remove(filename + '.gz')
            except OSError as e:
                if e.errno != 2:
                    raise e

            subprocess.check_call(['gzip', filename])

            gzipped[chrom] = filename + '.gz'

        return gzipped

    def close(self):
        """ Write final seqdata
        
        """

        self.reads_filenames = self.gzip_files(self.reads_filenames)
        self.alleles_filenames = self.gzip_files(self.alleles_filenames)

        create_seqdata(self.seqdata_filename, self.reads_filenames, self.alleles_filenames)


def _read_seq_data_full(seqdata_filename, record_type, chromosome):
    key = _get_key(record_type, chromosome)
    try:
        return pd.read_hdf(seqdata_filename, key)
    except KeyError:
        return empty_data[record_type]


def _get_seq_data_nrows(seqdata_filename, key):
    with pd.HDFStore(seqdata_filename, 'r') as store:
        try:
            return store.get_storer(key).nrows
        except AttributeError:
            return 0


def _read_seq_data_chunks(seqdata_filename, record_type, chromosome, chunksize):
    key = _get_key(record_type, chromosome)
    nrows = _get_seq_data_nrows(seqdata_filename, key)
    if nrows == 0:
        yield empty_data[record_type]
    else:
        for i in xrange(nrows//chunksize + 1):
            yield pd.read_hdf(seqdata_filename, key, start=i*chunksize, stop=(i+1)*chunksize)


def read_seq_data(seqdata_filename, record_type, chromosome, chunksize=None):
    """ Read sequence data from a HDF seqdata file.

    Args:
        seqdata_filename (str): name of seqdata file
        record_type (str): record type, can be 'alleles' or 'reads'
        chromosome (str): select specific chromosome

    KwArgs:
        chunksize (int): number of rows to stream at a time, None for the entire file

    Yields:
        pandas.DataFrame

    """

    if chunksize is None:
        return _read_seq_data_full(seqdata_filename, record_type, chromosome)
    else:
        return _read_seq_data_chunks(seqdata_filename, record_type, chromosome, chunksize)


def read_fragment_data(seqdata_filename, chromosome, chunksize=None):
    """ Read fragment data from a HDF seqdata file.

    Args:
        seqdata_filename (str): name of seqdata file
        chromosome (str): select specific chromosome, None for all chromosomes

    KwArgs:
        chunksize (int): number of rows to stream at a time, None for the entire file

    Yields:
        pandas.DataFrame

    Returned dataframe has columns 'fragment_id', 'start', 'end'

    """

    return read_seq_data(seqdata_filename, 'fragments', chromosome, chunksize=chunksize)


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


def read_filtered_fragment_data(seqdata_filename, chromosome, filter_duplicates=False, map_qual_threshold=1):
    """ Read filtered fragment data from a HDF seqdata file.

    Args:
        seqdata_filename (str): name of seqdata file
        chromosome (str): select specific chromosome, None for all chromosomes

    KwArgs:
        filter_duplicates (bool): filter reads marked as duplicate
        map_qual_threshold (int): filter reads with less than this mapping quality

    Yields:
        pandas.DataFrame

    Returned dataframe has columns 'fragment_id', 'start', 'end'

    """

    reads = remixt.seqdataio.read_fragment_data(seqdata_filename, chromosome)

    # Filter duplicates if necessary
    if filter_duplicates:
        reads = reads[reads['is_duplicate'] == 1]

    # Filter poor quality reads
    reads = reads[reads['mapping_quality'] >= map_qual_threshold]

    reads.drop(['is_duplicate', 'mapping_quality'], axis=1, inplace=True)

    return reads


