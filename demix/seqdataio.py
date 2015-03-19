import os
import contextlib
import tarfile
import gzip
import StringIO
import numpy as np
import pandas as pd


read_data_dtype = np.dtype([('start', np.uint32), ('length', np.uint16)])


allele_data_dtype = np.dtype([('fragment_id', np.uint32), ('position', np.uint32), ('is_alt', np.uint8)])


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


def write_allele_data(alleles_file, allele_data):
    """ Write allele data for a specific chromosome to a file

    Args:
        alleles_file (str): tar to which data will be written
        allele_data (pandas.DataFrame): allele data

    Input 'allele_data' dataframe has columns 'position', 'fragment_id', 'is_alt'.

    """

    raw_data = np.zeros(len(allele_data.index), dtype=allele_data_dtype)

    raw_data['fragment_id'] = allele_data['fragment_id']
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

    def write(self, read_data, allele_data):
        """ Write a chunk of reads and alleles data

        Args:
            read_data (pandas.DataFrame): read data
            allele_data (pandas.DataFrame): allele data

        Input 'read_data' dataframe has columns 'chromosome', 'start', 'end'.
        Input 'allele_data' dataframe has columns 'chromosome', 'position', 'fragment_id', 'is_alt'.

        """

        for chromosome in read_data['chromosome'].unique():

            if chromosome not in self.fragment_id_offset:

                with open(self.get_reads_filename(chromosome), 'w'):
                    pass

                with open(self.get_alleles_filename(chromosome), 'w'):
                    pass

                self.reads_filenames[chromosome] = self.get_reads_filename(chromosome)
                self.alleles_filenames[chromosome] = self.get_alleles_filename(chromosome)

                self.fragment_id_offset[chromosome] = 0

            chrom_read_data = read_data[read_data['chromosome'] == chromosome]
            chrom_allele_data = allele_data[allele_data['chromosome'] == chromosome].copy()

            # Remap fragment ids
            chrom_allele_data['fragment_id'] += self.fragment_id_offset[chromosome]

            with open(self.get_reads_filename(chromosome), 'ab') as f:
                write_read_data(f, chrom_read_data)

            with open(self.get_alleles_filename(chromosome), 'ab') as f:
                write_allele_data(f, chrom_allele_data)

            self.fragment_id_offset[chromosome] += len(chrom_read_data.index)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        create_seqdata(self.seqdata_filename, self.reads_filenames, self.alleles_filenames)


def read_raw_read_data(reads_file, num_rows=None):
    """ Read raw read data and reformat

    Args:
        reads_file (file): file like object, must support read()

    KwArgs:
        num_rows (int): number of rows to stream at a time, None for the entire file

    Yields:
        pandas.DataFrame

    Returned dataframe has columns 'start', 'end'

    """

    while True:

        if num_rows is not None and num_rows > 0:
            raw_data = reads_file.read(num_rows * read_data_dtype.itemsize)
        else:
            raw_data = reads_file.read()

        if raw_data == '':
            yield pd.DataFrame(columns=['start', 'end'])
            break

        data = np.fromstring(raw_data, dtype=read_data_dtype)

        df = pd.DataFrame(data)
        df['end'] = df['start'] + df['length'] - 1
        df = df.drop('length', axis=1)

        yield df


def read_raw_allele_data(alleles_file, num_rows=None):
    """ Read raw allele data and reformat

    Args:
        alleles_file (file): file like object, must support read()

    KwArgs:
        num_rows (int): number of rows to stream at a time, None for the entire file

    Yields:
        pandas.DataFrame

    Returned dataframe has columns 'position', 'is_alt', 'fragment_id'

    """

    while True:

        if num_rows is not None and num_rows > 0:
            raw_data = alleles_file.read(num_rows * allele_data_dtype.itemsize)
        else:
            raw_data = alleles_file.read()

        if raw_data == '':
            yield pd.DataFrame(columns=['position', 'is_alt', 'fragment_id'])
            break

        data = np.fromstring(raw_data, dtype=allele_data_dtype)

        df = pd.DataFrame(data)

        yield df


def read_seq_data(seqdata_filename, record_type, chromosome=None, num_rows=None):
    """ Read sequence data from a tar archive

    Args:
        seqdata_filename (str): name of seqdata tar file
        record_type (str): record type, can be 'alleles' or 'reads'

    KwArgs:
        chromosome (str): select specific chromosome, None for all chromosomes
        num_rows (int): number of rows to stream at a time, None for the entire file

    Yields:
        pandas.DataFrame

    """

    with tarfile.open(seqdata_filename, 'r:gz') as tar:
        
        for tarinfo in tar:

            rectype, chrom = tarinfo.name.split('.')

            if chromosome is not None and chromosome != chrom:
                continue

            if record_type != rectype:
                continue

            if rectype == 'reads':
                for data in read_raw_read_data(tar.extractfile(tarinfo), num_rows=num_rows):
                    yield data

            elif rectype == 'alleles':
                for data in read_raw_allele_data(tar.extractfile(tarinfo), num_rows=num_rows):
                    yield data


def read_read_data(seqdata_filename, chromosome=None, num_rows=None):
    """ Read read data from gzipped tar of chromosome files

    Args:
        seqdata_filename (str): name of seqdata tar file

    KwArgs:
        chromosome (str): select specific chromosome, None for all chromosomes
        num_rows (int): number of rows to stream at a time, None for the entire file

    Yields:
        pandas.DataFrame

    Returned dataframe has columns 'start', 'end'

    """

    return read_seq_data(seqdata_filename, 'reads', chromosome=chromosome, num_rows=num_rows)


def read_allele_data(seqdata_filename, chromosome=None, num_rows=None):
    """ Read allele data from gzipped tar of chromosome files

    Args:
        seqdata_filename (str): name of seqdata tar file

    KwArgs:
        chromosome (str): select specific chromosome, None for all chromosomes
        num_rows (int): number of rows to stream at a time, None for the entire file

    Yields:
        pandas.DataFrame

    Returned dataframe has columns 'position', 'is_alt', 'fragment_id'

    """

    return read_seq_data(seqdata_filename, 'alleles', chromosome=chromosome, num_rows=num_rows)


def read_chromosomes(seqdata_filename):
    """ Read chromosomes in sequence data tar

    Args:
        seqdata_filename (str): name of seqdata tar file

    Returns:
        list of chromsomes

    """

    with tarfile.open(seqdata_filename, 'r:gz') as tar:

        chromosomes = set()
        for tarinfo in tar:
            chromosomes.add(tarinfo.name.split('.')[1])

        return chromosomes


def create_seqdata(seqdata_filename, reads_filenames, alleles_filenames, gzipped=False):
    """ Create a seqdata tar object

    Args:
        seqdata_filename (str): path to output seqdata tar file
        reads_filenames (dict): individual seqdata read tables keyed by chromosome name
        alleles_filenames (dict): individual seqdata allele tables keyed by chromosome name
        gzipped (bool): the individual files are gzipped

    """

    with tarfile.open(seqdata_filename, 'w:gz') as tar:

        prefixes = ('reads.', 'alleles.')
        filenames = (reads_filenames, alleles_filenames)

        for prefix, chrom_filenames in zip(prefixes, filenames):

            for chromosome, filename in chrom_filenames.iteritems():

                name = prefix+chromosome

                if gzipped:

                    with gzip.open(filename, 'rb') as f:
                        data = StringIO.StringIO(f.read())
                    
                    tarinfo = tarfile.TarInfo(name=name)
                    tarinfo.size = len(data.buf)
                    tar.addfile(tarinfo=tarinfo, fileobj=data)                   

                else:

                    tar.add(filename, arcname=name)


