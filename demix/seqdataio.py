import contextlib
import tarfile
import numpy as np
import pandas as pd


@contextlib.contextmanager
def addtotar(tar, filename):
    data = io.BytesIO()
    yield data
    info = tarfile.TarInfo(name=filename)
    info.size = data.tell()
    data.seek(0)
    tar.addfile(tarinfo=info, fileobj=data)


read_data_dtype = np.dtype([('start', np.uint32), ('length', np.uint16)])


allele_data_dtype = np.dtype([('fragment_id', np.uint32), ('position', np.uint32), ('is_alt', np.uint8)])


def write_read_data(tar, data):
    """ Write read data to gzipped tar of chromosome files

    Args:
        tar (str): tar to which data will be written
        data (pandas.DataFrame): read data

    Input `data` dataframe has columns `chromosome`, `start`, `end`.

    """

    for chrom, chrom_data in data.groupby('chromosome'):

        raw_chrom_data = np.zeros(len(chrom_data.index), dtype=read_data_dtype)

        raw_chrom_data['start'] = chrom_data['start']
        raw_chrom_data['length'] = chrom_data['end'] - chrom_data['start']

        filename = 'reads.{0}'.format(chrom)

        with addtotar(tar, filename) as f:
            f.write(raw_chrom_data.tostring())


def write_allele_data(tar, data):
    """ Write allele data to gzipped tar of chromosome files

    Args:
        tar (str): tar to which data will be written
        data (pandas.DataFrame): allele data

    Input `data` dataframe has columns `chromosome`, `position`, `fragment_id`, `is_alt`.

    """

    for chrom, chrom_data in data.groupby('chromosome'):

        chrom_data = chrom_data.sort('fragment_id')

        raw_chrom_data = np.zeros(len(chrom_data.index), dtype=allele_data_dtype)

        raw_chrom_data['fragment_id'] = chrom_data['fragment_id']
        raw_chrom_data['position'] = chrom_data['position']
        raw_chrom_data['is_alt'] = chrom_data['is_alt']

        filename = 'alleles.{0}'.format(chrom)

        with addtotar(tar, filename) as f:
            f.write(raw_chrom_data.tostring())


def read_raw_read_data(reads_file, num_rows=None):
    """ Read raw read data and reformat

    Args:
        reads_file (file): file like object, must support read()

    KwArgs:
        num_rows (int): number of rows to stream at a time, None for the entire file

    Yields:
        pandas.DataFrame

    Returned dataframe has columns `start`, `end`

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

    Returned dataframe has columns `position`, `is_alt`, `fragment_id`

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

    Returned dataframe has columns `start`, `end`

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

    Returned dataframe has columns `position`, `is_alt`, `fragment_id`

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


def create_seqdata(seqdata_filename, reads_filenames, alleles_filenames):
    """ Create a seqdata tar object

    Args:
        seqdata_filename (str): path to output seqdata tar file
        reads_filenames (dict): individual seqdata read tables keyed by chromosome name
        alleles_filenames (dict): individual seqdata allele tables keyed by chromosome name

    """

    with tarfile.open(seqdata_filename, 'w') as output_tar:

        prefixes = ('reads.', 'alleles.')
        chrom_filenames = (reads_filenames, alleles_filenames)

        for prefix, (chrom, filename) in zip(prefixes, chrom_filenames.iteritems()):

            name = prefix+chrom
            tarinfo = tarfile.TarInfo(name=name)

            with open(filename, 'rb') as f:

                output_tar.addfile(tarinfo=tarinfo, fileobj=f)


