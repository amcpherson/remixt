import fnmatch
import os
import sys
import numpy
import tarfile
import urllib
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


external_dir = os.path.abspath('src/external')

blossom5_url = 'http://pub.ist.ac.at/~vnk/software/blossom5-v2.04.src.tar.gz'
blossom5_tar_gz = os.path.join(external_dir, 'blossom5-v2.04.src.tar.gz')
blossom5_dir = os.path.join(external_dir, 'blossom5-v2.04.src')
blossom5_bin = os.path.join(blossom5_dir, 'blossom5')

urllib.urlretrieve(blossom5_url, blossom5_tar_gz)

tar = tarfile.open(blossom5_tar_gz)
tar.extractall(path=external_dir)
tar.close()

bamtools_dir = os.path.join(external_dir, 'bamtools', 'src')
bamtools_api_dir = os.path.join(bamtools_dir, 'api')
bamtools_utils_dir = os.path.join(bamtools_dir, 'utils')

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

bamtools_sources = []
bamtools_sources += list(find_files(bamtools_api_dir, '*.cpp'))
bamtools_sources += list(find_files(bamtools_utils_dir, '*.cpp'))
bamtools_sources = filter(lambda a: not a.endswith('win_p.cpp'), bamtools_sources)

blossom_directory = 'src/external/blossom5-v2.04.src/'

blossom_source = [
    'PMduals.cpp',
    'PMexpand.cpp',
    'PMinit.cpp',
    'PMinterface.cpp',
    'PMmain.cpp',
    'PMrepair.cpp',
    'PMshrink.cpp',
    'MinCost/MinCost.cpp',
]

blossom_source = [os.path.join(blossom_directory, filename) for filename in blossom_source]

libraries = []
if 'linux' in sys.platform:
    libraries.append('rt')

extensions = [
    Extension(
        name='remixt.blossomv',
        sources=['remixt/blossomv.pyx'] + blossom_source,
        include_dirs=[blossom_directory],
        libraries=libraries,
    ),
    Extension(
        name='remixt.bamreader',
        sources=['remixt/bamreader.pyx', 'src/BamReader.cpp'] + bamtools_sources,
        include_dirs=['src', external_dir, bamtools_dir, numpy.get_include()],
        libraries=['z', 'bz2'],
    ),
    Extension(
        name='remixt.hmm',
        sources=['remixt/hmm.pyx'],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name='remixt',
    version='1.0',
    packages=find_packages(),
    package_data={'remixt': ['data/cn_proportions.tsv']},
    ext_modules=cythonize(extensions),
)
