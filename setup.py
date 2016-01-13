import fnmatch
import os
import sys
import subprocess
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

src_directory = 'src/'

bamreader_source = [
    'Common.cpp',
    'DebugCheck.cpp',
    'BamReader.cpp',
]

bamreader_source = [os.path.join(src_directory, filename) for filename in bamreader_source]

external_dir = os.path.abspath('src/external')

boost_version = '1.55.0'
boost_basename = 'boost_1_55_0'

boost_url = 'http://downloads.sourceforge.net/project/boost/boost/' + boost_version + '/' + boost_basename + '.tar.gz'
boost_tgz_filename = boost_basename + '.tar.gz'
boost_dir = os.path.join(external_dir, boost_basename)
boost_sentinal = os.path.join(boost_dir, 'sentinal')

if not os.path.exists(boost_sentinal):
    subprocess.check_call('wget -c ' + boost_url, shell=True)
    subprocess.check_call('tar -C ' + external_dir + ' -xzvf ' + boost_tgz_filename, shell=True)
    os.remove(boost_tgz_filename)
    with open(boost_sentinal, 'w'):
        pass

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

boost_iostreams_sources = list(find_files(os.path.join(boost_dir, 'libs/iostreams/src'), '*.cpp'))
boost_serialization_sources = list(find_files(os.path.join(boost_dir, 'libs/serialization/src'), '*.cpp'))

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
        sources=['remixt/bamreader.pyx'] + bamreader_source + bamtools_sources + boost_iostreams_sources + boost_serialization_sources,
        include_dirs=[src_directory, external_dir, bamtools_dir, boost_dir],
        libraries=['z', 'bz2'],
    ),
]

setup(
    name='remixt',
    version='1.0',
    packages=find_packages(),
    package_data={'remixt': ['data/cn_proportions.tsv']},
    ext_modules=cythonize(extensions),
)
