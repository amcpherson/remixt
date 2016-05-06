import fnmatch
import os
import sys
import numpy
import tarfile
import urllib
import versioneer
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


external_dir = os.path.abspath('src/external')

blossom5_url = 'http://pub.ist.ac.at/~vnk/software/blossom5-v2.04.src.tar.gz'
blossom5_tar_gz = os.path.join(external_dir, 'blossom5-v2.04.src.tar.gz')
blossom5_dir = os.path.join(external_dir, 'blossom5-v2.04.src')
blossom5_bin = os.path.join(blossom5_dir, 'blossom5')

if not os.path.exists(blossom5_tar_gz):
    urllib.urlretrieve(blossom5_url, blossom5_tar_gz)

if not os.path.exists(os.path.join(blossom5_dir, 'README.TXT')):
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
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='ReMixT is a tool for joint inference of clone specific segment and breakpoint copy number',
    author='Andrew McPherson',
    author_email='andrew.mcpherson@gmail.com',
    url='http://bitbucket.org/dranew/remixt',
    download_url='https://bitbucket.org/dranew/remixt/get/v{}.tar.gz'.format(versioneer.get_version()),
    keywords=['scientific', 'sequence analysis', 'cancer'],
    classifiers=[],
    ext_modules=cythonize(extensions),
    scripts=[
        'remixt/run/remixt_calc_bias.py',
        'remixt/run/remixt_extract_seqdata.py',
        'remixt/run/remixt_fit_model.py',
        'remixt/run/remixt_infer_haps.py',
        'remixt/run/remixt_prepare_counts.py',
        'remixt/run/remixt_run.py',
        'remixt/setup/remixt_create_ref_data.py',
        'remixt/setup/remixt_mappability_bwa.py',
    ],
)
