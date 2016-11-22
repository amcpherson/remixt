import fnmatch
import os
import sys
import numpy
import versioneer
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


external_dir = os.path.abspath('src/external')

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

libraries = []
if 'linux' in sys.platform:
    libraries.append('rt')

extensions = [
    Extension(
        name='remixt.bamreader',
        sources=['remixt/bamreader.pyx', 'src/BamReader.cpp'] + bamtools_sources,
        include_dirs=['src', external_dir, bamtools_dir, numpy.get_include()],
        libraries=['z', 'bz2'],
    ),
    Extension(
        name='remixt.model1',
        sources=['remixt/model1.pyx'],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        name='remixt.model1a',
        sources=['remixt/model1a.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-Wno-unused-function"],
    ),
    Extension(
        name='remixt.model2',
        sources=['remixt/model2.pyx'],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        name='remixt.model3',
        sources=['remixt/model3.pyx'],
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
    entry_points={'console_scripts': ['remixt = remixt.ui.main:main']},
)
