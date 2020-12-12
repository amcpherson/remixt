import fnmatch
import os
import sys
import versioneer
from setuptools import setup, find_packages, Extension

NUMPY_VERSION = '1.19.4'

class get_numpy_include(str):
    def __str__(self):
        import numpy
        if numpy.__version__ != NUMPY_VERSION:
            raise Exception(f'numpy compatibility error, {NUMPY_VERSION} != {numpy.__version__}')
        return numpy.get_include()

external_dir = 'src/external'

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
bamtools_sources = list(filter(lambda a: not a.endswith('win_p.cpp'), bamtools_sources))

libraries = []
extra_compile_args = ['-g']
extra_link_args = ['-g']
if 'linux' in sys.platform:
    libraries.append('rt')
elif sys.platform == 'darwin':
    extra_compile_args.extend(['-stdlib=libc++'])
    extra_link_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.9'])

extensions = [
    Extension(
        name='remixt.bamreader',
        sources=['remixt/bamreader.pyx', 'src/BamAlleleReader.cpp'] + bamtools_sources,
        include_dirs=['src', external_dir, bamtools_dir, get_numpy_include()],
        libraries=['z', 'bz2'],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        name='remixt.bpmodel',
        sources=['remixt/bpmodel.pyx'],
        include_dirs=[get_numpy_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
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
    ext_modules=extensions,
    setup_requires=[
        f'numpy=={NUMPY_VERSION}',
        'setuptools>=18.0',
        'cython',
    ],
    install_requires=[
        f'numpy=={NUMPY_VERSION}',
        'scipy',
        'pandas',
        'tables',
        'pypeliner',
        'statsmodels',
        'scikit-learn',
        'pyyaml',
        'matplotlib',
        'seaborn',
        'bokeh',
    ],
    entry_points={'console_scripts': ['remixt = remixt.ui.main:main']},
)
