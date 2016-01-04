import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

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

extensions = [
    Extension('blossomv',
        sources=['remixt/blossomv.pyx'] + blossom_source,
        include_dirs=[blossom_directory],
    )
]

setup(
    name='remixt',
    version='1.0',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
)
