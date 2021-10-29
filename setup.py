#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:setup.py
@time:2021/03/02
"""
from setuptools import setup, find_packages
import sys
from pathlib import Path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from numpy import get_include

if sys.version_info < (3, 7):
    sys.exit('stereopy requires Python >= 3.7')

ext_modules = cythonize({
    Extension("gef_cy",
              ["stereo/io/gef_cy.pyx", "stereo/io/H5Reader.cpp"],
              include_dirs=[get_include(), '/usr/include',
                            '/usr/local/include'],
              libraries=["hdf5"],
              language='c++',
              extra_compile_args=["-std=c++11", "-Wno-sign-compare"]
              )
})

setup(
    name='stereopy',
    version='0.2.0',
    setup_requires=['setuptools_scm', 'numpy', 'panel', 'pytest', 'quilt3', 'scipy', 'phenograph'],
    description='Spatial transcriptomic analysis in python.',
    long_description=Path('README.md').read_text('utf-8'),
    long_description_content_type="text/markdown",
    url='https://github.com/BGIResearch/stereopy',
    author='BGIResearch',
    author_email='qiuping1@genomics.cn',
    python_requires='>=3.7',
    install_requires=[
        l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    extras_require=dict(
        visualization=['bokeh>=1.4.0'],
        doc=['sphinx>=3.2'],
        test=['pytest>=4.4', 'pytest-nunit'],
    ),
    packages=find_packages(),
    include_package_data=True,
    ext_modules=ext_modules,
    classifiers=[
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)
