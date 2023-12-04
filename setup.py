#!/usr/bin/env python3
# coding: utf-8
"""
@author: Junhao Xu  xujunhao@genomics.cn
@last modified by: Junhao Xu
@file:setup.py
@time:2023/04/28
"""
import os
import sys

from pathlib import Path
from setuptools import setup
from setuptools import find_packages

if sys.version_info < (3, 8):
    sys.exit('stereopy requires Python >= 3.8')


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("version"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name='stereopy',
    version=get_version("stereo/common.py"),
    setup_requires=['setuptools_scm', 'numpy==1.21.6', 'panel', 'pytest', 'quilt3', 'scipy', 'phenograph'],
    description='Spatial transcriptomic analysis in python.',
    long_description=Path('README.md').read_text('utf-8'),
    long_description_content_type="text/markdown",
    url='https://github.com/STOmics/Stereopy',
    author='STOmics',
    author_email='tanliwei@stomics.tech',
    license='MIT License',
    python_requires='>=3.8,<3.9',
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
    entry_points={
        'console_scripts':[
            'ccd = stereo.scripts.ccd:main'
        ]
    },
    classifiers=[
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)
