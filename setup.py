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

if sys.version_info < (3, 8):
    sys.exit('stereopy requires Python >= 3.8')

setup(
    name='stereopy',
    version='0.9.0',
    setup_requires=['setuptools_scm', 'numpy', 'panel', 'pytest', 'quilt3', 'scipy', 'phenograph'],
    description='Spatial transcriptomic analysis in python.',
    long_description=Path('README.md').read_text('utf-8'),
    long_description_content_type="text/markdown",
    url='https://github.com/BGIResearch/stereopy',
    author='BGIResearch',
    author_email='qiuping1@genomics.cn',
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
    classifiers=[
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)
