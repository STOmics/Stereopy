#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
print(HERE)
print(HERE.parent.parent)
print(sys.path)
sys.path[:0] = [str(HERE.parent), str(HERE / 'extensions')]
sys.path.append(HERE.parent.parent)
import stereo

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# -- General configuration ------------------------------------------------


nitpicky = True  # Warn about broken links. This is here for a reason: Do not change.
needs_sphinx = '2.0'  # Nicer param docs
suppress_warnings = ['ref.citation']

# General information
project = 'stereopy'
copyright = '2021, Ping Qiu'
author = 'Ping Qiu'
version = 'v0.1'
release = 'v0.1'

# default settings
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
default_role = 'literal'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    # 'plot_generator',
    # 'matplotlib.sphinxext.plot_directive',
    'sphinx_autodoc_typehints',  # needs to be after napoleon
    # 'ipython_directive',
    # 'ipython_console_highlighting',
    'nbsphinx',
    # 'scanpydoc',
    *[p.stem for p in (HERE / 'extensions').glob('*.py')],
]

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_default_flags = ['members']
todo_include_todos = False
api_dir = HERE / 'api'  # function_images
# print(api_dir)
# The master toctree document.
language = None

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']



