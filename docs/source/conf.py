#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE / 'extensions')]

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# -- General configuration ------------------------------------------------


nitpicky = True  # Warn about broken links. This is here for a reason: Do not change.
needs_sphinx = '2.0'  # Nicer param docs
suppress_warnings = ['ref.citation']

# General information
project = 'stereopy'
copyright = '2023, BGI'
author = 'BGI'
version = 'v0.10.0'
release = 'v0.10.0'

# default settings
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
default_role = 'literal'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
#pygments_style = 'sphinx'

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'nbsphinx',
    "sphinx.ext.viewcode",
    "sphinx_gallery.load_style",
    "sphinx_autodoc_typehints",
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

#If you want to include a notebook without outputs and yet don’t want nbsphinx to execute it for you, you can explicitly disable this feature.
nbsphinx_execute = 'never'

html_theme = 'sphinx_book_theme'

html_title = "Stereopy"

html_theme_options = {
    # "sidebar_hide_name": True,
        "show_navbar_depth":0,
        "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
        "code-font-size": "var(--font-size--small)",
    },
}

html_static_path = ['_static']

html_css_files = [
    'html.css',
]
# html_css_files = [
#     '_static/html.css',
# ]

import nbclean, glob

for filename in glob.glob('**/*.ipynb', recursive=True):
    ntbk = nbclean.NotebookCleaner(filename)
    ntbk.clear('stderr')
    ntbk.save(filename)

nbsphinx_thumbnails = {
    "Tutorials/clustering": "_static/squareBin_clustering.png",
    "Tutorials/cell_clustering": "_static/cellBin_clustering.png",
    "Tutorials/clustering_by_gpu": "_static/GPU.png",
    "Tutorials/IO": "_static/io.png",
    "Tutorials/FormatConversion": "_static/conversion.png",
    "Tutorials/prepare": "_static/prepration.png",
    "Tutorials/performance": "_static/performance.png",
    "Tutorials/cell_correction": "_static/cell_bin_correction.png",
    "Tutorials/tissue_cut": "_static/tissue_segmentation.png",
    "Tutorials/gaussian_smooth": "_static/gaussian_smooth_1.png",
    "Tutorials/batches_integration": "_static/batches_integration.png",
    "Tutorials/rna_velocity": "_static/rna_velocity.png",
    "Tutorials/hotspot": "_static/hotspot.png",
    "Tutorials/Gaussian_Smoothing": "_static/gaussian_smooth_1.png",
    "Tutorials/high_resolution_export": "_static/box_select.gif",
    "Tutorials/sctransform":"_static/variance_sct.png"
}

