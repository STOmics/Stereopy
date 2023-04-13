.. Stereopy manual documentation master file, created by
   sphinx-quickstart on Mon Nov 21 18:07:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. =====================
.. Document Title
.. =====================

.. First level
.. -----------

.. Second level
.. ++++++++++++

.. Third level
.. ************

.. Fourth level
.. ~~~~~~~~~~~~


|stars| |pypi| |downloads| |docs| 


Stereopy -  Spatial Transcriptomics Analysis in Python
========================================================

**Stereopy** is a fundamental and comprehensive tool for mining and visualization \
based on spatial transcriptomics data, such as Stereo-seq (spatial enhanced resolution omics sequencing) data. \
More analysis will be added here, either from other popular tools or developed by ourselves, to meet diverse requirements. \
Meanwhile, we are still working on the improvement of performance and calculation efficiency.


* Get quickly started by browsing `Usage Principles <https://stereopy.readthedocs.io/en/latest/index.html>`_, `Tutorials <https://stereopy.readthedocs.io/en/latest/Tutorials/Examples.html>`_ or `API <https://stereopy.readthedocs.io/en/latest/api/index.html>`_.
* Open to discuss and provide feedback on `Github <https://github.com/BGIResearch/stereopy>`_.
* Follow changes in `Release Notes <https://stereopy.readthedocs.io/en/latest/release_note.html>`_.


Highlights
-----------

* More suitable for performing downstream analysis of Stereo-seq data.
* Support efficient reading and writing (IO), pre-processing, and standardization of multiple spatial transcriptomics data formats.
* Self-developed Gaussian smoothing model, tissue and cell segmentation algorithm models, and cell correction algorithm.
* Integrate various functions of dimensionality reduction, spatiotemporal clustering, cell clustering, spatial expression pattern analysis, etc.
* Develop interactive visualization functions based on features of Stereo-seq workflow.


Workflow
---------

.. image:: ./_static/Stereopy_workflow_v0.11.0.png
    :alt: Title figure
    :width: 700px
    :align: center

Latest Additions
------------------

Version 0.11.0
~~~~~~~~~~~~~~
0.11.0 : 2022-04-04

1. Addition of Cell-cell Communication analysis;
2. Addition of Gene Regulatory Network analysis;
3. Addition of SingleR function for automatic annotation;
4. Addition of `v2` algorithm fast cell correction;
5. Addition of dot plot to display gene-level results;
6. Addition of the sorting function and the limitation of output genes in `data.tl.find_marker_genes`;
7. Added `pct` and `pct_rest` to the output files of marker genes;
8. Addition of the parameter `mean_uni_gt` in `data.tl.filter_genes` to filter genes on average expression;
9. Fixed the bug that `adata.X` to output AnnData was the raw matrix;
10. Fixed the failed compatibility to analysis results from `.h5ad` (version <= 0.9.0);
11. Updated the tissue segmentation algorithm in the module of cell segmentation to avoid the lack of tissue;
12. Reconstructed the manual of Stereopy.
13. Updated requirements.txt.

Version 0.10.0
~~~~~~~~~~~~~~
0.10.0 : 2022-02-22

1. Supported installation on Windows.
2. Addition of displaying basic information of StereoExpData object when simply typing it.
3. Addition of saving statistic results when plotting.
4. Addition of marker gene proportion (optional), in-group and out-of-group, in `data.tl.find_marker_genes`. Otherwise, supported filtering marker genes via `data.tl.filter_marker_genes`.
5. Supported adapting to AnnData, use directly use data and results stored in AnnData for subsequent analysis.
6. Addition of the matrix of gene count among clusters so that transformed output `.rds` file could be used for annotation by SingleR directly.
7. Initial release of Stereopy development solution.
8. Updated requirements.txt.




.. toctree::
    :titlesonly:
    :maxdepth: 2
    :hidden: 

    content/00_Installation
    content/01_Usage_principles
    Tutorials/Cases
    content/03_API
    content/04_Community
    content/05_Contributing
    content/06_Release_notes
    content/07_References


.. |docs| image:: https://img.shields.io/static/v1?label=docs&message=stereopy&color=green
    :target: https://stereopy.readthedocs.io/en/latest/index.html
    :alt: docs

.. |stars| image:: https://img.shields.io/github/stars/BGIResearch/stereopy?logo=GitHub&color=yellow
    :target: https://github.com/BGIResearch/stereopy
    :alt: stars

.. |downloads| image:: https://static.pepy.tech/personalized-badge/stereopy?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads
    :target: https://pepy.tech/project/stereopy
    :alt: Downloads

.. |pypi| image:: https://img.shields.io/pypi/v/stereopy
    :target: https://pypi.org/project/stereopy/
    :alt: PyPI

