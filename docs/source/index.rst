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


* Get quickly started by browsing `Usage Principles <https://stereopy.readthedocs.io/en/latest/index.html>`_, `Tutorials <https://stereopy.readthedocs.io/en/latest/Tutorials/Cases.html>`_ or `API <https://stereopy.readthedocs.io/en/latest/content/03_API.html>`_.
* Open to discuss and provide feedback on `Github <https://github.com/STOmics/stereopy>`_.
* Follow changes in `Release Notes <https://stereopy.readthedocs.io/en/latest/content/06_Release_notes.html>`_.

News
--------------
The paper of Stereopy has been pre-printed on bioRxiv!

`Stereopy: modeling comparative and spatiotemporal cellular heterogeneity via multi-sample spatial transcriptomics <https://doi.org/10.1101/2023.12.04.569485>`_.


Upcoming functions
--------------------
* Batch Effect removal funciton
* Lasso expression matrix and image simultaneously
* ...


Highlights
------------

* More suitable for performing downstream analysis of Stereo-seq data.
* Support efficient reading and writing (IO), pre-processing, and standardization of multiple spatial transcriptomics data formats.
* Self-developed Gaussian smoothing model, tissue and cell segmentation algorithm models, and cell correction algorithm.
* Integrate various functions of dimensionality reduction, spatiotemporal clustering, cell clustering, spatial expression pattern analysis, etc.
* Develop interactive visualization functions based on features of Stereo-seq workflow.


Workflow
----------

.. image:: ./_static/Stereopy_workflow_v1.0.0.png
    :alt: Title figure
    :width: 700px
    :align: center

Latest Additions
------------------

Version 1.3.0
~~~~~~~~~~~~~~~~~~~
1.3.0 : 2024-05-31

Features:

1. Addition of `MSData.tl.st_gears <Tutorials(Multi-sample)/ST_Gears.html>`_ for spatial alignment of **Multi-sample**.
2. `High Resolution Matrix Export <Tutorials/High_Resolution_Export.html>`_ can support both **GEF** and **GEM** files.
3. Addition of parameters `min_count` and `max_count` for `st.tl.filter_genes <content/stereo.core.StPipeline.filter_genes.html>`_.
4. `MSData.integrate <content/stereo.core.ms_data.MSData.integrate.html>`_ can be compatible with sparse matrix when `MSData.var_type` is `union`.
5. Addition of `MSData.tl.set_scope_and_mode <content/stereo.core.ms_pipeline.MSDataPipeLine.set_scope_and_mode.html>`_ to set `scope` and `mode` globally on **Multi-sample** analysis.
6. Addition of `MSData.plt.ms_spatial_scatter <content/stereo.plots.PlotMsSpatialScatter.ms_spatial_scatter.html>`_ to plot spatial scatter plot for each **sample** in **Multi-sample** separately.

BUG Fixes:

1. Fixed the problem that `st.io.read_gem` is incompatible with **GEM** files containing **geneID**.
2. Fixed the bug of losing part of metadata when writing **StereoExpData** / **MSData** into **Stereo-h5ad** or **h5ms** file.
3. Fixed the incompatibility problem with **AnnData** when performing `st.tl.sctransform`.

Version 1.2.0
~~~~~~~~~~~~~~~~~~~
1.2.0 : 2024-03-30

Features:

1. `st.io.read_gem` and `st.io.read_gef` support expression matrix files with geneID information.
2. Analysis results of `find_marker_genes`  will be saved into the output AnnData h5ad.
3. Upgraded tissue segmentation algorithm.
4. Addition of `st.tl.adjusted_rand_score` to calculate the adjusted Rand coefficient between two clusters.
5. Addition of `st.tl.silhouette_score` to calculate the average silhouette coefficient of a cluster.
6. `h5ad2rds.R` is compatible with AnnData version > 0.7.5, to convert from h5ad to rds files.
7. Addition of the clustering category labels to the graph of `st.plt.paga_compare`.

BUG Fixes:

1. Fixed the error of high memory consumption when converting `X.raw` into AnnData.


.. toctree::
    :titlesonly:
    :maxdepth: 3
    :hidden: 

    content/00_Installation
    content/01_Usage_principles
    Tutorials(Multi-sample)/Multi_sample
    Tutorials/index
    content/03_API
    content/04_Community
    content/05_Contributing
    content/06_Release_notes
    content/07_References


.. |docs| image:: https://img.shields.io/static/v1?label=docs&message=stereopy&color=green
    :target: https://stereopy.readthedocs.io/en/latest/index.html
    :alt: docs

.. |stars| image:: https://img.shields.io/github/stars/STOmics/stereopy?logo=GitHub&color=yellow
    :target: https://github.com/STOmics/stereopy
    :alt: stars

.. |downloads| image:: https://static.pepy.tech/personalized-badge/stereopy?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads
    :target: https://pepy.tech/project/stereopy
    :alt: Downloads

.. |pypi| image:: https://img.shields.io/pypi/v/stereopy
    :target: https://pypi.org/project/stereopy/
    :alt: PyPI

