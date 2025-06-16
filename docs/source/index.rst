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


|stars| |pypi| |downloads| |docs| |doi|


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

Version 1.6.1
~~~~~~~~~~~~~~~~~~~
1.6.1 : 2025-06-16

.. _st.io.stereo_to_anndata: content/stereo.io.stereo_to_anndata.html
.. |st.io.stereo_to_anndata| replace:: `st.io.stereo_to_anndata`

.. _st.plt.batches_umap: content/stereo.plots.PlotCollection.batches_umap.html
.. |st.plt.batches_umap| replace:: `st.plt.batches_umap`

Features:

1. |st.io.stereo_to_anndata|_ supports adding multiple images.
2. |st.plt.batches_umap|_ supports downloading.

BUG Fixes:

1. Pinned the version of **fastcluster** in dependencies to **1.2.6** for ensuring installation compatibility.
2. Fixed the problem that the **width** and **height** of the images added to **AnnData** by `st.io.stereo_to_anndata` were swapped.

Version 1.6.0
~~~~~~~~~~~~~~~~~~~
1.6.0 : 2025-02-21

.. _Spatialign: Tutorials(Multi-sample)/Spatial_Alignment.html
.. |Spatialign| replace:: **Spatialign**

Features:

1. Addition of new algorithm |Spatialign|_ for batch effect removal.

Version 1.5.1
~~~~~~~~~~~~~~~~~~~
1.5.1 : 2024-12-26

.. _st.io.stereo_to_anndata: content/stereo.io.stereo_to_anndata.html
.. |st.io.stereo_to_anndata| replace:: `st.io.stereo_to_anndata`

.. _h5ad2rds.R: Tutorials/Format_Conversion.html
.. |h5ad2rds.R| replace:: **h5ad2rds.R**

Features:

1. |st.io.stereo_to_anndata|_ supports adding image information into the converted **AnnData** object.
2. |h5ad2rds.R|_ supports adding image information into the converted **RDS** file.
3. Optimized the visualization of the plotting scale for spatial scatter plot when inputting small data.

BUG Fixes:

1. Fixed the problem that the layers was lost when converting **StereoExpData** to **AnnData** by using `st.io.stereo_to_anndata`.
2. Fixed the problem that the result of `st.tl.gen_ccc_micro_envs` cannot be reproduced.


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

.. |doi| image:: https://zenodo.org/badge/344680781.svg
    :target: https://doi.org/10.5281/zenodo.14722435
    :alt: DOI

