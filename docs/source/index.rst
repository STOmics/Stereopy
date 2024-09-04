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

Version 1.4.0
~~~~~~~~~~~~~~~~~~~
1.4.0 : 2024-09-06

.. _SpaSEG: Tutorials(Multi-sample)/SpaSEG.html
.. |SpaSEG| replace:: **SpaSEG**

.. _st.plt.cells_plotting: content/stereo.plots.PlotCollection.cells_plotting.html
.. |st.plt.cells_plotting| replace:: `st.plt.cells_plotting`

.. _st.io.write_h5mu: content/stereo.io.write_h5mu.html
.. |st.io.write_h5mu| replace:: `st.io.write_h5mu`

.. _st.io.mudata_to_msdata: content/stereo.io.mudata_to_msdata.html
.. |st.io.mudata_to_msdata| replace:: `st.io.mudata_to_msdata`

Features:

1. Addition of new algorithm |SpaSEG|_ for multiple **SRT** analysis.
2. Addition of **colorbar** or **legend** for `st.plt.cells_plotting`.
3. |st.plt.cells_plotting|_ supports exporting plots as **PNG**, **SVG** or **PDF**.
4. Addition of method |st.io.write_h5mu|_ and |st.io.mudata_to_msdata|_ for conversion between **MSData** and **MuData**.

BUG Fixes:

1. Fixed the problem that **CellCorrection** is incompatible with small-size images (less than 2000px in any dimension) when using the method **EDM**.
2. Fixed the problem that `MSData.to_integrate` is incompatible when the number of cells in the integrated sample is less than the total number of cells in all single samples.
3. Fixed the problem that `st.plt.time_series_tree_plot` can not capture the result of **PAGA**, leading to an incorrect plot.
4. Fixed other bugs.

Version 1.3.1
~~~~~~~~~~~~~~~~~~~
1.3.1 : 2024-06-28

Features:

1. Addition of new method **'adaptive'** for `st.tl.get_niche <content/stereo.algorithm.get_niche.GetNiche.main.html>`_ (the original method is named **'fixed'**).
2. Changed some parameter names of `st.tl.filter_cells <content/stereo.core.StPipeline.filter_cells.html>`_ and `st.tl.filter_genes <content/stereo.core.StPipeline.filter_genes.html>`_ for eliminating ambiguity(old parameter names are still compatible).
3. Filter the results of **PCA** and **UMAP** simultaneously when running `st.tl.filter_cells`.

BUG Fixes:

1. Fixed the problem that `ms_data.to_isolated` is incompatible with that there are duplicate **cell names** in different samples.
2. Fixed the problem that `st.io.read_gef` is incompatible with those **GEF** files that contain **gene names** ending with **'_{number}'** (like **'ABC_123'**).
3. Upgraded **gefpy** to latest for fixing the error that **gene names** are lost after running **CellCorrection**.

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

