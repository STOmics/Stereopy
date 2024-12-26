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

Version 1.5.0
~~~~~~~~~~~~~~~~~~~
1.5.0 : 2024-11-08

.. _SpaTrack: Tutorials/SpaTrack.html
.. |SpaTrack| replace:: **SpaTrack**

.. _Layer: content/stereo.core.StPipeline.set_layer.html
.. |Layer| replace:: **Layer**

.. _st.tl.cal_qc: content/stereo.core.StPipeline.cal_qc.html
.. |st.tl.cal_qc| replace:: `st.tl.cal_qc`

.. _st.tl.filter_cells: content/stereo.core.StPipeline.filter_cells.html
.. |st.tl.filter_cells| replace:: `st.tl.filter_cells`

.. _st.tl.filter_genes: content/stereo.core.StPipeline.filter_genes.html
.. |st.tl.filter_genes| replace:: `st.tl.filter_genes`

.. _st.tl.log1p: content/stereo.core.StPipeline.log1p.html
.. |st.tl.log1p| replace:: `st.tl.log1p`

.. _st.tl.normalize_total: content/stereo.core.StPipeline.normalize_total.html
.. |st.tl.normalize_total| replace:: `st.tl.normalize_total`

.. _st.tl.scale: content/stereo.core.StPipeline.scale.html
.. |st.tl.scale| replace:: `st.tl.scale`

.. _st.tl.quantile: content/stereo.core.StPipeline.quantile.html
.. |st.tl.quantile| replace:: `st.tl.quantile`

.. _st.tl.disksmooth_zscore: content/stereo.core.StPipeline.disksmooth_zscore.html
.. |st.tl.disksmooth_zscore| replace:: `st.tl.disksmooth_zscore`

.. _st.tl.sctransform: content/stereo.core.StPipeline.sctransform.html
.. |st.tl.sctransform| replace:: `st.tl.sctransform`

.. _st.tl.highly_variable_genes: content/stereo.core.StPipeline.highly_variable_genes.html
.. |st.tl.highly_variable_genes| replace:: `st.tl.highly_variable_genes`

.. _st.tl.pca: content/stereo.core.StPipeline.pca.html
.. |st.tl.pca| replace:: `st.tl.pca`

.. _st.tl.find_marker_genes: content/stereo.core.StPipeline.find_marker_genes.html
.. |st.tl.find_marker_genes| replace:: `st.tl.find_marker_genes`

.. _st.plt.spatial_scatter: content/stereo.plots.PlotCollection.spatial_scatter.html
.. |st.plt.spatial_scatter| replace:: `st.plt.spatial_scatter`

Features:

1. Addition of new algorithm |SpaTrack|_ for trajectory inference.
2. Addition of |Layer|_ for saving expression matrices at different analysis stages, the functions that can use expression matrices in **Layer** as following:
            * |st.tl.cal_qc|_
            * |st.tl.filter_cells|_
            * |st.tl.filter_genes|_
            * |st.tl.log1p|_
            * |st.tl.normalize_total|_
            * |st.tl.scale|_
            * |st.tl.quantile|_
            * |st.tl.disksmooth_zscore|_
            * |st.tl.sctransform|_
            * |st.tl.highly_variable_genes|_
            * |st.tl.pca|_
            * |st.tl.find_marker_genes|_
3. Merger of multiple samples can merge some analysis result in every single samples when data type is **StereoExpData**.
4. |st.plt.spatial_scatter|_ supports setting **regist.tif** as background to display simultaneously on the plot.

BUG Fixes:

1. Fixed the problem that the proportion of chondriogenes was calculated incorrectly when input data contains **geneID**.
2. Fixed the problem that saving **MSData** into **h5mu** was failed after running `st.tl.highly_variable_genes`.


Version 1.4.0
~~~~~~~~~~~~~~~~~~~
1.4.0 : 2024-09-05

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

1. Fixed the problem that **CellCorrection** is incompatible with small-size images (less than 2000px in any dimension) when using the **EDM** method.
2. Fixed the problem that `MSData.to_integrate` is incompatible when the number of cells in the integrated sample is less than the total number of cells in all single samples.
3. Fixed the problem that `st.plt.time_series_tree_plot` can not capture the result of **PAGA**, leading to an incorrect plot.
4. Fixed other bugs.

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

