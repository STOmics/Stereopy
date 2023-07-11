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

About MSData
-----------
For multi-sample data analysis, we have recently launched a simple-to-use method that can distinguish single-sample data \
and multi-sample one while work on the parallel processings, but the The results of the two parties can be interacted.

In order to adapt to the new parameters and concepts in MSData analysis, the current version is a Beta one, which means there are \
inevitably a handful of BUGs. We sincerely hope to receive your feedback and suggestions for MSData.


Upcoming functions
----------
* Cell Co-occurrence
* Cell Community
* New algorithm for Batch Effect Correction


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

Version 0.13.0b1
~~~~~~~~~~~~~~
0.13.0b1 : 2023-07-11

Notice: this Beta version is specifically developed for multi-sample analysis. Major update points are listed below.

1. Addition of 3D Cell-cell Communication.
2. Addition of 3D Gene Regulatory Network.
3. Addition of Trajectory Inference, including PAGA and DPT algorithms.
4. Addition of Batch QC function for evaluation on batch effect.
5. Addition of `st.io.read_h5ad` for improved compatibility with AnnData H5ad, we highly recommend that instead of `st.io.read_ann_h5ad`.
6. Addition of analysis workflow tutorial based on multi-sample data, with assistant parameters scopeand mode.
7. Addition of resetting the image order of multi-sample analysis results.
8. Addition of 3D mesh visualization.
9. Improved the performance of Gaussian Smoothing.

Version 0.12.1
~~~~~~~~~~~~~~
0.12.1 : 2023-06-21

1. Addition of the pretreatment of calculating quality control metrics at the start of `st.tl.filter_genes` and `st.tl.filter_cells`.
2. Fixed the bug that loaded data from GEF file had the same expression matrix but in different row order, through updating gefpy package to v0.6.24.
3. Fixed the bug that `scale.data` had `np.nan` value in `st.tl.sctransform` , caused by data type limitation.
4. Fixed the bug that dot symbol ( '.' ) caused identification error of cluster name in `.csv` output, when doing `st.tl.find_marker_genes`.

Version 0.12.0
~~~~~~~~~~~~~~
0.12.0 : 2023-04-27

1. Addition of the algorithm of Cell Segmentation V3.0.
2. Addition of `method='hotspot'` to `st.tl.regulatory_network_inference`, which takes spatial coordinate information into account to calculate the relative importance between TFs and their target genes.
3. Addition of dpi and width/height setting for visualization, and addition of plotting scale for displaying static plot.
4. Optimized required memory while plotting UMAP embedding via `data.plt.umap` and cell distribution via `data.plt.cells_plotting`.
5. Fixed bug that input parameter of `var_features_n` was invalid, in `data.tl.scTransform`.
6. Updated requirements.txt.



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

.. |stars| image:: https://img.shields.io/github/stars/STOmics/stereopy?logo=GitHub&color=yellow
    :target: https://github.com/STOmics/stereopy
    :alt: stars

.. |downloads| image:: https://static.pepy.tech/personalized-badge/stereopy?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads
    :target: https://pepy.tech/project/stereopy
    :alt: Downloads

.. |pypi| image:: https://img.shields.io/pypi/v/stereopy
    :target: https://pypi.org/project/stereopy/
    :alt: PyPI

