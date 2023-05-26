.. module:: stereo
.. automodule:: stereo
   :noindex:

API
===

Import stereopy as::

   import stereo as st


IO
++++++++++++++++++++

The IO module includes input and output parts. 

The input part supports reading files in multiple formats to generate StereoExpData object, 
such as GEM, GEF, h5ad(from Scanpy and Seurat) and so on. In addition, we provide the conversion 
function between StereoExpData and AnnData so that swithing between tools would be smoother. 
In output part, we support writing StereoExpData as h5ad file for storage.


    StereoExpData
    --------------------

    The core data object is designed for expression matrix of spatial omics, which can be set 
    corresponding properties directly to initialize the data. 

    .. autosummary::
       :toctree: .

        core.StereoExpData


.. autosummary::
   :toctree: .

    io.read_gef_info
    io.read_gem
    io.read_gef
    io.read_ann_h5ad
    io.read_stereo_h5ad
    io.read_seurat_h5ad
    io.anndata_to_stereo
    io.stereo_to_anndata
    io.write_h5ad
    io.write_mid_gef
    io.update_gef


StPipeline: `tl`
++++++++++++++++++++

A class for basic analysis of StereoExpData. It is the `StereoExpData.tl`.

.. autosummary::
   :toctree: .

    core.StPipeline
    core.StPipeline.cal_qc
    core.StPipeline.filter_cells
    core.StPipeline.filter_genes
    core.StPipeline.raw_checkpoint
    core.StPipeline.sctransform
    core.StPipeline.normalize_total
    core.StPipeline.log1p
    core.StPipeline.scale
    core.StPipeline.highly_variable_genes
    core.StPipeline.pca
    core.StPipeline.neighbors
    core.StPipeline.umap
    core.StPipeline.leiden
    core.StPipeline.louvain
    core.StPipeline.phenograph
    core.StPipeline.find_marker_genes
    core.StPipeline.filter_marker_genes
    core.StPipeline.spatial_hotspot
    core.StPipeline.gaussian_smooth
    algorithm.cell_cell_communication.main.CellCellCommunication.main


Plots: `plt`
++++++++++++++++++++

.. module:: stereo.plt
.. currentmodule:: stereo

Visualization module

.. autosummary::
   :toctree: .


Plot Collection
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   plots.PlotCollection


Scatter
~~~~~~~~
.. autosummary::
   :toctree: .

   plots.scatter.base_scatter
   plots.scatter.volcano
   plots.scatter.highly_variable_genes


Interactive Plot
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: .

   plots.interact_spatial_cluster
   plots.interact_annotation_cluster
   plots.InteractiveScatter


Others
~~~~~~~~
.. autosummary::
   :toctree: .

   plots.violin_distribution
   plots.marker_genes_text
   plots.marker_genes_heatmap




Image: `im`
++++++++++++++++++++

Image parse module.

.. autosummary::
   :toctree: .

    image.pyramid.merge_pyramid
    image.pyramid.create_pyramid
    image.segmentation.segment.cell_seg
    image.cellbin.modules.cell_segmentation.cell_seg_v3
    image.segmentation_deepcell.segment.cell_seg_deepcell
    image.tissue_cut.SingleStrandDNATissueCut
    image.tissue_cut.RNATissueCut


Tools: `tools`
++++++++++++++++++++

Tools module.

.. autosummary::
   :toctree: .

    tools.cell_correct.cell_correct
    tools.cell_cut.CellCut.cell_cut


utils: `utils`
+++++++++++++++++++

Utils Module.

.. autosummary::
   :toctree: .

    utils.data_helper.merge
    utils.data_helper.split