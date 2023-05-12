.. module:: stereo
.. automodule:: stereo
   :noindex:

StPipeline:  `tl`
====================

Import Stereopy :

   import stereo as st

A tool-integrated class for basic analysis of StereoExpData, \
which is compromised of basic preprocessing, embedding, clustering, and so on. 

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
    core.StPipeline.spatial_neighbors
    core.StPipeline.umap
    core.StPipeline.leiden
    core.StPipeline.louvain
    core.StPipeline.phenograph
    core.StPipeline.find_marker_genes
    core.StPipeline.spatial_hotspot
    core.StPipeline.gaussian_smooth
    core.StPipeline.annotation
    algorithm.single_r.SingleR.main
    algorithm.cell_cell_communication.CellCellCommunication.main
    algorithm.regulatory_network_inference.RegulatoryNetworkInference.main