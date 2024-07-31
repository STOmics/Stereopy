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
    core.StPipeline.filter_coordinates
    core.StPipeline.filter_by_clusters
    core.StPipeline.filter_marker_genes
    core.StPipeline.filter_by_hvgs
    core.StPipeline.raw_checkpoint
    core.StPipeline.sctransform
    core.StPipeline.normalize_total
    core.StPipeline.log1p
    core.StPipeline.scale
    core.StPipeline.disksmooth_zscore
    core.StPipeline.quantile
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
    core.StPipeline.batches_integrate
    core.StPipeline.gaussian_smooth
    core.StPipeline.annotation
    core.StPipeline.adjusted_rand_score
    core.StPipeline.silhouette_score
    algorithm.single_r.SingleR.main
    algorithm.batch_qc.BatchQc.main
    algorithm.paste.Paste.main
    algorithm.paste.pairwise_align
    algorithm.paste.center_align
    algorithm.get_niche.GetNiche.main
    algorithm.gen_ccc_micro_envs.GenCccMicroEnvs.main
    algorithm.cell_cell_communication.CellCellCommunication.main
    algorithm.regulatory_network_inference.RegulatoryNetworkInference.main
    algorithm.co_occurrence.CoOccurrence.main
    algorithm.community_detection.CommunityDetection.main
    algorithm.time_series_analysis.TimeSeriesAnalysis.main
    algorithm.dendrogram.Dendrogram.main
    algorithm.st_gears.StGears.main
    algorithm.st_gears.StGears.stack_slices_pairwise_rigid
    algorithm.st_gears.StGears.stack_slices_pairwise_elas_field
    core.ms_pipeline.MSDataPipeLine.set_scope_and_mode
    algorithm.spa_seg.SpaSeg.main