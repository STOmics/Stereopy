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
    algorithm.score_genes.ScoreGenes.main
    algorithm.score_genes_cell_cycle.ScoreGenesCellCycle.main
    algorithm.spa_track.SpaTrack.main
    algorithm.spa_track.SpaTrack.assess_start_cluster
    algorithm.spa_track.SpaTrack.set_start_cells
    algorithm.spa_track.SpaTrack.auto_estimate_param
    algorithm.spa_track.SpaTrack.calc_alpha_by_moransI
    algorithm.spa_track.SpaTrack.get_ot_matrix
    algorithm.spa_track.SpaTrack.get_ptime
    algorithm.spa_track.SpaTrack.get_velocity_grid
    algorithm.spa_track.SpaTrack.get_velocity
    algorithm.spa_track.SpaTrack.auto_get_start_cluster
    algorithm.spa_track.SpaTrack.lasso_select
    algorithm.spa_track.SpaTrack.create_vector_field
    algorithm.spa_track.SpaTrack.set_lap_endpoints
    algorithm.spa_track.SpaTrack.least_action
    algorithm.spa_track.SpaTrack.map_cell_to_LAP
    algorithm.spa_track.SpaTrack.filter_genes
    algorithm.spa_track.SpaTrack.ptime_gene_GAM
    algorithm.spa_track.SpaTrack.order_trajectory_genes
    algorithm.spa_track.SpaTrack.gr_training
    algorithm.ms_spa_track.MSSpaTrack.main
    algorithm.ms_spa_track.MSSpaTrack.transfer_matrix
    algorithm.ms_spa_track.MSSpaTrack.generate_animate_input
    algorithm.ms_spa_track.MSSpaTrack.gr_training