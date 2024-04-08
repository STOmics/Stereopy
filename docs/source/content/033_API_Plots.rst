.. module:: stereo
.. automodule:: stereo
    :noindex:

Plots: `plt`
===================

Import Stereopy :

    import stereo as st

.. module:: stereo.plt
.. currentmodule:: stereo

Here supports both static and interactive plotting modes to visualize our analysis results vividly. 


Plot Collection
~~~~~~~~~~~~~~~~~~~
The plot collection for StereoExpData object.

.. autosummary::
    :toctree: .

    plots.PlotCollection.cells_plotting
    plots.PlotCollection.cluster_scatter
    plots.PlotCollection.gaussian_smooth_scatter_by_gene
    plots.PlotCollection.genes_count
    plots.PlotCollection.highly_variable_genes
    plots.PlotCollection.hotspot_local_correlations
    plots.PlotCollection.hotspot_modules
    plots.PlotCollection.interact_cluster
    plots.PlotCollection.interact_spatial_scatter
    plots.PlotCollection.interact_annotation_cluster
    plots.PlotCollection.marker_genes_volcano
    plots.PlotCollection.marker_genes_heatmap
    plots.PlotCollection.marker_genes_text
    plots.PlotCollection.marker_genes_scatter
    plots.PlotCollection.spatial_scatter
    plots.PlotCollection.spatial_scatter_by_gene
    plots.PlotCollection.umap
    plots.PlotCollection.batches_umap
    plots.PlotCollection.violin
    algorithm.cell_cell_communication.PlotCellCellCommunication.ccc_dot_plot
    algorithm.cell_cell_communication.PlotCellCellCommunication.ccc_heatmap
    algorithm.cell_cell_communication.PlotCellCellCommunication.ccc_circos_plot
    algorithm.cell_cell_communication.PlotCellCellCommunication.ccc_sankey_plot
    algorithm.regulatory_network_inference.PlotRegulatoryNetwork.auc_heatmap_by_group
    algorithm.regulatory_network_inference.PlotRegulatoryNetwork.auc_heatmap
    algorithm.regulatory_network_inference.PlotRegulatoryNetwork.grn_dotplot
    algorithm.regulatory_network_inference.PlotRegulatoryNetwork.spatial_scatter_by_regulon_3D
    algorithm.regulatory_network_inference.PlotRegulatoryNetwork.spatial_scatter_by_regulon
    plots.PlotCoOccurrence.co_occurrence_plot
    plots.PlotCoOccurrence.co_occurrence_heatmap
    plots.PlotPaga.paga_plot
    plots.PlotPaga.paga_compare
    plots.PlotDendrogram.dendrogram
    plots.ClustersGenesScatter.clusters_genes_scatter
    plots.ClustersGenesHeatmap.clusters_genes_heatmap
    plots.PlotTimeSeries.boxplot_transit_gene
    plots.PlotTimeSeries.TVG_volcano_plot
    plots.PlotTimeSeries.paga_time_series_plot
    plots.PlotTimeSeries.fuzz_cluster_plot
    plots.PlotTimeSeriesAnalysis.time_series_tree_plot
    plots.PlotTimeSeriesAnalysis.ms_paga_time_series_plot
    plots.PlotElbow.elbow
    plots.PlotGenesInPseudotime.plot_genes_in_pseudotime
    plots.PlotVec.plot_vec
    plots.PlotVec.plot_time_scatter
    plots.PlotVec3D.plot_vec_3d