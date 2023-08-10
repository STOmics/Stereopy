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

.. autosummary::
   :toctree: .

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
