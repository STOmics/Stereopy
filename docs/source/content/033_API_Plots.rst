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
   plots.PlotCollection.marker_gene_volcano
   plots.PlotCollection.marker_genes_heatmap
   plots.PlotCollection.marker_genes_text
   plots.PlotCollection.spatial_scatter
   plots.PlotCollection.spatial_scatter_by_gene
   plots.PlotCollection.umap
   plots.PlotCollection.violin
