Release Notes
=============

.. role:: small


Version 0.2.1
-----------
0.2.1 :2021-10-15
~~~~~~~~~~~~~~~~~~~~~~~
1. fix the bug of marker_genes_heatmap IndexError and sort the text of heatmap plot.
2. invert yaxis one the top for spatial_scatter and cluster_scatter plot funcs.
3. solve the problem that multiple results of sctransform run are inconsistent.
4. update requirements.txt.


Version 0.2.0
-----------
0.2.0 :2021-09-16
~~~~~~~~~~~~~~~~~~~~~~~~~

Stereopy provides the analysis process based on spatial omics, including reading, preprocessing, clustering,
differential expression testing and visualization, etc. There are the updates we made in this version.

1. We propose StereoExpData, which is  a data format specially adapted to spatial omics analysis.
2. Support reading the gef file, which is faster than reading gem file.
3. Support the conversion between StereoExpData and AnnData.
4. Add the interactive visualization function for selecting data, you can dynamically select the area of interest, and then perform the next step of analysis.
5. Dynamically display clustering scatter plots, you can modify the color and point size.
6. Updated clustering related methods, such as leiden, louvain, which are comparable to the original algorithms.
7. Add some analysis, such as the method of logres for find marker genes, highly variable genes analysis, sctransform method of normalization like Seruat.


0.1.0 :2021-05-30
~~~~~~~~~~~~~~~~~~~~~~~~~
- Initial release