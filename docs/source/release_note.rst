Release Notes
=============

.. role:: small

Version 0.4.0
------------------
0.4.0 :2022-07-30
~~~~~~~~~~~~~~~~~~~~~
1. Update tissue cut algorithm.
2. Increase param(n_jobs) in neighbors and phenograph.
3. Increase read_gef function filter by region of gene list.
4. Update requirements.txt.

Version 0.3.1
------------------
0.3.1 :2022-06-30
~~~~~~~~~~~~~~~~~~~~~
1. Add gaussian smooth function.
2. The pca function increases the svd_solver parameter.
3. The write_h5ad function increases the output parameter.
4. Update requirements.txt.

Version 0.3.0
------------------
0.3.0 :2022-06-10
~~~~~~~~~~~~~~~~~~~~~
1. Add cell bin correction function.
2. Add scale function in normalization.
3. Support write the StereoExpData into a gef file.
4. Fix bug of sctransform, reading the gef/gem file and annh5ad2rds.R.
5. Update default cluster groups to start at 1.
6. Support write StereoExpData to stereo h5ad function.
7. Update requirements.txt.

Version 0.2.4
------------------
0.2.4 :2022-01-19
~~~~~~~~~~~~~~~~~~~~~
1. Fix bug of tar package.

Version 0.2.3
-----------
0.2.3 :2022-01-17
~~~~~~~~~~~~~~~~~~~~~~~
1. Add cell segmentation and tissuecut segmentation function.
2. Update stereo_to_anndata function and support output to h5ad file.
3. Add the Rscript supporting h5ad file(with anndata object) to rds file.
4. Support DEG output to the csv file.

Version 0.2.2
-----------
0.2.2 :2021-11-17
~~~~~~~~~~~~~~~~~~~~~~~
1. Optimize the performance of find marker.
2. Add Cython setup_build function and optimize gef io performance.
3. Add hotspot pipeline for spatial data and squidpy for spatial_neighbor func.
4. Add polygon selection for interactive scatter plot and simplify the visualization part of the code.


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

1. We propose StereoExpData, which is a data format specially adapted to spatial omics analysis.
2. Support reading the gef file, which is faster than reading gem file.
3. Support the conversion between StereoExpData and AnnData.
4. Add the interactive visualization function for selecting data, you can dynamically select the area of interest, and then perform the next step of analysis.
5. Dynamically display clustering scatter plots, you can modify the color and point size.
6. Updated clustering related methods, such as leiden, louvain, which are comparable to the original algorithms.
7. Add some analysis, such as the method of logres for find marker genes, highly variable genes analysis, sctransform method of normalization like Seruat.


0.1.0 :2021-05-30
~~~~~~~~~~~~~~~~~~~~~~~~~
- Initial release
