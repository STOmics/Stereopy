Release Notes 
==============

.. role:: small

Version 0.9.0
-----------------
0.9.0 : 2023-01-10
~~~~~~~~~~~~~~~~~~~~~
1. Resolved cell boundary overlapping issues during cell correction visualization.
2. Addition of manually annotating cells and clusters via command lines or interactive visualization features.
3. Addition of GPU version of UMAP, Neighbors, Leiden, and Louvian.
4. Updated requirements.txt.

Version 0.8.0
------------------
0.8.0 : 2022-12-02
~~~~~~~~~~~~~~~~~~~~~
1. Reconstructed SCTransform normalization in python.
2. Optimized the efficiency of FAST cell correction.
3. Enabled to read Seurat output H5AD file for further analysis in Stereopy.

Version 0.7.0
------------------
0.7.0 : 2022-11-15
~~~~~~~~~~~~~~~~~~~~~
1. Acquired the cell expression matrix (cellbin) from the GEM file.
2. Updated hotspot to the latest version. Allow to output gene lists for every module.
3. Allowed to merge and arrange more than two matrices in a row.
4. Speeded up Stereopy installation and allowed installing heavy frameworks such as TensorFlow and PyTorch later before using.
5. Updated requirements.txt.

Version 0.6.0
------------------
0.6.0 : 2022-09-30
~~~~~~~~~~~~~~~~~~~~~
1. Increased 'Remove Batch Effect' algorithm.
2. Increased RNA velocity analysis function.
3. Increased export_high_res_area method to export high resolution matrix file(cell bin GEF) after lasso.
4. Updated algorithm of scale.
5. Optimized the efficiency of cell correction.
6. Increased multi-chip fusion analysis.
7. Updated requirements.txt.

Version 0.5.1
------------------
0.5.1 : 2022-09-4
~~~~~~~~~~~~~~~~~~~~~
1. Fixed bug when using GEM file to run fast-cell-correction algorithm.

Version 0.5.0
------------------
0.5.0 : 2022-09-2
~~~~~~~~~~~~~~~~~~~~~
1. Increased fast-cell-correction algorithm.
2. Updated gmm-cell-correction algorithm(slower version), fix bug that genes in the same position(bin) were assigned to different cells.
3. Increased data.plt.cells_plotting method to show the detail of the cells.
4. Increased export_high_res_area method to export high resolution matrix file(GEF) after lasso.
5. Increased tissue_extraction_to_bgef method to extract the tissue area.
6. Updated algorithm of highly_variable_genes, umap and normalization.
7. Updated requirements.txt.

Version 0.4.0
------------------
0.4.0 : 2022-07-30
~~~~~~~~~~~~~~~~~~~~~
1. Updated tissue cut algorithm.
2. Increased param(n_jobs) in neighbors and phenograph.
3. Increased read_gef function filter by region of gene list.
4. Updated requirements.txt.

Version 0.3.1
------------------
0.3.1 : 2022-06-30
~~~~~~~~~~~~~~~~~~~~~
1. Added gaussian smooth function.
2. The pca function increased the svd_solver parameter.
3. The write_h5ad function increased the output parameter.
4. Updated requirements.txt.

Version 0.3.0
------------------
0.3.0 : 2022-06-10
~~~~~~~~~~~~~~~~~~~~~
1. Added cell bin correction function.
2. Added scale function in normalization.
3. Supported write the StereoExpData into a GEF file.
4. Fixed bug of sctransform, reading the GEF/GEM file and annh5ad2rds.R.
5. Updated default cluster groups to start at 1.
6. Supported write StereoExpData to stereo h5ad function.
7. Updated requirements.txt.

Version 0.2.4
------------------
0.2.4 : 2022-01-19
~~~~~~~~~~~~~~~~~~~~~
1. Fixed bug of tar package.

Version 0.2.3
-----------
0.2.3 : 2022-01-17
~~~~~~~~~~~~~~~~~~~~~~~
1. Added cell segmentation and tissuecut segmentation function.
2. Updated stereo_to_anndata function and support output to h5ad file.
3. Added the Rscript supporting h5ad file(with anndata object) to rds file.
4. Supported DEG output to the csv file.

Version 0.2.2
-----------
0.2.2 : 2021-11-17
~~~~~~~~~~~~~~~~~~~~~~~
1. Optimized the performance of find marker.
2. Added Cython setup_build function and optimize GEF io performance.
3. Added hotspot pipeline for spatial data and squidpy for spatial_neighbor func.
4. Added polygon selection for interactive scatter plot and simplify the visualization part of the code.


Version 0.2.1
-----------
0.2.1 : 2021-10-15
~~~~~~~~~~~~~~~~~~~~~~~
1. Fixed the bug of marker_genes_heatmap IndexError and sort the text of heatmap plot.
2. Inverted yaxis one the top for spatial_scatter and cluster_scatter plot funcs.
3. Solved the problem that multiple results of sctransform run are inconsistent.
4. Updated requirements.txt.


Version 0.2.0
-----------
0.2.0 : 2021-09-16
~~~~~~~~~~~~~~~~~~~~~~~~~

Stereopy provides the analysis process based on spatial omics, including reading, preprocessing, clustering,
differential expression testing and visualization, etc. There are the updates we made in this version.

1. We proposed StereoExpData, which is a data format specially adapted to spatial omics analysis.
2. Supported reading the GEF file, which is faster than reading GEM file.
3. Supported the conversion between StereoExpData and AnnData.
4. Added the interactive visualization function for selecting data, you can dynamically select the area of interest, and then perform the next step of analysis.
5. Supported dynamically displaying clustering scatter plots, you can modify the color and point size.
6. Updated clustering related methods, such as leiden, louvain, which are comparable to the original algorithms.
7. Added some analysis, such as the method of logres for find marker genes, highly variable genes analysis, sctransform method of normalization like Seruat.


0.1.0 : 2021-05-30
~~~~~~~~~~~~~~~~~~~~~~~~~
- Initial release
