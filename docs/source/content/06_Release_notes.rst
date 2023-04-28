Release Notes 
==============

.. role:: small

Version 0.12.0
---------------------
0.12.0 : 2023-04-27
~~~~~~~~~~~~~~~~~~~~~~~~
1. Addition of the algorithm of Cell Segmentation V3.0.
2. Addition of method='hotspot' to data.tl.regulatory_network_inference, which takes spatial coordinate information into account to calculate the relative importance between TFs and their target genes.
3. Addition of dpi and width/height setting for visualization, and addition of plotting scale for displaying static plot.
4. Optimized required memory while plotting UMAP embedding via data.plt.umap and cell distribution via data.plt.cells_plotting.
5. Fixed bug that input parameter of var_features_n was invalid, in data.tl.scTransform.
6. Updated requirements.txt.

Version 0.11.0
---------------------
0.11.0 : 2023-04-04
~~~~~~~~~~~~~~~~~~~~~~~~
1. Addition of Cell-cell Communication analysis.
2. Addition of Gene Regulatory Network analysis.
3. Addition of SingleR function for automatic annotation.
4. Addition of `v2` algorithm fast cell correction.
5. Addition of dot plot to display gene-level results.
6. Addition of the sorting function and the limitation of output genes in `data.tl.find_marker_genes`.
7. Added `pct` and `pct_rest` to the output files of marker genes.
8. Addition of the parameter `mean_uni_gt` in `data.tl.filter_genes` to filter genes on average expression.
9. Fixed the bug that `adata.X` to output AnnData was the raw matrix.
10. Fixed the failed compatibility to analysis results from `.h5ad` (version <= 0.9.0).
11. Updated the tissue segmentation algorithm in the module of cell segmentation to avoid the lack of tissue.
12. Reconstructed the manual of Stereopy.
13. Updated requirements.txt.

Version 0.10.0
------------------
0.10.0 :2023-02-22
~~~~~~~~~~~~~~~~~~~~~
1. Supported installation on Windows.
2. Addition of displaying basic information of StereoExpData object when simply typing it.
3. Addition of saving statistic results when plotting.
4. Addition of marker gene proportion (optional), in-group and out-of-group, in `data.tl.find_marker_genes`. Otherwise, supported filtering marker genes via `data.tl.filter_marker_genes`.
5. Supported adapting to AnnData, to directly use data and results stored in AnnData for subsequent analysis.
6. Addition of the matrix of gene count among clusters so that transformed output `.rds` file could be used for annotation by SingleR directly. 
7. Initial release of Stereopy development solution.
8. Updated requirements.txt.

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
1. Reconstructed scTransform normalization in Stereopy.
2. Optimized the efficiency of fast-cell-correction.
3. Enabled to read Seurat output `.h5ad` file for further analysis.

Version 0.7.0
------------------
0.7.0 : 2022-11-15
~~~~~~~~~~~~~~~~~~~~~
1. Supported acquiring the cell expression matrix (cellbin) from GEM file.
2. Updated hotspot to the latest version. Allow to output gene lists for every module.
3. Allowed to merge and arrange more than two matrices in a row.
4. Speeded up Stereopy installation and allowed installing heavy frameworks, such as, TensorFlow and PyTorch later before using.
5. Updated requirements.txt.

Version 0.6.0
------------------
0.6.0 : 2022-09-30
~~~~~~~~~~~~~~~~~~~~~
1. Added 'Remove Batch Effect' algorithm.
2. Added RNA velocity analysis.
3. Added `export_high_res_area` method to export high resolution matrix file(cell bin GEF) after lasso operation.
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
1. Added fast-cell-correction algorithm.
2. Updated gmm-cell-correction algorithm(slower version),  and fixed bug that genes in the same position(bin) were assigned to different cells.
3. Added `plt.cells_plotting` method to display cell details.
4. Added `tl.export_high_res_area` method to export high resolution matrix file(GEF) after lasso.
5. Increased tissue_extraction_to_bgef method to extract the tissue area.
6. Updated algorithm of highly_variable_genes, umap and normalization.
7. Updated requirements.txt.

Version 0.4.0
------------------
0.4.0 : 2022-07-30
~~~~~~~~~~~~~~~~~~~~~
1. Updated tissue segmentation algorithm.
2. Added the `n_jobs` parameter in `tl.neighbors` and `tl.phenograph`.
3. Added `io.read_gef` function filtered by the list of gene region.
4. Updated requirements.txt.

Version 0.3.1
------------------
0.3.1 : 2022-06-30
~~~~~~~~~~~~~~~~~~~~~
1. Added gaussian smooth function.
2. Added the `svd_solver` parameter in `tl.pca`.
3. Added the `output` parameter in `io.write_h5ad`.
4. Updated requirements.txt.

Version 0.3.0
------------------
0.3.0 : 2022-06-10
~~~~~~~~~~~~~~~~~~~~~
1. Added cell bin correction function.
2. Added `tl.scale` function in normalization.
3. Supported writing StereoExpData object into a GEF file.
4. Fixed bug of scTransform, reading the GEF/GEM file and annh5ad2rds.R.
5. Updated default cluster groups to start at 1.
6. Supported writing StereoExpData to stereo `.h5ad` function.
7. Updated requirements.txt.

Version 0.2.4
------------------
0.2.4 : 2022-01-19
~~~~~~~~~~~~~~~~~~~~~
1. Fixed bug of tar package.

Version 0.2.3
------------------
0.2.3 : 2022-01-17
~~~~~~~~~~~~~~~~~~~~~~~
1. Added cell segmentation and tissue segmentation function.
2. Updated stereo_to_anndata function and supported output to `.h5ad` file.
3. Added the Rscript supporting h5ad file(with anndata object) to rds file.
4. Supported differentially expressed gene (DEG) output to the `.csv` file.

Version 0.2.2
------------------
0.2.2 : 2021-11-17
~~~~~~~~~~~~~~~~~~~~~~~
1. Optimized the performance of finding marker genes.
2. Added Cython setup_build function and optimized IO performance of GEF.
3. Added hotspot pipeline for spatial data and Squidpy for spatial_neighbor function.
4. Added polygon selection for interactive scatter plot and simplify the visualization part of the code.


Version 0.2.1
------------------
0.2.1 : 2021-10-15
~~~~~~~~~~~~~~~~~~~~~~~
1. Fixed the bug of marker_genes_heatmap IndexError and sorted the text of heatmap plot.
2. Inverted yaxis on the top for spatial_scatter and cluster_scatter plot funcs.
3. Solved the problem that multiple results of sctransform run were inconsistent.
4. Updated requirements.txt.


Version 0.2.0
------------------
0.2.0 : 2021-09-16
~~~~~~~~~~~~~~~~~~~~~~~~~

Stereopy provides the analysis process based on spatial omics, including reading, preprocessing, clustering,
differential expression testing and visualization, etc. There are the updates we made in this version.

1. We proposed StereoExpData, which is a data format specially adapted to spatial omics analysis.
2. Supported reading the GEF file, which is faster than reading GEM file.
3. Supported the conversion between StereoExpData and AnnData.
4. Added the interactive visualization function for selected data, you can dynamically select the area of interest, and then perform the next step of analysis.
5. Supported dynamically displaying clustering scatter plots, you can modify the color and point size.
6. Updated clustering related methods, such as leiden, louvain, which are comparable to the original algorithms.
7. Added some analysis, such as the method of logres for find marker genes, highly variable genes analysis, sctransform method of normalization like Seruat.


0.1.0 : 2021-05-30
~~~~~~~~~~~~~~~~~~~~~~~~~
- Initial release
