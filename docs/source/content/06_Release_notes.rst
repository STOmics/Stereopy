Release Notes 
==============

.. role:: small

Version 1.6.0
------------------
1.6.0 : 2025-02-21
~~~~~~~~~~~~~~~~~~~

.. _Spatialign: ../Tutorials(Multi-sample)/Spatial_Alignment.html
.. |Spatialign| replace:: **Spatialign**

Features:

1. Addition of new algorithm |Spatialign|_ for batch effect removal.

Version 1.5.1
------------------
1.5.1 : 2024-12-26
~~~~~~~~~~~~~~~~~~~

.. _st.io.stereo_to_anndata: stereo.io.stereo_to_anndata.html
.. |st.io.stereo_to_anndata| replace:: `st.io.stereo_to_anndata`

.. _h5ad2rds.R: ../Tutorials/Format_Conversion.html
.. |h5ad2rds.R| replace:: **h5ad2rds.R**

Features:

1. |st.io.stereo_to_anndata|_ supports adding image information into the converted **AnnData** object.
2. |h5ad2rds.R|_ supports adding image information into the converted **RDS** file.
3. Optimized the visualization of the plotting scale for spatial scatter plot when inputting small data.

BUG Fixes:

1. Fixed the problem that the layers was lost when converting **StereoExpData** to **AnnData** by using `st.io.stereo_to_anndata`.
2. Fixed the problem that the result of `st.tl.gen_ccc_micro_envs` cannot be reproduced.

Version 1.5.0
------------------
1.5.0 : 2024-11-08
~~~~~~~~~~~~~~~~~~~

.. _SpaTrack: ../Tutorials/SpaTrack.html
.. |SpaTrack| replace:: **SpaTrack**

.. _Layer: stereo.core.StPipeline.set_layer.html
.. |Layer| replace:: **Layer**

.. _st.tl.cal_qc: stereo.core.StPipeline.cal_qc.html
.. |st.tl.cal_qc| replace:: `st.tl.cal_qc`

.. _st.tl.filter_cells: stereo.core.StPipeline.filter_cells.html
.. |st.tl.filter_cells| replace:: `st.tl.filter_cells`

.. _st.tl.filter_genes: stereo.core.StPipeline.filter_genes.html
.. |st.tl.filter_genes| replace:: `st.tl.filter_genes`

.. _st.tl.log1p: stereo.core.StPipeline.log1p.html
.. |st.tl.log1p| replace:: `st.tl.log1p`

.. _st.tl.normalize_total: stereo.core.StPipeline.normalize_total.html
.. |st.tl.normalize_total| replace:: `st.tl.normalize_total`

.. _st.tl.scale: stereo.core.StPipeline.scale.html
.. |st.tl.scale| replace:: `st.tl.scale`

.. _st.tl.quantile: stereo.core.StPipeline.quantile.html
.. |st.tl.quantile| replace:: `st.tl.quantile`

.. _st.tl.disksmooth_zscore: stereo.core.StPipeline.disksmooth_zscore.html
.. |st.tl.disksmooth_zscore| replace:: `st.tl.disksmooth_zscore`

.. _st.tl.sctransform: stereo.core.StPipeline.sctransform.html
.. |st.tl.sctransform| replace:: `st.tl.sctransform`

.. _st.tl.highly_variable_genes: stereo.core.StPipeline.highly_variable_genes.html
.. |st.tl.highly_variable_genes| replace:: `st.tl.highly_variable_genes`

.. _st.tl.pca: stereo.core.StPipeline.pca.html
.. |st.tl.pca| replace:: `st.tl.pca`

.. _st.tl.find_marker_genes: stereo.core.StPipeline.find_marker_genes.html
.. |st.tl.find_marker_genes| replace:: `st.tl.find_marker_genes`

.. _st.plt.spatial_scatter: stereo.plots.PlotCollection.spatial_scatter.html
.. |st.plt.spatial_scatter| replace:: `st.plt.spatial_scatter`

Features:

1. Addition of new algorithm |SpaTrack|_ for trajectory inference.
2. Addition of |Layer|_ for saving expression matrices at different analysis stages, the functions that can use expression matrices in **Layer** as following:
            * |st.tl.cal_qc|_
            * |st.tl.filter_cells|_
            * |st.tl.filter_genes|_
            * |st.tl.log1p|_
            * |st.tl.normalize_total|_
            * |st.tl.scale|_
            * |st.tl.quantile|_
            * |st.tl.disksmooth_zscore|_
            * |st.tl.sctransform|_
            * |st.tl.highly_variable_genes|_
            * |st.tl.pca|_
            * |st.tl.find_marker_genes|_
3. Merger of multiple samples can merge some analysis result in every single samples when data type is **StereoExpData**.
4. |st.plt.spatial_scatter|_ supports setting **regist.tif** as background to display simultaneously on the plot.

BUG Fixes:

1. Fixed the problem that the proportion of chondriogenes was calculated incorrectly when input data contains **geneID**.
2. Fixed the problem that saving **MSData** into **h5mu** was failed after running `st.tl.highly_variable_genes`.

Version 1.4.0
------------------
1.4.0 : 2024-09-05
~~~~~~~~~~~~~~~~~~~

.. _SpaSEG: ../Tutorials(Multi-sample)/SpaSEG.html
.. |SpaSEG| replace:: **SpaSEG**

.. _st.plt.cells_plotting: stereo.plots.PlotCollection.cells_plotting.html
.. |st.plt.cells_plotting| replace:: `st.plt.cells_plotting`

.. _st.io.write_h5mu: stereo.io.write_h5mu.html
.. |st.io.write_h5mu| replace:: `st.io.write_h5mu`

.. _st.io.mudata_to_msdata: stereo.io.mudata_to_msdata.html
.. |st.io.mudata_to_msdata| replace:: `st.io.mudata_to_msdata`

Features:

1. Addition of new algorithm |SpaSEG|_ for multiple **SRT** analysis.
2. Addition of **colorbar** or **legend** for `st.plt.cells_plotting`.
3. |st.plt.cells_plotting|_ supports exporting plots as **PNG**, **SVG** or **PDF**.
4. Addition of method |st.io.write_h5mu|_ and |st.io.mudata_to_msdata|_ for conversion between **MSData** and **MuData**.

BUG Fixes:

1. Fixed the problem that **CellCorrection** is incompatible with small-size images (less than 2000px in any dimension) when using the **EDM** method.
2. Fixed the problem that `MSData.to_integrate` is incompatible when the number of cells in the integrated sample is less than the total number of cells in all single samples.
3. Fixed the problem that `st.plt.time_series_tree_plot` can not capture the result of **PAGA**, leading to an incorrect plot.
4. Fixed other bugs.


Version 1.3.1
------------------
1.3.1 : 2024-06-28
~~~~~~~~~~~~~~~~~~~

Features:

1. Addition of new method **'adaptive'** for `st.tl.get_niche <stereo.algorithm.get_niche.GetNiche.main.html>`_ (the original method is named **'fixed'**).
2. Changed some parameter names of `st.tl.filter_cells <stereo.core.StPipeline.filter_cells.html>`_ and `st.tl.filter_genes <stereo.core.StPipeline.filter_genes.html>`_ for eliminating ambiguity(old parameter names are still compatible).
3. Filter the results of **PCA** and **UMAP** simultaneously when running `st.tl.filter_cells`.

BUG Fixes:

1. Fixed the problem that `ms_data.to_isolated` is incompatible with that there are duplicate **cell names** in different samples.
2. Fixed the problem that `st.io.read_gef` is incompatible with those **GEF** files that contain **gene names** ending with **'_{number}'** (like **'ABC_123'**).
3. Upgraded **gefpy** to latest for fixing the error that **gene names** are lost after running **CellCorrection**.


Version 1.3.0
------------------
1.3.0 : 2024-05-31
~~~~~~~~~~~~~~~~~~~

Features:

1. Addition of `MSData.tl.st_gears <../Tutorials(Multi-sample)/ST_Gears.html>`_ for spatial alignment of **Multi-sample**.
2. `High Resolution Matrix Export <../Tutorials/High_Resolution_Export.html>`_ can support both **GEF** and **GEM** files.
3. Addition of parameters `min_count` and `max_count` for `st.tl.filter_genes <stereo.core.StPipeline.filter_genes.html>`_.
4. `MSData.integrate <stereo.core.ms_data.MSData.integrate.html>`_ can be compatible with sparse matrix when `MSData.var_type` is `union`.
5. Addition of `MSData.tl.set_scope_and_mode <stereo.core.ms_pipeline.MSDataPipeLine.set_scope_and_mode.html>`_ to set `scope` and `mode` globally on **Multi-sample** analysis.
6. Addition of `MSData.plt.ms_spatial_scatter <stereo.plots.PlotMsSpatialScatter.ms_spatial_scatter.html>`_ to plot spatial scatter plot for each **sample** in **Multi-sample** separately.

BUG Fixes:

1. Fixed the problem that `st.io.read_gem` is incompatible with **GEM** files containing **geneID**.
2. Fixed the bug of losing part of metadata when writing **StereoExpData** / **MSData** into **Stereo-h5ad** or **h5ms** file.
3. Fixed the incompatibility problem with **AnnData** when performing `st.tl.sctransform`.


Version 1.2.0
------------------
1.2.0 : 2024-03-30
~~~~~~~~~~~~~~~~~~~

Features:

1. `st.io.read_gem` and `st.io.read_gef` support expression matrix files with geneID information.
2. Analysis results of `find_marker_genes`  will be saved into the output AnnData h5ad.
3. Upgraded tissue segmentation algorithm.
4. Addition of `st.tl.adjusted_rand_score` to calculate the adjusted Rand coefficient between two clusters.
5. Addition of `st.tl.silhouette_score` to calculate the average silhouette coefficient of a cluster.
6. `h5ad2rds.R` is compatible with AnnData version > 0.7.5, to convert from h5ad to rds files.
7. Addition of the clustering category labels to the graph of `st.plt.paga_compare`.

BUG Fixes:

1. Fixed the error of high memory consumption when converting `X.raw` into AnnData.


Version 1.1.0
------------------
1.1.0 : 2024-01-17
~~~~~~~~~~~~~~~~~~~

Features:

1. Reconstructed `st.plt.violin` visualizing function which is now not only applied to display QC indicators;
2. `ins.export_high_res_area` can handle expression matrix and image simultaneously, to lasso region of interest and corresponding sub-image.
3. Interactive visualizing `st.plt.cells_plotting` supported displaying expression heatmap and spatial distribution of a single gene.
4. When input GEF and GEM at cell level, information of DNB count and cell area would be added into `cells` / `obs`, and cell border would be added into `cells_matrix` / `obsm`.

BUG Fixes:

1. `slideio` package removed historical versions, resulting in an installation failure.
2. Calculating error when performing `ms_data.tl.batch_qc`, due to abnormal `os.getlogin`.
3. `st.plt.paga_time_series_plot` indicated that the image was too large to draw, due to unprocessed boundary values when computing median.

Version 1.0.0
------------------
1.0.0 : 2023-12-04
~~~~~~~~~~~~~~~~~~~

Features:

1. Addition of GPU acceleration on SinlgeR for large-volume data, and optimized calculating based on CPU version.
2. Addition of `st.plt.elbow` to visualize PCA result, for appropriate number of pcs.
3. Addition of color, max, min setting for colorbar, when plotting heatmap.
4. Addition of cell segmentation of `Deep Learning Model V1_Pro`, which is improved based on `V1`.
5. Supplemented parameters of `st.plt.auc_heatmap` and `st.plt.auc_heatmap_by_group`, full access to `seaborn.clustermap`;
6. Addition of thread and seed setting in `st.tl.umap`, of which the default method have been changed to single thread with the sacrifice of computational efficiency to ensure reproducibility of results. More in https://umap-learn.readthedocs.io/en/latest/reproducibility.html.
7. Modification of computing method of bin coordinates when reading GEM, consistent with GEF.
8. Optimized `st.io.stereo_to_anndata` for efficient format conversion.
9. Renamed `st.tl.spatial_alignment` function as `st.tl.paste`.
10. `export_high_res_area` removed parameter `cgef`.

BUG Fixes:

1. Occasional square-hollowing area in `Deep Learning Model V3` of cell segmentation processing.
2. `st.tl.annotation` could not set two or more clusters as a same name. 
3. The data object `ins.selected_exp_data` obtained from `st.plt.interact_spatial_scatter` could not be used for subsequent analysis.
4. Part of data was missing when performed `st.plt.interact_spatial_scatter` to output high-resolution matrix in GEF format.
5. Some files met reading error, led by no default setting of `bin_type` and `bin_size` in `st.io.read_h5ms`.
6. Error in Batch QC calculation due to data type problem.
7. There is NaN in Cell Community Detection output after threshold filtering, resulting in a calculating error when performed Find marker genes based on it.
8. `st.plt.paga_time_series_plot` indicated the image is too large to draw, leading to graph overlap, due to the limitation of matplotlib package.

Version 0.14.0b1 (Beta)
------------------------
0.14.0b1 : 2023-9-15
~~~~~~~~~~~~~~~~~~~~~~~~
Notice: this Beta version is specifically developed for multi-sample analysis.

Features:

1. Addition of Cell Community Detection (CCD) analysis.
2. Addition of Cell Co-occurrence analysis.
3. Addition of Cellpose in cell segmentation, especially for cell cytoplasm using `model_type='cyto2'`.
4. Addition of circos (`st.plt.ccc_circos_plot`) and sankey (`st.plt.ccc_sankey_plot`) plots in Cell-cell Communication analysis.
5. Addition of volcano (`st.plt.TVG_volcano_plot`) and tree (`st.plt.time_series_tree_plot`) plots in Time Series analysis.
6. Addition of PAGA tree plot, `st.plt.paga_plot`.
7. Addition of visuallization of `st.tl.dendrogram`.
8. Addition of version check using `st.__version__`.
9. Supported obtain subset from a data object, using clustering output, by `st.tl.filter_by_clusters`.
10. Supported filtering data using hvgs, by `st.tl.filter_by_hvgs`.
11. Supported mapping the clustering result of SquareBin analysis to the same data but in CellBin.
12. Supported writing annotation information into CellBin GEF file, only clustering result available before.
13. Supported saving images of PNG and PDF formats, in interactive interface.
14. Optimized the function of `st.tl.find_marker_genes`.
15. Optimized the modification of titles in horizontal axis, vertical axis and plot.

BUG Fixes:

1. Fixed the issue that SingleR calculating did not add filtration to the column field when traversing expression matrix, resulting in the subsequent absence of the column index.
2. Fixed the issue that output Seurat h5ad could not be transformed into R format.
3. Fixed the issue that clustering output of Leiden was in wrong data type under the scene of GPU acceleration, leading to errors in subsequent analysis which work on the clustering result.
4. Fixed the issue that clustering result could not be written into GEF file, using `st.io.update_gef`, caused by data type error. From v0.12.1 on, `date.cells.cell_name` has changed from int to string. 

Version 0.13.0b1 (Beta)
------------------------
0.13.0b1 : 2023-07-11
~~~~~~~~~~~~~~~~~~~~~~~~
Notice: this Beta version is specifically developed for multi-sample analysis. Major update points are listed below.

1. Addition of 3D Cell-cell Communication.
2. Addition of 3D Gene Regulatory Network.
3. Addition of Trajectory Inference, including PAGA and DPT algorithms.
4. Addition of Batch QC function for evaluation on batch effect.
5. Addition of `st.io.read_h5ad` for improved compatibility with AnnData H5ad, we highly recommend that instead of `st.io.read_ann_h5ad`.
6. Addition of analysis workflow tutorial based on multi-sample data, with assistant parameters `scope` and `mode`.
7. Addition of resetting the image order of multi-sample analysis results.
8. Addition of 3D mesh visualization.
9. Improved the performance of Gaussian Smoothing.

Version 0.12.1
---------------------
0.12.1 : 2023-06-21
~~~~~~~~~~~~~~~~~~~~~~~~
1. Addition of the pretreatment of calculating quality control metrics at the start of `st.tl.filter_genes` and `st.tl.filter_cells`.
2. Fixed the bug that loaded data from GEF file had the same expression matrix but in different row order, through updating gefpy package to v0.6.24.
3. Fixed the bug that `scale.data` had `np.nan` value in `st.tl.sctransform` , caused by data type limitation.
4. Fixed the bug that dot symbol ( '.' ) caused identification error of cluster name in `.csv` output, when doing `st.tl.find_marker_genes`.

Version 0.12.0
---------------------
0.12.0 : 2023-04-27
~~~~~~~~~~~~~~~~~~~~~~~~
1. Addition of the algorithm of Cell Segmentation V3.0.
2. Addition of `method='hotspot'` to `st.tl.regulatory_network_inference`, which takes spatial coordinate information into account to calculate the relative importance between TFs and their target genes.
3. Addition of dpi and width/height setting for visualization, and addition of plotting scale for displaying static plot.
4. Optimized required memory while plotting UMAP embedding via `data.plt.umap` and cell distribution via `data.plt.cells_plotting`.
5. Fixed bug that input parameter of `var_features_n` was invalid, in `data.tl.scTransform`.
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
3. Addition of saving static results plots.
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
3. Added `data.plt.cells_plotting` method to display cell details.
4. Added `data.tl.export_high_res_area` method to export high resolution matrix file(GEF) after lasso.
5. Increased tissue_extraction_to_bgef method to extract the tissue area.
6. Updated algorithm of highly_variable_genes, umap and normalization.
7. Updated requirements.txt.

Version 0.4.0
------------------
0.4.0 : 2022-07-30
~~~~~~~~~~~~~~~~~~~~~
1. Updated tissue segmentation algorithm.
2. Added the `n_jobs` parameter in `st.tl.neighbors` and `st.tl.phenograph`.
3. Added `st.io.read_gef` function filtered by the list of gene region.
4. Updated requirements.txt.

Version 0.3.1
------------------
0.3.1 : 2022-06-30
~~~~~~~~~~~~~~~~~~~~~
1. Added gaussian smooth function.
2. Added the `svd_solver` parameter in `data.tl.pca`.
3. Added the `output` parameter in `st.io.write_h5ad`.
4. Updated requirements.txt.

Version 0.3.0
------------------
0.3.0 : 2022-06-10
~~~~~~~~~~~~~~~~~~~~~
1. Added cell bin correction function.
2. Added `data.tl.scale` function in normalization.
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
