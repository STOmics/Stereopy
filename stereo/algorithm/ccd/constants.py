# flake8: noqa
COMMUNITY_DETECTION_DEFAULTS = {
    # File path to Anndata object with calculated cell mixtures for data windows, output of calc_feature_matrix.
    'tfile': None,
    # Absolute path to store outputs.
    'out_path': "results",
    # Clustering algorithm.
    'cluster_algo': "leiden",
    # Resolution of leiden clustering algorithm. Ignored for spectral and agglomerative.
    'resolution': 0.2,
    # Number of clusters for spectral and agglomerative clustering. Ignored for leiden.
    'n_clusters': 10,
    # Size of the spot on plot.
    'spot_size': 30,
    # Show logging messages. 0 - Show warnings, >0 show info.
    'verbose': 0,
    # Save plots flag. 0 - No plotting/saving, 1 - save clustering plot, 2 - additionally save plots of cell type images statistics and cell mixture plots,
    # 3 - additionally save cell and cluster abundance plots and cell mixture plots for all slices and cluster mixture plots and boxplots for each slice,  # noqa
    # 4 - additionally save cell type images, abundance plots and cell percentage table for each slice, 5 - additionally save color plots.
    'plotting': 5,
    # Project name that is used to name a directory containing all the slices used.
    'project_name': "community",
    # Skip statistics calculation on cell community clustering result. A table of cell mixtures and comparative spatial plots of cell types and mixtures will not be created.
    'skip_stats': False,
    # Total number of cells per window mixture after normalization.
    'total_cell_norm': 10000,
    # Rate by which the binary image of cells is downsampled before calculating the entropy and scatteredness metrics.
    # If no value is provided, downsample_rate will be equal to 1/2 of minimal window size.
    'downsample_rate': None,
    # Number of threads that will be used to speed up community calling.
    'num_threads': 5,
    # Threshold value for spatial cell type entropy for filtering out overdispersed cell types.
    'entropy_thres': 1.0,
    # Threshold value for spatial cell type scatteredness for filtering out overdispersed cell types.
    'scatter_thres': 1.0,
    # Comma separated list of window sizes for analyzing the cell community.
    'win_sizes': 'NA',
    # Comma separated list of sliding steps for sliding window.
    'sliding_steps': 'NA',
    # Minimum number of cell for cluster to be plotted in plot_stats().
    'min_cluster_size': 200,
    # Minimum percentage of cell type in cluster for cell type to be plotted in plot_stats().
    'min_perc_to_show': 4,
    # Minimum number of cell types that have more than `min_perc_celltype` in a cluster, for a cluster to be shown in plot_celltype_table().
    'min_num_celltype': 1,
    # Minimum percentage of cells of a cell type which at least min_num_celltype cell types need to have to show a cluster in plot_celltype_table().
    'min_perc_celltype': 10,
    # Multiple od standard deviations from mean values where the cutoff for m.
    'min_cells_coeff': 1.5,
    # Color system for display of cluster specific windows.
    'color_plot_system': 'rgb',
    # Save adata file with resulting .obs column of cell community labels.
    'save_adata': False,
    # Minimum number of cells per cell type needed to use the cell type for cell communities extraction (in percentages).
    'min_count_per_type': 0.1,
    # Stop plots from displaying in notebooks or standard ouput. Used for batch processing.
    'hide_plots': True,
    # DPI (dots per inch) used for plotting figures.
    'dpi': 100,
    # Whether in development mode
    'dev': False
}

# index to find community id in figure names for html report
BOXPLT_C_INDEX = 1
COLORPLT_C_INDEX = 2
CMIXT_C_INDEX = -1

CT_COLORPLT_INDEX = -3
