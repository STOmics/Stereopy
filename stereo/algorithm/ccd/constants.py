COMMUNITY_DETECTION_DEFAULTS = {
    'tfile': None,
    'out_path': "results",
    'cluster_algo': "leiden",
    'resolution': 0.2,
    'n_clusters': 10,
    'spot_size': 30,
    'verbose': 0,
    'plotting': 2,
    'project_name': "community",
    'skip_stats': False,
    'total_cell_norm': 10000,
    'downsample_rate': 50,
    'num_threads': 5,
    'entropy_thres': 1.0,
    'scatter_thres': 1.0,
    'win_sizes': 'NA',
    'sliding_steps': 'NA',
    'min_cluster_size': 200,
    'min_perc_to_show': 4,
    'min_num_celltype': 1,
    'min_perc_celltype': 10,
    'min_cells_coeff': 1.5,
    'color_plot_system': 'rgb',
    'save_adata': False,
    'min_count_per_type': 0.1,
    'hide_plots': False,
    'dpi': 100
}

# index to find community id in figure names for html report
BOXPLT_C_INDEX = 1
COLORPLT_C_INDEX = 2
CMIXT_C_INDEX = -1

CT_COLORPLT_INDEX = -3
