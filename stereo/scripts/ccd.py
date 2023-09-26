import argparse
import os
from typing import Any

from stereo.algorithm.ccd.constants import COMMUNITY_DETECTION_DEFAULTS
from stereo.algorithm.community_detection import _CommunityDetection
from stereo.io.reader import read_h5ad


class Params:
    def __init__(self):
        self.params = COMMUNITY_DETECTION_DEFAULTS

    def __setattr__(self, __name: str, __value: Any):
        if __name not in COMMUNITY_DETECTION_DEFAULTS:
            self.__dict__[__name] = __value
        else:
            self.params[__name] = __value


def get_args():
    parser = argparse.ArgumentParser(description='Cell Community Detection')
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+',
                        help='A h5ad file, space separated list of h5ad files or path of directory contains some h5ad files.')  # noqa
    parser.add_argument('-a', '--annotation', required=True, type=str, help='The key specified the cell type in obs.')
    parser.add_argument('-tf', '--tfile', required=False, type=str, default=None,
                        help='File path to Anndata object with calculated cell mixtures for data windows, output of calc_feature_matrix.')  # noqa
    parser.add_argument('-o', '--out_path', required=False, type=str, default='./results',
                        help="Absolute path to store outputs, default to './results'.")
    parser.add_argument('-ca', '--cluster_algo', required=False, type=str, default='leiden',
                        help='Clustering algorithm, default to leiden.')
    parser.add_argument('-r', '--resolution', required=False, type=float, default=0.2,
                        help='Resolution of leiden clustering algorithm. Ignored for spectral and agglomerative, default to 0.2.')  # noqa
    parser.add_argument('-c', '--n_clusters', required=False, type=int, default=10,
                        help='Number of clusters for spectral and agglomerative clustering, ignored for leiden, default to 10.')  # noqa
    parser.add_argument('-sp', '--spot_size', required=False, type=int, default=30,
                        help='Size of the spot on plot, default to 30.')
    parser.add_argument('-vb', '--verbose', required=False, type=int, default=0,
                        help='Show logging messages. 0 - Show warnings, >0 show info, default to 0.')
    parser.add_argument(
        '-p', '--plotting',
        required=False, type=int, default=5,
        help='''
            Save plots flag, default to 5, available values include
            0 - No plotting and saving;
            1 - save clustering plot;
            2 - additionally save plots of cell type images statistics and cell mixture plots;
            3 - additionally save cell and cluster abundance plots and cell mixture plots for all slices and cluster mixture plots and boxplots for each slice; 
            4 - additionally save cell type images, abundance plots and cell percentage table for each slice;
            5 - additionally save color plots.
        '''  # noqa
    )
    parser.add_argument('-j', '--project_name', required=False, type=str, default='community',
                        help='Project name that is used to name a directory containing all the slices used, default to community.')  # noqa
    parser.add_argument('-sk', '--skip_stats', required=False, type=bool, default=False,
                        help='Skip statistics calculation on cell community clustering result.A table of cell mixtures and comparative spatial plots of cell types and mixtures will not be created, default to False.')  # noqa
    parser.add_argument('-t', '--total_cell_norm', required=False, type=int, default=10000,
                        help='Total number of cells per window mixture after normalization, default to 10000.')
    parser.add_argument('-d', '--downsample_rate', required=False, type=float, default=None,
                        help='Rate by which the binary image of cells is downsampled before calculating the entropy and scatteredness metrics.If no value is provided, downsample_rate will be equal to 1/2 of minimal window size, default to None.')  # noqa
    parser.add_argument('-th', '--num_threads', required=False, type=int, default=5,
                        help='Number of threads that will be used to speed up community calling, default to 5.')
    parser.add_argument('-et', '--entropy_thres', required=False, type=float, default=1.0,
                        help='Threshold value for spatial cell type entropy for filtering out overdispersed cell types, default to 1.0.')  # noqa
    parser.add_argument('-st', '--scatter_thres', required=False, type=float, default=1.0,
                        help='Threshold value for spatial cell type scatteredness for filtering out overdispersed cell types, defaykt to 1.0.')  # noqa
    parser.add_argument('-w', '--win_sizes', required=False, type=str, default='NA',
                        help='Comma separated list of window sizes for analyzing the cell community.')
    parser.add_argument('-s', '--sliding_steps', required=False, type=str, default='NA',
                        help='Comma separated list of sliding steps for sliding window.')
    parser.add_argument('-cs', '--min_cluster_size', required=False, type=int, default=200,
                        help='Minimum number of cell for cluster to be plotted in plot_stats(), default to 200.')
    parser.add_argument('-ps', '--min_perc_to_show', required=False, type=float, default=4,
                        help='Minimum percentage of cell type in cluster for cell type to be plotted in plot_stats(), default to 4.')  # noqa
    parser.add_argument('-nc', '--min_num_celltype', required=False, type=int, default=1,
                        help='Minimum number of cell types that have more than `min_perc_celltype` in a cluster, for a cluster to be shown in plot_celltype_table(), default to 1.')  # noqa
    parser.add_argument('-pc', '--min_perc_celltype', required=False, type=float, default=10,
                        help='Minimum percentage of cells of a cell type which at least min_num_celltype cell types need to have to show a cluster in plot_celltype_table().')  # noqa
    parser.add_argument('-cf', '--min_cells_coeff', required=False, type=float, default=1.5,
                        help='Multiple od standard deviations from mean values where the cutoff for m, default to 1.5.')
    parser.add_argument('-cp', '--color_plot_system', required=False, type=str, default='rgb',
                        help='Color system for display of cluster specific windows, default rgb.')
    parser.add_argument('-sd', '--save_adata', required=False, type=bool, default=False,
                        help='Save adata file with resulting .obs column of cell community labels, default to False.')
    parser.add_argument('-cpt', '--min_count_per_type', required=False, type=float, default=0.1,
                        help='Minimum number of cells per cell type needed to use the cell type for cell communities extraction (in percentages), default to 0.1.')  # noqa
    parser.add_argument('-dpi', '--dpi', required=False, type=int, default=100,
                        help='DPI (dots per inch) used for plotting figures, default to 100.')

    args = parser.parse_args(namespace=Params())
    return args


def main():
    args = get_args()
    input = args.input
    annotation = args.annotation
    params = args.params

    if len(input) == 1:
        if os.path.isdir(input[0]):
            input_dir = input[0]
            input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('h5ad')]
        else:
            input_files = input
    else:
        input_files = input

    data_list = [read_h5ad(f) for f in input_files]
    ccd = _CommunityDetection()
    ccd._main(data_list, annotation=annotation, **params)


if __name__ == '__main__':
    main()
