import os
import time
from collections import defaultdict
from functools import reduce
from itertools import cycle
from typing import List

import anndata as ad
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.log_manager import logger
from .ccd import (
    timeit,
    plot_spatial,
    generate_report,
    set_figure_params,
    reset_figure_params,
    calculate_spatial_metrics,
    SlidingWindowMultipleSizes,
    COMMUNITY_DETECTION_DEFAULTS,
)
from .ccd.community_clustering_algorithm import cluster_palette


class _CommunityDetection:
    """
    Class for performing community detection on a set of slices.
    """

    def params_init(
            self,
            slices: List[AnnBasedStereoExpData],
            annotation: str,
            **kwargs
    ) -> None:
        """
        Initialize the CommunityDetection object's params.

        Example:
            Download data (~700MB) with command:
                wget https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E16.5_E1S3_cell_bin_whole_brain.h5ad

            Execute:
                from stereo.core.stereo_exp_data import AnnBasedStereoExpData
                from community_detection import CommunityDetection
                adata = AnnBasedStereoExpData('data.h5ad')
                cd = CommunityDetection([adata], 'sim anno') # The algorithm can also be run for multiple slices.
                cd.main()

        Parameters:
        - slices (List[AnnBasedStereoExpData]): A list of AnnBasedStereoExpData objects representing the slices of a tissue. # noqa
        - annotation (str): The annotation string.
        - **kwargs: Additional keyword arguments (Check constants.py for the description of additional arguments)
        """
        self.params = {**COMMUNITY_DETECTION_DEFAULTS, **kwargs}
        self.params['annotation'] = annotation
        self.slices = slices
        self.cell_types = set([])
        self.annotation_palette = None
        missing_cell_type_palette = False
        for slice in self.slices:
            # check for spatial data stored under labels 'X_spatial' and 'spatial_stereoseq'
            if 'X_spatial' in slice._ann_data.obsm:
                slice._ann_data.obsm['spatial'] = slice._ann_data.obsm['X_spatial'].copy()
            elif 'spatial_stereoseq' in slice._ann_data.obsm:
                slice._ann_data.obsm['spatial'] = np.array(slice._ann_data.obsm['spatial_stereoseq'].copy())
            # annotation data must be of string type
            # slice._ann_data.obs[annotation] = slice._ann_data.obs[annotation].astype('str')
            # create a set of existing cell types in all slices
            self.cell_types = self.cell_types.union(set(slice._ann_data.obs[annotation].unique()))
            # if any of the samples lacks the cell type palette, set the flag
            if f'{annotation}_colors' not in slice._ann_data.uns_keys():
                missing_cell_type_palette = True

        self.cell_types = list(sorted(self.cell_types))

        self.file_names = [fname for fname in self.params['files'].split(',')] if 'files' in self.params else [
            f"Slice_{id}" for id in range(len(slices))]
        if self.params['win_sizes'] == 'NA' or self.params['sliding_steps'] == 'NA':
            logger.info(
                "Window sizes and/or sliding steps not provided by user - proceeding to calculate optimal values")
            self.params['win_sizes'], self.params['sliding_steps'] = self.calc_optimal_win_size_and_slide_step()
        else:
            self.log_win_size_full_info()

        # if downsample_rate is not provided, use the dimension of the smallest window size as reference
        if self.params['downsample_rate'] == None:  # noqa
            logger.info(
                "Downsample rate is not provided by user - proceeding to calculate one based on minimal window size.")
            min_win_size = np.min([int(i) for i in self.params['win_sizes'].split(',')])
            self.params['downsample_rate'] = min_win_size // 2
            logger.info(f"donwsample_rate = {self.params['downsample_rate']}")

        # if cell type palette is not available, create and add to each slice anndata object
        if missing_cell_type_palette:
            self.annotation_palette = {cellt: cluster_palette[-id - 1] for id, cellt in enumerate(self.cell_types)}
            for slice in self.slices:
                slice._ann_data.uns[f'{annotation}_colors'] = [self.annotation_palette[cellt] for cellt in
                                                               list(sorted(slice._ann_data.obs[annotation].unique()))]

    @timeit
    def _main(self, slices, annotation="annotation", **kwargs):
        """
        Executes the community detection algorithm.

        This method performs community detection using the specified parameters and data slices. It follows the following steps:

        1. Data Processing Loop:
        - Iterates over each data slice, identified by `slice_id` and the corresponding file name.
        - Initializes a `SlidingWindowMultipleSizes` algorithm object, `algo`, with the slice and other parameters.
        - Optionally plots the original annotation if the plotting level is greater than 0.
        - Runs the algorithm for feature extraction and cell type filtering based on entropy and scatteredness.
        - Optionally plots the histogram of cell sum per window if the plotting level is greater than 1.
        - Calculates entropy, scatteredness, and cell type images using the `calculate_spatial_metrics` function.
        - Optionally plots binary images of cell types' spatial positions if the plotting level is greater than 3.
        - Filters out cell types that are not localized based on calculated metrics.
        - Appends the `algo` object to the `algo_list`.

        2. Annotation Plotting:
        - If the plotting level is greater than 0 and there are multiple algorithm objects in `algo_list`, plots the annotation for all slices together.

        3. Tissue Merging:
        - Merges the tissue Anndata objects from all algorithm objects into a single Anndata object, `merged_tissue`.

        4. Clustering:
        - Performs clustering on the merged tissue using the specified clustering algorithm.

        5. Algorithm Execution:
        - Performs community calling using the majority voting method on each algorithm object.
        - Saves the Anndata objects, community labels, and tissue data for further use.
        - Optionally plots the clustering results if the plotting level is greater than 0.
        - If the `skip_stats` flag is not active, calculates cell mixture statistics, saves them, and generates corresponding plots.
        - Optionally saves the final tissue with statistics.

        6. Clustering Plotting:
        - If the plotting level is greater than 0 and there are multiple algorithm objects in `algo_list`, plots the clustering results for all slices together.

        7. Additional Plots:
        - Generates additional plots based on the plotting level and the data from `algo_list`. These include cell type mixtures, cell abundance, and cluster abundance plots.

        8. Report Generation:
        - Generates a HTML report o the results.
        """  # noqa
        # only
        self.params_init(slices, annotation, **kwargs)

        start_time = time.perf_counter()

        if not os.path.exists(self.params['out_path']):
            os.makedirs(self.params['out_path'])

        algo_list = []
        win_sizes = "_".join([i for i in self.params['win_sizes'].split(',')])
        sliding_steps = "_".join([i for i in self.params['sliding_steps'].split(',')])
        self.params['project_name_orig'] = self.params['project_name']
        self.params['out_path_orig'] = self.params['out_path']
        cluster_string = f"_r{self.params['resolution']}" if self.params[
                                                                 'cluster_algo'] == 'leiden' else f"_nc{self.params['n_clusters']}"  # noqa
        self.params[
            'project_name'] += f"_c{self.params['cluster_algo']}{cluster_string}_ws{win_sizes}_ss{sliding_steps}_sct{self.params['scatter_thres']}_dwr{self.params['downsample_rate']}_mcc{self.params['min_cells_coeff']}"  # noqa
        self.params['out_path'] = os.path.join(self.params['out_path'], self.params['project_name'])

        if not os.path.exists(self.params['out_path']):
            os.makedirs(self.params['out_path'])

        for slice_id, (slice, file) in enumerate(zip([slice._ann_data for slice in self.slices], self.file_names)):
            slice.uns['slice_id'] = slice_id

            algo = SlidingWindowMultipleSizes(slice, slice_id, file, **self.params)
            # plot original annotation
            if self.params['plotting'] > 0:
                algo.plot_annotation()
            # run algorithm for feature extraction and cell type filtering based on entropy and scatteredness
            algo.run()
            if self.params['plotting'] > 1:
                algo.plot_histogram_cell_sum_window()
            # CELL TYPE FILTERING
            # [NOTE] This is not valid for multislice. A consensus on removing a cell type must
            # be made for all slices before removing it from any slice.
            # here I have tissue, I want to calculate entropy and scatteredness for each cell type in adata
            # and based on this information remove certain cell types
            entropy, scatteredness, cell_type_images = \
                calculate_spatial_metrics(algo.adata, algo.unique_cell_type, algo.downsample_rate, algo.annotation)
            # init var layer of tissue anndata object
            algo.tissue.var = algo.tissue.var.copy()
            algo.tissue.var.loc[:, 'entropy'] = entropy.loc[algo.tissue.var.index]
            algo.tissue.var.loc[:, 'scatteredness'] = scatteredness.loc[algo.tissue.var.index]
            algo.tissue.uns['cell_t_images'] = cell_type_images
            # save a .csv file with metrics per cell type
            algo.save_metrics()
            # plot binary images of cell types spatial positions
            if self.params['plotting'] > 3:
                algo.plot_celltype_images()
            # filter the cell types which are not localized using calculated metrics (entropy and scatteredness)
            algo.cell_type_filtering()
            # check if cell type filtering removes all cell types and raise an Error
            if algo.tissue.n_vars == 0:
                raise ValueError(r'Empty algo.tissue object. All cell types removed. Adjust scatter_thres and '
                                 r'entropy_thres to preserve useful cell types.')

            # add algo object for each slice to a list
            algo_list.append(algo)
        self.algo_list = algo_list
        if self.params['plotting'] > 0 and len(algo_list) > 1:
            self.plot_all_annotation()

        # MERGE TISSUE ANNDATA
        # each tissue has slice_id as 3rd coordinate in tissue.obsm['spatial']
        merged_tissue = ad.concat([a.get_tissue() for a in algo_list], axis=0, join='outer')
        # if tissues have different sets of cell those columns are filled with NANs
        # this is corrected by writing 0s
        merged_tissue.X[np.isnan(merged_tissue.X)] = 0.0

        # CLUSTERING (WINDOW_LABELS)
        self.cluster(merged_tissue)

        for slice_id, algo in enumerate(algo_list):
            # extract clustering data from merged_tissue
            algo.set_clustering_labels(
                merged_tissue.obs.loc[merged_tissue.obsm['spatial'][:, 2] == slice_id, self.params['cluster_algo']])

            # COMMUNITY CALLING (MAJORITY VOTING)
            algo.community_calling()
            # copy final Cell Community Detection (CCD) result to original slices
            self.slices[slice_id]._ann_data.obs.loc[
                algo.adata.obs[f'tissue_{algo.method_key}'].index, 'cell_communities'] = algo.adata.obs[
                f'tissue_{algo.method_key}']
            if np.nan in self.slices[slice_id]._ann_data.obs['cell_communities'].values:
                if 'unknown' not in self.slices[slice_id]._ann_data.obs['cell_communities'].cat.categories:
                    self.slices[slice_id]._ann_data.obs['cell_communities'] = self.slices[slice_id]._ann_data.obs[
                        'cell_communities'].cat.add_categories('unknown')
                self.slices[slice_id]._ann_data.obs['cell_communities'].fillna('unknown', inplace=True)
            # self.slices[slice_id]._ann_data.obs['cell_communities'].fillna('unknown', inplace=True)

            # save anndata objects for further use
            if self.params['save_adata']:
                algo.save_anndata()
            algo.save_community_labels()
            algo.save_tissue()

            # PLOT COMMUNITIES & STATISTICS
            # plot cell communities clustering result
            if self.params['plotting'] > 0:
                algo.plot_clustering()

            # if flag skip_stats is active, skip cell mixture statistics analysis
            if not self.params['skip_stats']:
                algo.calculate_cell_mixture_stats()
                algo.save_mixture_stats()
                if self.params['plotting'] > 1:
                    try:
                        algo.plot_stats()
                    except Exception as e:
                        print('plot_stats raise exception while running multi slice, err=%s' % str(e))
                    try:
                        algo.plot_celltype_table()
                    except Exception as e:
                        print('plot_celltype_table raise exception while running multi slice, err=%s' % str(e))
                if self.params['plotting'] > 2:
                    algo.plot_cluster_mixtures()
                    algo.boxplot_stats()
                if self.params['plotting'] > 4:
                    algo.colorplot_stats(color_system=self.params['color_plot_system'])
                    algo.colorplot_stats_per_cell_types()
                # save final tissue with stats
                algo.save_tissue(suffix='_stats')

        if self.params['plotting'] > 0 and len(algo_list) > 1:
            self.plot_all_clustering()
        if self.params['plotting'] > 2:
            self.plot_celltype_mixtures_total([algo.get_cell_mixtures().to_dict() for algo in algo_list])
            self.plot_cell_abundance_total()
            self.plot_cluster_abundance_total()
        if self.params['plotting'] > 3:
            self.plot_cell_abundance_per_slice()
            self.plot_cluster_abundance_per_slice()
            self.plot_cell_perc_in_community_per_slice()

        end_time = time.perf_counter()

        self.params['execution_time'] = end_time - start_time
        generate_report(self.params)
        reset_figure_params()

    @timeit
    def cluster(self, merged_tissue):  # TODO, merged_tissue da bude AnnBasedStereoExpData
        """
        Perform clustering on merged tissue data from all slices.
        Supported clustering algorithms are:
        'leiden' - Leiden (stereopy) with neighbors similarity metric,
        'spectral' - Spectral (skimage) with neighbors similarity metric, and
        'agglomerative' - Agglomerative (skimage) with 'ward' linkage type
        and 'euclidian' distance metric.
        Cluster labels are stored in merged_tissue.obs[cluster_algo]
        and updated inplace.

        Parameters:
        - merged_tissue (AnnBasedStereoExpData): AnnBasedStereoExpData object containin features of all slices

        """
        merged_tissue = AnnBasedStereoExpData(h5ad_file_path=None, based_ann_data=merged_tissue)
        if self.params['cluster_algo'] == 'leiden':
            merged_tissue._ann_data.obsm['X_pca_dummy'] = merged_tissue._ann_data.X
            merged_tissue.tl.neighbors(pca_res_key='X_pca_dummy', n_neighbors=15,
                                       n_jobs=-1, res_key='neighbors')
            merged_tissue.tl.leiden(neighbors_res_key='neighbors', res_key='leiden',
                                    resolution=self.params['resolution'])
            merged_tissue._ann_data.obs['leiden'] = merged_tissue._ann_data.obs['leiden'].astype('int')
            merged_tissue._ann_data.obs['leiden'] -= 1
            merged_tissue._ann_data.obs['leiden'] = merged_tissue._ann_data.obs['leiden'].astype('str')
            # merged_tissue._ann_data.obs['leiden'] = merged_tissue._ann_data.obs['leiden'].astype('category')
        elif self.params['cluster_algo'] == 'spectral':
            merged_tissue._ann_data.obsm['X_pca_dummy'] = merged_tissue._ann_data.X
            merged_tissue.tl.neighbors(pca_res_key='X_pca_dummy', n_neighbors=15)
            spcl = SpectralClustering(n_clusters=self.params['n_clusters'], eigen_solver='arpack', random_state=0,
                                      affinity='precomputed', n_jobs=5)
            merged_tissue._ann_data.obs[self.params['cluster_algo']] = (
                spcl.fit_predict(merged_tissue._ann_data.obsp['connectivities'])).astype('str')
        elif self.params['cluster_algo'] == 'agglomerative':
            ac = AgglomerativeClustering(n_clusters=self.params['n_clusters'], affinity='euclidean',
                                         compute_full_tree=False, linkage='ward', distance_threshold=None)
            merged_tissue._ann_data.obs[self.params['cluster_algo']] = (
                ac.fit_predict(merged_tissue._ann_data.X)).astype('str')
        else:
            logger.error('Unsupported clustering algorithm')
            raise ValueError("Unsupported clustering algorithm")

    def log_win_size_full_info(self):
        for slice, fname in zip([slice._ann_data for slice in self.slices], self.file_names):
            x_min, x_max = np.min(slice.obsm['spatial'][:, 0]), np.max(slice.obsm['spatial'][:, 0])
            y_min, y_max = np.min(slice.obsm['spatial'][:, 1]), np.max(slice.obsm['spatial'][:, 1])
            x_range, y_range = abs(abs(x_max) - abs(x_min)), abs(abs(y_max) - abs(y_min))
            for win_size, slide_step in zip([int(w) for w in self.params['win_sizes'].split(',')],
                                            [int(s) for s in self.params['sliding_steps'].split(',')]):
                self.log_win_size_info_per_slice(slice, fname, win_size, slide_step, x_range, y_range)

    def log_win_size_info_per_slice(self, slice, fname, win_size, slide_step, x_range, y_range):
        """
        Logs window size information for a given slice.

        Parameters:
        - slice: The slice.
        - fname: The filename of the slice.
        - win_size: The size of the window.
        - slide_step: The sliding step for the window.
        - x_range: The range of x-coordinates.
        - y_range: The range of y-coordinates.

        """
        cell_to_loc = defaultdict(int)
        for x, y in slice.obsm['spatial']:
            cell_to_loc[(x // win_size, y // win_size)] += 1

        logger.info(f"""Window size info for slice: {fname}
                     window size: {win_size}
                     sliding step: {slide_step}
                     cells mean: {np.mean(list(cell_to_loc.values())):.2f}
                     cells median: {np.median(list(cell_to_loc.values()))}
                     num horizontal windows: {int(x_range // win_size)}
                     num vertical windows: {int(y_range // win_size)}\n
                     """)

    @timeit
    def calc_optimal_win_size_and_slide_step(self):
        """
        Method for calculating the optimal window size and sliding step.
        Window size is calculated such that it covers between MIN_COVERED and MAX_COVERED cells.
        Sliding step is set to the half of the window size.

        """
        MAX_ITERS = 10
        MIN_COVERED = 30
        MAX_COVERED = 60
        AVG_COVERED_GOAL = (MAX_COVERED + MIN_COVERED) // 2

        x_min, x_max = np.min(self.slices[0]._ann_data.obsm['spatial'][:, 0]), np.max(
            self.slices[0]._ann_data.obsm['spatial'][:, 0])
        y_min, y_max = np.min(self.slices[0]._ann_data.obsm['spatial'][:, 1]), np.max(
            self.slices[0]._ann_data.obsm['spatial'][:, 1])
        x_range, y_range = abs(abs(x_max) - abs(x_min)), abs(abs(y_max) - abs(y_min))

        win_size = int(x_range // 50 if x_range < y_range else y_range // 50)
        delta_multiplier = win_size * 0.15

        avg_covered = -1
        delta = -1
        iters = 0
        while iters < MAX_ITERS:
            cell_to_loc = defaultdict(int)
            for x, y in self.slices[0]._ann_data.obsm['spatial']:
                cell_to_loc[(x // win_size, y // win_size)] += 1

            # using median instead of mean because many windows can be empty (space is not fully occupied by tissue)
            avg_covered = np.median(list(cell_to_loc.values()))

            if MIN_COVERED < avg_covered < MAX_COVERED:
                break

            delta = np.sign(AVG_COVERED_GOAL - avg_covered) * (
                AVG_COVERED_GOAL / avg_covered if AVG_COVERED_GOAL > avg_covered else avg_covered / AVG_COVERED_GOAL
            ) * delta_multiplier
            win_size += delta
            iters += 1

        # closest doubly even number so that sliding step is also even number
        win_size = round(win_size)
        win_size = win_size + ((win_size & 0b11) ^ 0b11) + 1 if win_size & 0b11 else win_size

        if iters == MAX_ITERS:
            logger.warning(f"Optimal window size not obtained in {MAX_ITERS} iterations.")
        self.log_win_size_info_per_slice(self.slices[0]._ann_data, self.file_names[0], win_size, win_size // 2, x_range,
                                         y_range)

        return (str(win_size), str(win_size // 2))

    def plot_all_slices(self, img_name, clustering=False):
        """
        Plot all slices using the specified algorithms and annotations.

        Parameters:
        - img_name (str): The name of the output image file.
        - clustering (bool, optional): Whether to plot clustering or cell type annotation. Defaults to False.

        """
        number_of_samples = len(self.algo_list)
        number_of_rows = 2 if number_of_samples % 2 == 0 and number_of_samples > 2 else 1
        number_of_columns = (
                number_of_samples // 2) if number_of_samples % 2 == 0 and number_of_samples > 2 else number_of_samples

        figure, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, squeeze=False, layout='constrained',
                                    figsize=(10, 6))
        h_d = {}
        unknown_label = []
        for (algo, ax) in zip(self.algo_list, axes.flatten()):
            palette = algo.cluster_palette if clustering else algo.annotation_palette
            annotation = f'tissue_{self.algo_list[0].method_key}' if clustering else self.algo_list[0].annotation
            clusters = np.unique(algo.adata.obs[annotation].values)
            if len(clusters) > len(cluster_palette):
                logger.warning(f"Number of clusters ({len(clusters)}) is larger than pallette size. All clusters "
                               f"will be colored gray.")
            plot_spatial(algo.adata, annotation=annotation, palette=palette, spot_size=algo.spot_size, ax=ax)
            ax.get_legend().remove()
            ax.set_title(f'{algo.filename}', fontsize=6, loc='center', wrap=True)
            hands, labs = ax.get_legend_handles_labels()
            for h, l in zip(hands, labs):
                h._sizes = [11]
                if l == 'unknown':
                    unknown_label = np.array([[h, l]])
                    continue
                if l not in h_d.values():
                    h_d[h] = l
        try:
            handles = np.array([[h, int(l)] for (h, l) in h_d.items()])
        except Exception:
            handles = np.array([[h, l] for (h, l) in h_d.items()])

        handles = handles[handles[:, 1].argsort()]
        handles[:, 1] = handles[:, 1].astype('str')

        if len(unknown_label) > 0:
            handles = np.concatenate((handles, unknown_label), axis=0)

        legend_ncols = 1 if len(handles) <= 12 else 2
        figure.legend(handles[:, 0], handles[:, 1], bbox_to_anchor=(1.15, 0.5), loc='center', fontsize=4, frameon=False,
                      borderaxespad=0., ncol=legend_ncols, labelspacing=1)
        figure.savefig(f'{self.params["out_path"]}/{img_name}', dpi=self.params['dpi'], bbox_inches='tight')
        if not self.params['hide_plots']:
            plt.show()
        plt.close()

    @timeit
    def plot_all_annotation(self):
        self.plot_all_slices('cell_type_per_slice.png')

    @timeit
    def plot_all_clustering(self):
        self.plot_all_slices('clustering_per_slice.png', True)

    @timeit
    def plot_celltype_mixtures_total(self, cell_mixtures):
        """
        Plot the total cell type mixtures.

        Parameters:
        - cell_mixtures (list): A list of dictionaries containing cell type mixtures.

        """

        def merge_dicts(dict1, dict2):
            return {cluster: dict1.get(cluster, 0) + dict2.get(cluster, 0) for cluster in set(dict1) | set(dict2)}

        def merge_dicts_of_dicts(dict1, dict2):
            return {celltype: merge_dicts(dict1.get(celltype, {}), dict2.get(celltype, {})) for celltype in
                    set(dict1) | set(dict2)}

        total_dict = reduce(merge_dicts_of_dicts, cell_mixtures)
        total = pd.DataFrame(total_dict).fillna(0)

        total['total_counts'] = np.array([sum(total.loc[row, :]) for row in total.index]).astype(int)

        cell_type_counts = {ct: [int(sum(total[ct]))] for ct in total.columns}
        total = pd.concat([total, pd.DataFrame(cell_type_counts, index=['total_cells'])])

        total.iloc[:-1, :-1] = total.iloc[:-1, :-1].div(total['total_counts'][:-1], axis=0).mul(100)
        total['perc_of_all_cells'] = np.around(total['total_counts'] / total['total_counts'][-1] * 100, decimals=1)
        total = total.loc[sorted(total.index.values, key=lambda x: float(x) if x != "total_cells" else float('inf'))]

        set_figure_params(dpi=self.params['dpi'], facecolor='white')
        sns.set(font_scale=1.5)

        ncols = len(total.columns)
        fig, axes = plt.subplots(ncols=ncols, figsize=(30, 20))
        fig.subplots_adjust(wspace=0)

        vmax_perc = np.max(np.ravel(total.iloc[:-1, :-2]))
        for i, ax in enumerate(axes[:-2]):
            sns.heatmap(pd.DataFrame(total.iloc[:, i]), vmin=0.0, vmax=vmax_perc, linewidths=0, linecolor=None,
                        annot=True, cbar=False, ax=ax, cmap="Greys", fmt='4.0f', xticklabels=True,
                        yticklabels=True if i == 0 else False, square=True)
        sns.heatmap(pd.DataFrame(total.iloc[:, -2]), annot=True, vmin=0, vmax=np.max(total.iloc[:-1, -2]), linewidths=0,
                    linecolor=None, cbar=False, cmap='Greens', ax=axes[-2], fmt='4.0f', xticklabels=True,
                    yticklabels=False, square=True)
        sns.heatmap(pd.DataFrame(total.iloc[:, -1]), annot=True, vmin=0, vmax=np.max(total.iloc[:-1, -1]), linewidths=0,
                    linecolor=None, cbar=False, cmap='Greens', ax=axes[-1], fmt='4.0f', xticklabels=True,
                    yticklabels=False, square=True)

        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
            ax.xaxis.tick_top()

        plt.savefig(os.path.join(self.params['out_path'], 'total_cell_mixtures_table.png'), bbox_inches='tight')
        if not self.params['hide_plots']:
            plt.show()
        plt.close()

    @timeit
    def plot_cell_perc_in_community_per_slice(self):
        """
        Plots the percentage of cells in each community per slice.
        """
        cells_in_comm_per_slice = {
            algo.filename: algo.get_community_labels().value_counts(normalize=True).rename(algo.filename) for algo in
            self.algo_list}
        df = pd.concat(cells_in_comm_per_slice.values(), axis=1).fillna(0).mul(100).T
        df = df[sorted(df.columns.values, key=lambda x: float(x) if x != "unknown" else float('inf'))]
        set_figure_params(dpi=self.params['dpi'], facecolor='white')
        sns.set(font_scale=1.5)
        plt.figure(figsize=(30, 20))

        ax = sns.heatmap(df, annot=True, fmt="4.0f", cmap="Greys", xticklabels=True, yticklabels=True, square=True,
                         cbar=False)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        plt.savefig(os.path.join(self.params['out_path'], 'cell_perc_in_community_per_slice.png'), bbox_inches='tight')
        if not self.params['hide_plots']:
            plt.show()
        plt.close()

    @timeit
    def plot_cell_abundance_total(self):
        """
        Plots the total cell abundance for each algorithm.
        """
        fig, ax = plt.subplots(figsize=(20, 20))
        fig.subplots_adjust(wspace=0)
        set_figure_params(dpi=self.params['dpi'], facecolor='white')

        greys = cycle(['darkgray', 'gray', 'dimgray', 'lightgray'])
        colors = [next(greys) for _ in range(len(self.algo_list))]
        cell_percentage_dfs = []
        plot_columns = []
        for algo in self.algo_list:
            cell_percentage_dfs.append(pd.DataFrame(
                algo.get_adata().obs[algo.annotation].value_counts(normalize=True).mul(100).rename(algo.filename)))
            plot_columns.append(algo.filename)

        cummulative_df = pd.concat(cell_percentage_dfs, axis=1).fillna(0)
        cummulative_df.plot(y=plot_columns, kind="bar", rot=70, ax=ax, xlabel="", color=colors)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.grid(False)
        ax.set_facecolor('white')
        plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

        plt.savefig(os.path.join(self.params['out_path'], 'cell_abundance_all_slices.png'), bbox_inches='tight')
        if not self.params['hide_plots']:
            plt.show()
        plt.close()

    @timeit
    def plot_cell_abundance_per_slice(self):
        """
        Plots the cell abundance for each algorithm per slice.
        """
        number_of_samples = len(self.algo_list)
        if number_of_samples <= 2:
            number_of_rows = 1
            number_of_columns = number_of_samples
        else:
            number_of_rows = 2 if number_of_samples % 2 == 0 else 1
            number_of_columns = number_of_samples // 2 if number_of_samples % 2 == 0 else number_of_samples
        fig, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, figsize=(20, 20), squeeze=False)
        axes = axes.ravel()
        fig.subplots_adjust(wspace=0)
        set_figure_params(dpi=self.params['dpi'], facecolor='white')

        cell_percentage_dfs = []
        plot_columns = []
        for algo in self.algo_list:
            cell_percentage_dfs.append(pd.DataFrame(
                algo.get_adata().obs[algo.annotation].value_counts(normalize=True).mul(100).rename(algo.filename)))
            plot_columns.append(algo.filename)

        cummulative_df = pd.concat(cell_percentage_dfs, axis=1).fillna(0)

        for i in range(number_of_rows * number_of_columns):
            axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            axes[i].set_facecolor('white')
            axes[i].set_title(plot_columns[i])
            cummulative_df.plot(y=plot_columns[i], kind="bar", rot=70, ax=axes[i], xlabel="", color="grey",
                                legend=False)
            axes[i].grid(False)

        for ax in axes:
            ax.grid(False)

        plt.savefig(os.path.join(self.params['out_path'], 'cell_abundance_per_slice.png'), bbox_inches='tight')
        if not self.params['hide_plots']:
            plt.show()
        plt.close()

    @timeit
    def plot_cluster_abundance_total(self):
        """
        Plots the total cluster abundance for each algorithm.
        """
        fig, ax = plt.subplots(figsize=(20, 20))
        fig.subplots_adjust(wspace=0)
        set_figure_params(dpi=self.params['dpi'], facecolor='white')

        greys = cycle(['darkgray', 'gray', 'dimgray', 'lightgray'])
        colors = [next(greys) for _ in range(len(self.algo_list))]
        cell_percentage_dfs = []
        plot_columns = []
        for algo in self.algo_list:
            cell_percentage_dfs.append(pd.DataFrame(
                algo.get_adata().obs[f'tissue_{algo.method_key}'].value_counts(normalize=True).mul(100).rename(
                    algo.filename)))
            plot_columns.append(algo.filename)

        cummulative_df = pd.concat(cell_percentage_dfs, axis=1).fillna(0)
        cummulative_df = cummulative_df.loc[
            sorted(cummulative_df.index.values, key=lambda x: float(x) if x != "unknown" else float('inf'))]
        cummulative_df.plot(y=plot_columns, kind="bar", rot=0, ax=ax, xlabel="", color=colors)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.grid(False)
        ax.set_facecolor('white')
        plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

        plt.savefig(os.path.join(self.params['out_path'], 'cluster_abundance_all_slices.png'), bbox_inches='tight')
        if not self.params['hide_plots']:
            plt.show()
        plt.close()

    @timeit
    def plot_cluster_abundance_per_slice(self):
        """
        Plots the cluster abundance for each algorithm per slice.
        """
        number_of_samples = len(self.algo_list)
        if number_of_samples <= 2:
            number_of_rows = 1
            number_of_columns = number_of_samples
        else:
            number_of_rows = 2 if number_of_samples % 2 == 0 else 1
            number_of_columns = number_of_samples // 2 if number_of_samples % 2 == 0 else number_of_samples
        fig, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, figsize=(20, 20), squeeze=False)
        axes = axes.ravel()
        fig.subplots_adjust(wspace=0)
        set_figure_params(dpi=self.params['dpi'], facecolor='white')

        cell_percentage_dfs = []
        plot_columns = []
        for algo in self.algo_list:
            cell_percentage_dfs.append(pd.DataFrame(
                algo.get_adata().obs[f'tissue_{algo.method_key}'].value_counts(normalize=True).mul(100).rename(
                    algo.filename)))
            plot_columns.append(algo.filename)

        cummulative_df = pd.concat(cell_percentage_dfs, axis=1).fillna(0)
        cummulative_df = cummulative_df.loc[
            sorted(cummulative_df.index.values, key=lambda x: float(x) if x != "unknown" else float('inf'))]

        for i in range(number_of_rows * number_of_columns):
            axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            axes[i].set_facecolor('white')
            axes[i].set_title(plot_columns[i])
            cummulative_df.plot(y=plot_columns[i], kind="bar", rot=0, ax=axes[i], xlabel="", color="grey", legend=False)
            axes[i].grid(False)

        for ax in axes:
            ax.grid(False)

        plt.savefig(os.path.join(self.params['out_path'], 'cluster_abundance_per_slice.png'), bbox_inches='tight')
        if not self.params['hide_plots']:
            plt.show()
        plt.close()

    def plot(self, function_ind: str, slice_id=0, community_id=None):
        orig_hide_plots = self.params['hide_plots']
        try:
            self.params['hide_plots'] = False
            if function_ind == "all_annotations":
                self.plot_all_annotation()
            if function_ind == "all_clustering":
                self.plot_all_clustering()
            if function_ind == "cell_type_mixtures_total":
                self.plot_celltype_mixtures_total(
                    [algo.get_cell_mixtures().to_dict() for algo in self.algo_list])
            if function_ind == "cell_perc_in_community_per_slice":
                self.plot_cell_perc_in_community_per_slice()
            if function_ind == "cell_abundance_total":
                self.plot_cell_abundance_total()
            if function_ind == "cell_abundance_per_slice":
                self.plot_cell_abundance_per_slice()
            if function_ind == "cluster_abundance_total":
                self.plot_cluster_abundance_total()
            if function_ind == "cluster_abundance_per_slice":
                self.plot_cluster_abundance_per_slice()

            self.algo_list[slice_id].hide_plots = False
            if function_ind == "annotation":
                self.algo_list[slice_id].plot_annotation()
            if function_ind == "clustering":
                self.algo_list[slice_id].plot_clustering()
            if function_ind == "colorplot":
                self.algo_list[slice_id].colorplot_stats(self.params['color_plot_system'], community_id)
            if function_ind == "colorplot_cell_type":
                self.algo_list[slice_id].colorplot_stats_per_cell_types()
            if function_ind == "cell_types_table":
                self.algo_list[slice_id].plot_celltype_table()
            if function_ind == "boxplot":
                self.algo_list[slice_id].boxplot_stats(community_id)
            if function_ind == "cell_types_images":
                self.algo_list[slice_id].plot_celltype_images()
            if function_ind == "histogram_cell_sums":
                self.algo_list[slice_id].plot_histogram_cell_sum_window()
            if function_ind == "cluster_mixtures":
                self.algo_list[slice_id].plot_cluster_mixtures(community_id)
            if function_ind == "cell_mixture_table":
                self.algo_list[slice_id].plot_stats()
        finally:
            self.params['hide_plots'] = orig_hide_plots
            self.algo_list[slice_id].hide_plots = orig_hide_plots
            reset_figure_params()


class CommunityDetection(AlgorithmBase, _CommunityDetection):

    def main(self, **kwargs):
        r"""

        CCD divides the tissue using sliding windows by accommodating multiple window sizes, and enables the simultaneous analysis of multiple slices from the same tissue. CCD consists of the three main steps:

        1. Single or multiple-size sliding windows ($w$) are moved through the surface of the tissue with defined horizontal and vertical step while calculating the percentages ($[p_1, p_2,...,p_n]$) of each cell type inside of it. A feature vector ($fv$) with size equal to the number of cell types ($n$) is created for each processed window across all available tissue slices:

            .. math::

                \begin{equation}
                    \forall w_i\rightarrow (fv_i = [p_1, p_2,...,p_n])
                \end{equation}

        2. Feature vectors from all windows are fed to the clustering algorithm ($C$) such as Leiden, Spectral or Hierarchical to obtain community labels ($l$). The number of the desired communities ($cn$) can be predefined explicitly as a parameter (Spectral or Hierarchical clustering) or by setting the resolution of clustering (Leiden):

            .. math::

                \begin{equation}
                    C(\forall fv_i) \rightarrow l_i, l_i \in {l_1, l_2, ..., l_{cn}}
                \end{equation}

        3. Community label is assigned to each cell-spot ($cs$) by majority voting ($MV$) using community labels from all windows covering it:

            .. math::

                \begin{equation}
                    MV(\forall l_i)\text{ where } spatial(cs_j) \in w_i \rightarrow l_j, l_j \in {l_1, l_2, ..., l_{cn}}
                \end{equation}

        The window size and sliding step are optional CCD parameters. If not provided, the optimal window size is calculated throughout the iterative process with goal of having average number of cell-spots in all windows in range [30, 50]. Sliding step is set to the half of the window size.

        .. note::

            All the parameters are key word arguments.

        :param annotation: The key specified the cell type in obs.
        :param tfile: File path to Anndata object with calculated cell mixtures for data windows, output of calc_feature_matrix.
        :param out_path: Absolute path to store outputs, default to './results'.
        :param cluster_algo: Clustering algorithm, default to leiden.
        :param resolution: Resolution of leiden clustering algorithm. Ignored for spectral and agglomerative, default to 0.2.
        :param n_clusters: Number of clusters for spectral and agglomerative clustering, ignored for leiden, default to 10.
        :param spot_size: Size of the spot on plot, default to 30.
        :param verbose: Show logging messages. 0 - Show warnings, >0 show info, default to 0.
        :param plotting: Save plots flag, default to 5, available values include:

                        | 0 - No plotting and saving.
                        | 1 - save clustering plot.
                        | 2 - additionally save plots of cell type images statistics and cell mixture plots.
                        | 3 - additionally save cell and cluster abundance plots and cell mixture plots for all slices and cluster mixture plots and boxplots for each slice.
                        | 4 - additionally save cell type images, abundance plots and cell percentage table for each slice.
                        | 5 - additionally save color plots.
        :param project_name: Project name that is used to name a directory containing all the slices used, default to community.
        :param skip_stats: Skip statistics calculation on cell community clustering result.
                            A table of cell mixtures and comparative spatial plots of cell types and mixtures will not be created, default to False.
        :param total_cell_norm: Total number of cells per window mixture after normalization, default to 10000.
        :param downsample_rate: Rate by which the binary image of cells is downsampled before calculating the entropy and scatteredness metrics.
                                If no value is provided, downsample_rate will be equal to 1/2 of minimal window size, default to None.
        :param num_threads: Number of threads that will be used to speed up community calling, default to 5.
        :param entropy_thres: Threshold value for spatial cell type entropy for filtering out overdispersed cell types, default to 1.0.
        :param scatter_thres: Threshold value for spatial cell type scatteredness for filtering out overdispersed cell types, default to 1.0.
        :param win_sizes: Comma separated list of window sizes for analyzing the cell community.
        :param sliding_steps: Comma separated list of sliding steps for sliding window.
        :param min_cluster_size: Minimum number of cell for cluster to be plotted in plot_stats(), default to 200.
        :param min_perc_to_show: Minimum percentage of cell type in cluster for cell type to be plotted in plot_stats(), default to 4.
        :param min_num_celltype: Minimum number of cell types that have more than `min_perc_celltype` in a cluster,
                                for a cluster to be shown in plot_celltype_table(), default to 1.
        :param min_perc_celltype: Minimum percentage of cells of a cell type which at least min_num_celltype cell types
                                need to have to show a cluster in plot_celltype_table().
        :param min_cells_coeff: Multiple od standard deviations from mean values where the cutoff for m, default to 1.5.
        :param color_plot_system: Color system for display of cluster specific windows, default rgb.
        :param save_adata: Save adata file with resulting .obs column of cell community labels, default to False.
        :param min_count_per_type: Minimum number of cells per cell type needed to use the cell type for cell communities extraction (in percentages), default to 0.1.
        :param hide_plots: Stop plots from displaying in notebooks or standard ouput. Used for batch processing, default to True.
        :param dpi: DPI (dots per inch) used for plotting figures, default to 100.

        :return: Object of CommunityDetection.
        """  # noqa

        assert type(self.stereo_exp_data) is AnnBasedStereoExpData, \
            "this method can only run with AnnBasedStereoExpData temporarily"
        self._main([self.stereo_exp_data], **kwargs)
        return self
