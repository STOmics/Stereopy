import copy
import os
import time
from collections import defaultdict
from typing import List

import anndata as ad
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from .ccd import *
from .ms_algorithm_base import MSDataAlgorithmBase


class CommunityDetection(MSDataAlgorithmBase):
    """
    Class for performing community detection on a set of slices.
    """

    @timeit
    def main(self, annotation: str, **kwargs):
        """
        Executes the community detection algorithm.

        Parameters:
        - slices (List[AnnBasedStereoExpData]): A list of AnnBasedStereoExpData objects representing the slices of a tissue.
        - annotation (str): The annotation string.
        - **kwargs: Additional keyword arguments (Check constants.py for the description of additional arguments)

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
        """

        self.params = {**COMMUNITY_DETECTION_DEFAULTS, **kwargs}
        self.params['annotation'] = annotation
        # self.stereo_data_slices = slices
        self.slices = self.ms_data.data_list
        for slice in self.slices:
            if 'X_spatial' in slice._ann_data.obsm:
                slice._ann_data.obsm['spatial'] = slice._ann_data.obsm['X_spatial'].copy()
            elif 'spatial_stereoseq' in slice._ann_data.obsm:
                slice._ann_data.obsm['spatial'] = np.array(slice._ann_data.obsm['spatial_stereoseq'].copy())

        self.file_names = [fname for fname in self.params['files'].split(',')] if 'files' in self.params else [
            f"Slice_{id}" for id in range(len(self.slices))]
        if self.params['win_sizes'] == 'NA' or self.params['sliding_steps'] == 'NA':
            logger.info(
                "Window sizes and/or sliding steps not provided by user - proceeding to calculate optimal values")
            self.params['win_sizes'], self.params['sliding_steps'] = self.calc_optimal_win_size_and_slide_step()
        else:
            self.log_win_size_full_info()

        start_time = time.perf_counter()

        if not os.path.exists(self.params['out_path']):
            os.makedirs(self.params['out_path'])

        algo_list = []
        win_sizes = "_".join([i for i in self.params['win_sizes'].split(',')])
        sliding_steps = "_".join([i for i in self.params['sliding_steps'].split(',')])
        self.params['project_name_orig'] = self.params['project_name']
        self.params['out_path_orig'] = self.params['out_path']
        cluster_string = f"_r{self.params['resolution']}" if self.params[
                                                                 'cluster_algo'] == 'leiden' else f"_nc{self.params['n_clusters']}"
        self.params[
            'project_name'] += f"_c{self.params['cluster_algo']}{cluster_string}_ws{win_sizes}_ss{sliding_steps}_sct{self.params['scatter_thres']}_dwr{self.params['downsample_rate']}_mcc{self.params['min_cells_coeff']}"
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

            # add algo object for each slice to a list
            algo_list.append(algo)
        self.algo_list = algo_list
        # if self.params['plotting'] > 0 and len(algo_list) > 1:
        #     self.plot_all_annotation()

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
                    algo.plot_stats()
                    algo.plot_celltype_table()
                if self.params['plotting'] > 2:
                    algo.plot_cluster_mixtures()
                    algo.boxplot_stats()
                if self.params['plotting'] > 4:
                    algo.colorplot_stats(color_system=self.params['color_plot_system'])
                    algo.colorplot_stats_per_cell_types()
                # save final tissue with stats
                algo.save_tissue(suffix='_stats')

        self.pipeline_res['ccd'] = {'algo_list': algo_list}
        print(self.pipeline_res)

        # if self.params['plotting'] > 0 and len(algo_list) > 1:
        #     self.plot_all_clustering()
        # if self.params['plotting'] > 2:
        #     self.plot_celltype_mixtures_total([algo.get_cell_mixtures().to_dict() for algo in algo_list])
        #     self.plot_cell_abundance_total()
        #     self.plot_cluster_abundance_total()
        # if self.params['plotting'] > 3:
        #     self.plot_cell_abundance_per_slice()
        #     self.plot_cluster_abundance_per_slice()
        #     self.plot_cell_perc_in_community_per_slice()

        end_time = time.perf_counter()

        self.params['execution_time'] = end_time - start_time
        # generate_report(self.params)

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
        merged_tissue = AnnBasedStereoExpData(based_ann_data=merged_tissue)
        if self.params['cluster_algo'] == 'leiden':
            merged_tissue._ann_data.obsm['X_pca_dummy'] = merged_tissue._ann_data.X
            merged_tissue.tl.neighbors(pca_res_key='X_pca_dummy', n_neighbors=15)
            merged_tissue.tl.leiden(neighbors_res_key='neighbors', res_key='leiden',
                                    resolution=self.params['resolution'])
            merged_tissue._ann_data.obs['leiden'] = merged_tissue._ann_data.obs['leiden'].astype('int')
            merged_tissue._ann_data.obs['leiden'] -= 1
            merged_tissue._ann_data.obs['leiden'] = merged_tissue._ann_data.obs['leiden'].astype('str')
            merged_tissue._ann_data.obs['leiden'] = merged_tissue._ann_data.obs['leiden'].astype('category')
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

