import multiprocessing as mp
import os
from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
# import scanpy as sc
from anndata import AnnData
from tqdm.auto import tqdm

from stereo.log_manager import logger
from .community_clustering_algorithm import CommunityClusteringAlgo
from .utils import timeit


class SlidingWindow(CommunityClusteringAlgo):
    """
    A class that represents the sliding window algorithm for community clustering.

    This class extends the CommunityClusteringAlgo base class and provides the implementation of the sliding window
    algorithm for community clustering. It takes in a spatial transcriptomics dataset and additional parameters to
    perform community clustering using sliding windows.

    """

    def __init__(self, adata, slice_id, input_file_path, **params):
        """
        This method initializes the SlidingWindow object by setting up the necessary attributes.

        Args:
        - adata (anndata.AnnData): Annotated data object containing spatial transcriptomics data.
        - slice_id (int): ID of the slice for which community clustering is performed.
        - input_file_path (str): Path to the input file.
        - **params: Additional parameters for the sliding window algorithm.

        Raises:
        - AssertionError: If the number of sliding steps is not equal to the number of window sizes.

        """
        super().__init__(adata, slice_id, input_file_path, **params)
        self.win_sizes_list = [int(w) for w in self.win_sizes.split(',')]
        self.sliding_steps_list = [int(s) for s in self.sliding_steps.split(',')]
        assert len(self.win_sizes_list) == len(self.sliding_steps_list), \
            "The number of sliding steps must be equal to the number of window sizes."
        win_sizes = "_".join([str(i) for i in self.win_sizes_list])
        sliding_steps = "_".join([str(i) for i in self.sliding_steps_list])
        cluster_string = f"_r{self.resolution}" if self.cluster_algo == 'leiden' else f"_nc{self.n_clusters}"
        self.params_suffix = f"_sldwin_sl{self.slice_id}_c{self.cluster_algo}{cluster_string}_ws{win_sizes}_ss" \
                             f"{sliding_steps}_sct{self.scatter_thres}_dwr{self.downsample_rate}_mcc" \
                             f"{self.min_cells_coeff}"
        self.filename = self.adata.uns['sample_name']
        self.dir_path = os.path.join(self.adata.uns['algo_params']['out_path'], self.filename)
        # create results folder
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

        self.method_key = 'sliding_window'

    def run(self):
        """
        This method executes the sliding window algorithm for community clustering. If the 'tfile' attribute is not set,
        it calls the 'calc_feature_matrix' method with the first window size and sliding step from the respective lists.

        Raises:
        - AttributeError: If the 'tfile' attribute ends with an extension other than '.h5ad'.

        """
        if not self.tfile:
            self.calc_feature_matrix(self.win_sizes_list[0], self.sliding_steps_list[0])
        else:
            if self.tfile.endswith('.h5ad'):
                self.tissue = ad.read_h5ad(self.tfile)
            else:
                raise AttributeError(f"File '{self.tfile}' extension is not .h5ad")

    @timeit
    def calc_feature_matrix(self, win_size, sliding_step):
        """
        Calculate the feature matrix for sliding window analysis.

        This method calculates the feature matrix for sliding window analysis based on the given window size and sliding step.
        It assigns centroids to each sliding step of windows and calculates features for each subwindow. The feature matrix
        is then placed in an AnnData object with specified spatial coordinates of the sliding windows.

        Parameters:
        - win_size (int): The size of the sliding window.
        - sliding_step (int): The sliding step for moving the window.

        """  # noqa

        # window size needs to be a multiple of sliding step
        sliding_step = (win_size / int((win_size / sliding_step))) if sliding_step != None else win_size  # noqa
        bin_slide_ratio = int(win_size / sliding_step)

        # create centroids for each sliding step of windows
        # dtype of obs DataFrame index column and annotation column should be 'str'
        # to remove the warning for editing the view of self.adata.obs we reinit the .obs
        self.adata.obs = self.adata.obs.copy()
        self.adata.obs[f'Centroid_X_{win_size}'] = pd.Series(
            ((self.adata.obsm['spatial'][:, 0]) / sliding_step).astype(int), index=self.adata.obs_names)
        self.adata.obs[f'Centroid_Y_{win_size}'] = pd.Series(
            ((self.adata.obsm['spatial'][:, 1]) / sliding_step).astype(int), index=self.adata.obs_names)
        # need to understand borders and padding
        # subwindows belonging to borders will not have a complete cell count
        x_max = self.adata.obs[f'Centroid_X_{win_size}'].max()
        y_max = self.adata.obs[f'Centroid_Y_{win_size}'].max()
        tmp_data = self.adata.obs[f'Centroid_X_{win_size}'].astype(str) + '_' + self.adata.obs[
            f'Centroid_Y_{win_size}'].astype(str) + '_' + str(self.slice_id) + '_' + str(win_size)
        self.adata.obs[f'window_spatial_{win_size}'] = tmp_data

        tmp = self.adata.obs[[f'window_spatial_{win_size}', self.annotation]]
        ret = {}
        # calculate features for each subwindow
        for sw_ind, sw_data in tmp.groupby(f'window_spatial_{win_size}'):
            templete_dic = {ct: 0 for ct in self.unique_cell_type}
            for cell in sw_data[self.annotation]:
                templete_dic[cell] += 1
            ret[sw_ind] = templete_dic
        # merge features by windows
        feature_matrix = {}
        for subwindow in ret.keys():
            # index of window is in the top left corner of the whole window
            feature_matrix[subwindow] = {}
            x_curr = int(subwindow.split("_")[0])
            y_curr = int(subwindow.split("_")[1])
            z_curr = int(subwindow.split("_")[2])
            w_curr = int(subwindow.split("_")[3])

            for slide_x in range(0, np.min([bin_slide_ratio, x_max - x_curr + 1])):
                for slide_y in range(0, np.min([bin_slide_ratio, y_max - y_curr + 1])):
                    window_key = f'{x_curr + slide_x}_{y_curr + slide_y}_{z_curr}_{w_curr}'
                    if window_key in ret.keys():
                        feature_matrix[subwindow] = {
                            k: feature_matrix[subwindow].get(k, 0) + ret[window_key].get(k, 0)
                            for k in sorted(set(feature_matrix[subwindow]).union(ret[window_key]))
                        }

        feature_matrix = pd.DataFrame(feature_matrix).T
        # feature_matrix is placed in AnnData object with specified spatial coordinated of the sliding windows
        self.tissue = AnnData(feature_matrix.astype(np.float32), dtype=np.float32)
        # spatial coordinates are expanded with 3rd dimension with slice_id
        # this should enable calculation of multi-slice cell communities
        self.tissue.obsm['spatial'] = np.array([x.split('_') for x in feature_matrix.index]).astype(int)
        self.tissue.obs['window_size'] = np.array([win_size for _ in feature_matrix.index])
        self.tissue.obs = self.tissue.obs.copy()
        self.tissue.obs['window_cell_sum'] = np.sum(self.tissue.X, axis=1)
        # scale the feature vector by the total number of cells in it
        self.tissue.X = ((self.tissue.X.T * self.total_cell_norm) / self.tissue.obs['window_cell_sum'].values).T
        # remove feature vectors which have less than a specified amount of cells
        mean_cell_sum = np.mean(self.tissue.obs['window_cell_sum'].values)
        stddev_cell_sum = np.std(self.tissue.obs['window_cell_sum'].values)
        min_cells_per_window = mean_cell_sum - self.min_cells_coeff * stddev_cell_sum
        self.tissue = self.tissue[self.tissue.obs['window_cell_sum'].values >= min_cells_per_window, :]

    @timeit
    def community_calling(self, win_size, sliding_step):
        """
        Perform community calling for subwindows based on overlapping window labels.

        This method defines the subwindow cluster label based on the labels of all overlapping windows. It goes through
        all subwindow positions and gathers clustering labels of all windows that contain it. The final label of the
        subwindow is determined by majority vote. Window cluster labels are stored in `self.tissue_pruned.obs[self.cluster_algo]`,
        and the subwindow labels are placed in `self.tissue.obs[self.cluster_algo + '_max_vote']`. The `self.tissue_pruned` object is
        used only for clustering and is discarded.

        Parameters:
        - win_size (int): The size of the sliding window.
        - sliding_step (int): The sliding step for moving the window.

        """  # noqa
        sliding_step = (win_size / int((win_size / sliding_step))) if sliding_step != None else win_size  # noqa

        bin_slide_ratio = int(win_size / sliding_step)
        x_min = self.adata.obs[f'Centroid_X_{win_size}'].min()
        y_min = self.adata.obs[f'Centroid_Y_{win_size}'].min()

        # max voting on cluster labels
        subwindow_locations = np.unique(self.adata.obs[f'window_spatial_{win_size}'])
        # variable for final subwindow labels
        cluster_max_vote = pd.Series(index=subwindow_locations, name=f'{self.cluster_algo}_max_vote', dtype=np.float64)
        for location in subwindow_locations:
            # extract x,y,z coordinates from location string
            x_curr, y_curr, z_curr, _ = np.array(location.split("_")).astype(int)
            # index of subwindow is in the top left corner of the whole window
            subwindow_labels = {}
            for slide_x in range(0, np.min([bin_slide_ratio, x_curr - x_min + 1])):
                for slide_y in range(0, np.min([bin_slide_ratio, y_curr - y_min + 1])):
                    # check if location exist (spatial area is not complete)
                    window_key = f'{x_curr - slide_x}_{y_curr - slide_y}_{z_curr}_{win_size}'
                    if window_key in self.tissue.obs.index:
                        new_value = self.tissue.obs.loc[window_key, self.cluster_algo]
                        subwindow_labels[new_value] = subwindow_labels[
                                                          new_value] + 1 if new_value in subwindow_labels.keys() else 1

            # MAX VOTE
            # max vote is saved in a new variable (not changed in tissue.obs) so that it does not have diagonal effect on other labels during refinement # noqa
            # max_voting result is created for each subwindow, while the self.cluster_algo clustering was defined for each window # noqa
            cluster_max_vote.loc[location] = max(subwindow_labels,
                                                 key=subwindow_labels.get) if subwindow_labels != {} else 'unknown'

        # copy clustering results from subwindows to cells of those subwindows in adata object
        self.adata.obs.loc[:, f'tissue_{self.method_key}'] = list(
            cluster_max_vote.loc[self.adata.obs[f'window_spatial_{win_size}']])
        self.adata.obs[f'tissue_{self.method_key}'] = self.adata.obs[f'tissue_{self.method_key}'].astype('category')

        logger.info(
            f'Sliding window cell mixture calculation done. Added results to adata.obs["tissue_{self.method_key}"]')


class SlidingWindowMultipleSizes(SlidingWindow):
    """Class for performing sliding window analysis with multiple window sizes."""

    def __init__(self, adata, slice_id, input_file_path, **params):
        """
        Initialize the SlidingWindowMultipleSizes object.

        Parameters:
        - adata: Annotated data object containing the input data.
        - slice_id: Identifier for the current slice.
        - input_file_path: Path to the input file.
        - **params: Additional parameters for initialization.

        """
        super().__init__(adata, slice_id, input_file_path, **params)

    def run(self):
        """
        This method calculates the feature matrix for each window size and sliding step combination, and concatenates
        the resulting `tissue` objects into a single `tissue` object. If a `tfile` is provided, the method delegates
        the execution to the base class `run` method.
        """
        tissue_list = []

        if not self.tfile:
            n = len(self.win_sizes_list)
            for i in range(n):
                super().calc_feature_matrix(self.win_sizes_list[i], self.sliding_steps_list[i])
                tissue_list.append(self.tissue)

            self.tissue = ad.concat(tissue_list, axis=0, join='outer', fill_value=0.0)
        else:
            super().run()

    def community_calling(self):
        """
        Perform community calling based on sliding window analysis.

        This method performs community calling based on the results of sliding window analysis. If there is only one
        window size and one sliding step specified, it delegates the execution to the base class `community_calling`
        method with the specified window size and sliding step. Otherwise, it executes the
        `community_calling_multiple_window_sizes_per_cell_multiprocessing` method.

        """
        if len(self.win_sizes_list) == 1 and len(self.sliding_steps_list) == 1:
            super().community_calling(self.win_sizes_list[0], self.sliding_steps_list[0])
        else:
            self.community_calling_multiple_window_sizes_per_cell_multiprocessing()
            logger.info(
                f'Sliding window cell mixture calculation done. Added results to adata.obs["tissue_{self.method_key}"]')

    @timeit
    def community_calling_multiple_window_sizes_per_cell_multiprocessing(self):
        """
        Perform community calling using multiple window sizes per cell with multiprocessing.

        For every cell, the method collects the location of its subwindow in `window_spatial_{win_size}`.
        It gathers all windows that cover that subwindow in `cell_labels_all` and repeats this process for all
        window sizes. Finally, it performs majority voting to obtain the final label for each cell. The cells are
        split into as many batches as there are available CPUs and processed in parallel.

        """
        # if cell batches are too small, caching is not efficient so we want at least 5000 cells per batch
        # TODO: Hope this advice useful. In the real high performance machine, it will be very terrifying if you
        #    choose to run process as the same number as cpu_count, may be hundreds of process will be started up.
        #    Because what `mp` does is fork, it will copy all things in the main process after these process starting.
        #    I experienced this before, and some of other processes and Mine were killed by the operating system. ~.~
        #    this code may be more safe:
        #       min(mp.cpu_count(), 16)
        #    or which i think is:
        #       num_cpus_used = min(mp.cpu_count() if mp.cpu_count() < self.num_threads else self.num_threads, 16)
        #    16 only a example number, it map be adjusted after you test in the real use-cases, and choose a number
        #    like which performs not bad.
        num_cpus_used = mp.cpu_count() if mp.cpu_count() < self.num_threads else self.num_threads

        with mp.Pool(processes=num_cpus_used) as pool:
            split_df = np.array_split(self.adata.obs, num_cpus_used)
            partial_results = pool.map(self.community_calling_partial, split_df)

        self.adata.obs[f'tissue_{self.method_key}'] = pd.concat(partial_results)
        self.adata.obs[f'tissue_{self.method_key}'] = self.adata.obs[f'tissue_{self.method_key}'].astype('category')

    def community_calling_partial(self, df):
        result = pd.Series(index=df.index, dtype='str')
        cache = {}

        for index, cell in tqdm(df.iterrows(), desc="Per cell computation in a subset of all cells... ",
                                total=df.shape[0]):
            cell_labels_all = defaultdict(int)

            for win_size, sliding_step in zip(self.win_sizes_list, self.sliding_steps_list):
                sliding_step = (win_size / int((win_size / sliding_step)))
                bin_slide_ratio = int(win_size / sliding_step)

                x_min = self.adata.obs[f'Centroid_X_{win_size}'].min()
                y_min = self.adata.obs[f'Centroid_Y_{win_size}'].min()

                cell_labels = defaultdict(int)
                x_curr, y_curr, z_curr, w_size = [int(num) for num in cell[f'window_spatial_{win_size}'].split("_")]

                if (x_curr, y_curr, z_curr, w_size) in cache:
                    cell_labels = cache[(x_curr, y_curr, z_curr, w_size)]
                else:
                    # index of cell is in the top left corner of the whole window
                    for slide_x in range(0, np.min([bin_slide_ratio, x_curr - x_min + 1])):
                        for slide_y in range(0, np.min([bin_slide_ratio, y_curr - y_min + 1])):
                            # check if location exist (spatial area is not complete)
                            window_key = f'{x_curr - slide_x}_{y_curr - slide_y}_{z_curr}_{win_size}'
                            if window_key in self.tissue.obs.index:
                                win_label = self.tissue.obs.loc[window_key, self.cluster_algo]
                                cell_labels[win_label] += 1

                    cache[(x_curr, y_curr, z_curr, w_size)] = cell_labels
                cell_labels_all = {
                    key: cell_labels_all.get(key, 0) + cell_labels.get(key, 0)
                    for key in set(cell_labels_all) | set(cell_labels)
                }
            max_vote_label = max(cell_labels_all, key=cell_labels_all.get) if cell_labels_all != {} else 'unknown'
            result[index] = max_vote_label

        return result
