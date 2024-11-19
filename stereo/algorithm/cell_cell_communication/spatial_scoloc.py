# -*- coding: utf-8 -*-
# @Time    : 2023/3/10 11:03
# @Author  : liuxiaobin
# @File    : spatial_scoloc.py
# @Version：V 0.1
# @desc : When applied to spatial transcriptomics data, each spot has a corresponding 2D or 3D coordinate.
# We will use the KL-divergence of 2D/3D kernel density estimates (KDE) to generate a distance matrix
# for each cell type pairs.
# Thereafter, we can generate a micro-environment file by filtering the matrix on a user-defined threshold.
# This file will be used as an input for our main CCC module.


import random

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from stereo.algorithm.cell_cell_communication.analysis_helper import write_to_file
from stereo.algorithm.cell_cell_communication.exceptions import InvalidMicroEnvInput


class GetMicroEnvs:
    # FIXME: change the default output path
    def main(self,
             meta: pd.DataFrame,
             coord: pd.DataFrame,
             n_boot: int = 20,
             boot_prop: float = 0.8,
             dimension: int = 3,
             fill_rare: bool = True,
             min_num: int = 30,
             binsize: float = 2,
             eps: float = 1e-20,
             output_path: str = None
             ):
        """
        Generate the micro-environment file used for the CCC analysis.
        The output is in the following format:

        cell_type	microenviroment
        NKcells_1	location_1
        NKcells_0	location_2
        Tcells	    location_1
        Myeloid	    location_2

        :param meta: dataframe of two columns: cell, cell_type.
        :param coord: spots and their 2D/3D coordinates, columns: cell, coord_x, coord_y, coord_z.
        :param n_boot: number of bootstrap samples, default = 100.
        :param boot_prop: proportion of each bootstrap sample, default = 0.8.
        :param dimension: 2 or 3.
        :param fill_rare: bool, whether simulate cells for rare cell type when calculating kde.
        :param min_num: if a cell type has cells < min_num, it is considered rare.
        :param binsize: grid size used for kde.
        :param eps: fill eps to zero kde to avoid inf KL divergence.
        :param output_path: the directory to save the result.
        """

        """
        0. Verify input parameters
        """
        # output_path = output_path + '\\result_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        if boot_prop <= 0 or boot_prop > 1:
            raise InvalidMicroEnvInput('Bootstrap proportion should be between 0 and 1.')
        if dimension not in [2, 3]:
            raise InvalidMicroEnvInput('Dimension number can only be 2 or 3.')
        """
        1. Preprocessing, get one-hot encoded cell type data
        """
        meta = meta.dropna().reset_index(drop=True)
        meta_onehot, type_name, type_dist = self._get_dummy_df(meta)

        data = pd.merge(coord, meta_onehot, on='cell')
        data = data.dropna().reset_index(drop=True)
        n_cell = data.shape[0]
        n_type = len(type_name)

        """
        2. Bootstrap
        """
        # initialize a list to save MST results
        mst_boot = []
        # initialize a dataframe to save pairwise kl results in every bootstrap sample.
        # KL's direction: from row to column
        pairwise_kl_df = pd.DataFrame(index=type_name, columns=type_name)
        # every initial cell value of the dataframe is an empty list.
        for col in type_name:
            pairwise_kl_df[col] = pairwise_kl_df[col].apply(lambda x: [])
        # do bootstrap
        for i in range(n_boot):
            random.seed(i)
            # Get the bootstrap sample
            idx = np.random.choice(range(n_cell), round(n_cell * boot_prop), replace=True)
            data_boot = data.iloc[idx,]
            # if a cell type has fewer than min_num cells, randomly assign some other cells to be of this type to make
            # the total cell number equal to min_num.
            if fill_rare:
                data_boot_filled = self._fill_rare_type(data_boot, type_name, min_num)
            else:
                data_boot_filled = data_boot

            # Calculate 2d/3d kde for each cell type
            # calculate locations where kernel density is calculated
            if dimension == 2:
                grid_points_x, grid_points_y = self._grid_of_single_slice(data_boot_filled['coord_x'],
                                                                          data_boot_filled['coord_y'], binsize)
                grid_points = np.stack([grid_points_x, grid_points_y]).T
            if dimension == 3:
                grid_points = self._grid_of_all_slices(data_boot_filled, binsize)

            n_grid_points = len(grid_points)

            # fit kde for each cell type
            kde_cell_type = []
            for cty in type_name:
                sample = data_boot_filled[data_boot_filled[cty] == 1]
                if dimension == 2:
                    sample_points = np.vstack([sample['coord_x'].to_numpy(), sample['coord_y'].to_numpy()])
                if dimension == 3:
                    sample_points = np.vstack(
                        [sample['coord_x'].to_numpy(), sample['coord_y'].to_numpy(), sample['coord_z'].to_numpy()])
                sample_points = self._check_coordinates(sample_points)
                if not sample.empty:
                    kde = stats.gaussian_kde(sample_points)  # Calculate kernel
                    kde_at_grid = kde(grid_points.T)  # Calculate kde at the grid points
                    if np.any(kde_at_grid > 0):
                        kde_at_grid = kde_at_grid / np.sum(kde_at_grid)
                    kde_at_grid[kde_at_grid < eps] = eps  # fill zeros to avoid inf kl divergence
                else:
                    kde_at_grid = np.array([eps] * n_grid_points)

                kde_cell_type.append(kde_at_grid)

            # Calculate pairwise KL-divergence for all the cell types. KL divergence is not symmetric.
            kl_array = self._pairwise_kl_divergence(pairwise_kl_df, kde_cell_type, n_type)
            # get MST from the kl matrix
            tree_edge_df = self._get_mst(type_name, kl_array)
            tree_edge_df['weight'] = [1.0 / float(n_boot)] * tree_edge_df.shape[0]
            mst_boot.append(tree_edge_df)

        """
        3. Combine all the bootstrap MSTs to get the final MST
        """
        all_mst = pd.concat(mst_boot, ignore_index=True)
        final_mst = all_mst.groupby(['from', 'to']).agg('sum').reset_index()

        """
        4. Average over all the bootstrap samples to get the final pairwise KL-divergence matrix
        """

        def mean_boot(ls):
            """
            define this function to avoid warning when calculate np.mean to empty list
            """
            if ls:
                return np.mean(ls)

        pairwise_kl_df = pairwise_kl_df.applymap(mean_boot)
        subgroups_by_thrshold = self._threshold_detection(pairwise_kl_df, type_name)

        if output_path is not None:
            write_to_file(all_mst, 'mst_in_boot', output_path=output_path, output_format='csv')
            write_to_file(final_mst, 'mst_final', output_path=output_path, output_format='csv')
            write_to_file(pairwise_kl_df, 'pairwise_kl_divergence', output_path=output_path, output_format='csv')
            write_to_file(subgroups_by_thrshold, 'split_by_different_threshold', output_path=output_path,
                          output_format='csv')

        return all_mst, final_mst, pairwise_kl_df, subgroups_by_thrshold

    def _check_coordinates(self, coordinates: np.ndarray):
        for coord in coordinates:
            if np.all(coord == coord[0]):
                if coord[0] == 0:
                    coord[0] = 0.01
                else:
                    coord[0] *= 1.01
        return coordinates

    # FIXME: change the default output path
    def generate_micro_envs(
            self,
            method,
            threshold: float = None,
            result_df: pd.DataFrame = None,
            output_path: str = None
    ):
        """
        5. Define micro environments using two methods:
        1) minimum spanning tree, or
        2) pruning the fully connected tree based on a given threshold of KL, then split the graph into
        multiple strongly connected component.

        """
        if method not in ['mst', 'split']:
            raise InvalidMicroEnvInput("Choose method from 'mst' and 'split'.")

        microenv = pd.DataFrame()

        if method == 'mst':
            if threshold:
                result_df = result_df[result_df['weight'] > threshold].reset_index(drop=True)

            types = []
            microenvironment = []
            for index, row in result_df.iterrows():
                types = types + [row['from'], row['to']]
                microenvironment = microenvironment + ['microenv_' + str(index)] * 2
            microenv = pd.DataFrame({'cell_type': types, 'microenvironment': microenvironment})

        if method == 'split':
            type_name = result_df.columns
            subgroups = self.split_micro_envs(result_df, type_name, threshold)

            types = []
            microenvironment = []
            for index, subgroup in enumerate(subgroups):
                types = types + list(subgroup)
                microenvironment = microenvironment + ['microenv_' + str(index)] * len(subgroup)
            microenv = pd.DataFrame({'cell_type': types, 'microenvironment': microenvironment})

        if output_path is not None:
            write_to_file(microenv, 'microenv_' + method, output_path=output_path, output_format='csv')
        return microenv

    def _get_dummy_df(self, meta, col_type='cell_type'):
        """
        Add one-hot coded cell type columns (#cols added = #cell types) to the original meta dataframe.
        (Because when calculate the kde, we randomly assign some cells to the rare types.
        So a cell might have multiple cell_type labels)

        """
        out = meta.copy()
        out[col_type] = out[col_type].values.astype('str')
        type_name = np.sort(np.unique(out[col_type]))
        for ctp in type_name:
            out[ctp] = np.where(out[col_type] == ctp, 1, 0)
        type_dist = dict(out[type_name].sum(axis=0, skipna=True))
        return out, type_name, type_dist

    def _fill_rare_type(self, data_onehot, type_name, min_num=15, col_type='cell_type'):
        """
        If some cell type has fewer than a given number of cells (min_num),
        randomly assign more cells to be of this type, so that the total number is min_num (default=15) cells.
        """
        data_onehot = data_onehot.astype({col_type: 'string'})
        count_dict = dict(data_onehot[type_name].sum(axis=0, skipna=True))
        rare_type = [key for key, value in count_dict.items() if value < min_num]
        for t in rare_type:
            zero_row = data_onehot[data_onehot[t] == 0].index
            fill_row = np.random.choice(zero_row, min_num - count_dict[t], replace=False)
            data_onehot.loc[fill_row, t] = 1
        return data_onehot

    def _grid_of_all_slices(self, data, binsize):
        """
        For each of the slices, calculate the grid locations where 3D kde is to be performed.
        """
        grid_for_density = []
        for slice in np.unique(data['coord_z']):
            slice_data = data[data['coord_z'] == slice]
            x_in_slice = slice_data['coord_x'].to_numpy()
            y_in_slice = slice_data['coord_y'].to_numpy()
            x_in_grid, y_in_grid = self._grid_of_single_slice(x_in_slice, y_in_slice, binsize)
            grid_for_density.append(np.vstack([x_in_grid, y_in_grid, (np.ones(x_in_grid.shape) * slice)]).T)
        positions = np.vstack(grid_for_density)
        return positions

    def _grid_of_single_slice(self, x, y, binsize):
        """
        For a single slice (2D), calculate the grid locations where kde is to be performed.
        Only keep grid locations where there are cells.
        """
        pos_data_x = x.copy()
        pos_data_y = y.copy()

        xmin = np.min(pos_data_x) - 2 * binsize
        xmax = np.max(pos_data_x) + 2 * binsize
        ymin = np.min(pos_data_y) - 2 * binsize
        ymax = np.max(pos_data_y) + 2 * binsize

        pos_data_x = (pos_data_x - xmin) / binsize
        pos_data_x = pos_data_x.astype(int)
        pos_data_y = (pos_data_y - ymin) / binsize
        pos_data_y = pos_data_y.astype(int)

        X, Y = np.mgrid[xmin:xmax:binsize, ymin:ymax:binsize]
        body_mask = np.zeros(X.shape)
        body_mask[pos_data_x, pos_data_y] = 1

        x_in_grid = X[body_mask == 1]
        y_in_grid = Y[body_mask == 1]
        return x_in_grid, y_in_grid

    def _pairwise_kl_divergence(self, pairwise_kl_df, kde, n_type):
        """
        Calculate pairwise KL divergence for all the cell types
        """
        kl_array = np.zeros((n_type, n_type))  # this array is generated for the mst calculation
        for i, ct1 in enumerate(pairwise_kl_df.index):
            for j, ct2 in enumerate(pairwise_kl_df.columns):
                if ct1 == ct2:
                    continue
                # one row for one ref celltype
                kl_divergence = stats.entropy(pk=kde[i], qk=kde[j], base=2)
                pairwise_kl_df.loc[ct1, ct2].append(kl_divergence)
                kl_array[i, j] = kl_divergence
        return kl_array

    def split_micro_envs(self, pairwise_kl_df, type_name, threshold):
        kl_graph = nx.DiGraph()
        for i in type_name:
            kl_graph.add_node(i)
        for ctp1 in type_name:
            for ctp2 in type_name:
                if ctp1 == ctp2:
                    continue
                if pairwise_kl_df.loc[ctp1, ctp2] < threshold:
                    kl_graph.add_edge(ctp1, ctp2, weight=pairwise_kl_df.loc[ctp1, ctp2])
        # get a list of subgroups
        subgroups = list(nx.weakly_connected_components(kl_graph))
        return subgroups

    def _threshold_detection(self, pairwise_kl_df, type_name):
        min_kl = np.min(np.min(pairwise_kl_df))
        max_kl = np.max(np.max(pairwise_kl_df))
        n_type = len(type_name)
        thresh = np.linspace(min_kl, max_kl, n_type * n_type)

        threshold = []
        subgroup = []
        for t in thresh:
            sub = self.split_micro_envs(pairwise_kl_df, type_name, t)
            threshold.append(t)
            subgroup.append(str(sub))
            if len(sub) == 1:
                break
        subgroup_df = pd.DataFrame({"threshold": threshold, "subgroup_result": subgroup})
        return subgroup_df

    def _get_mst(self, type_name, kl_array):
        """
        Generate minimum spanning tree from given kl matrix
        """
        xidx, yidx = np.nonzero(kl_array)
        edges = []
        weights = []
        for xid, yid in zip(xidx, yidx):
            edges.append([xid, yid])  # from KL ref to KL query
            weights.append(kl_array[xid, yid])
        ag = ig.Graph(n=len(type_name), edges=edges, edge_attrs={'weight': weights}, directed=True)
        mst = ag.spanning_tree(weights=ag.es["weight"])
        mst_maskarray = np.array(mst.get_adjacency().data)

        xidx, yidx = np.nonzero(mst_maskarray)
        type_name = np.array(type_name, dtype='str')
        round_pd = pd.DataFrame()
        round_pd['from'] = type_name[xidx.tolist()]
        round_pd['to'] = type_name[yidx.tolist()]
        return round_pd

