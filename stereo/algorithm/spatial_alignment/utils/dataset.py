#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/9/23 5:16 PM
# @Author  : zhangchao
# @File    : dataset.py
# @Email   : zhangchao5@genomics.cn
# import scanpy as sc
from typing import List, Tuple
import torch
import scipy.sparse as sp
import torch_geometric

from anndata import AnnData
from torch_geometric.data import Data
import torch_geometric.transforms
from torch_geometric.utils import to_undirected

from stereo.log_manager import logger

from .time_decorator import get_format_time


class Dataset:
    """
    Multiple dataset loading, preprocessing and convert to dataloader
    :param data_path: str
        Input dataset path.
    :param min_genes: int
        Minimum number of genes expressed required for a cell to pass filtering, default 20.
    :param min_cells: int
        Minimum number of cells expressed required for a gene to pass filtering, default 20.
    :param batch_key: str
        the batch annotation to :attr:`obs` using this key, default, 'batch'.
    :param is_norm_log: bool
        Whether to perform 'sc.pp.normalize_total' and 'sc.pp.log1p' processing, default, True.
    :param is_scale: bool
        Whether to perform 'sc.pp.scale' processing, default, False.
    :param is_hvg: bool
        Whether to perform 'sc.pp.highly_variable_genes' processing, default, False.
    :param is_reduce: bool
        Whether to perform PCA reduce dimensional processing, default, False.
    :param n_pcs: int
        PCA dimension reduction parameter, valid when 'is_reduce' is True, default, 100.
    :param n_hvg: int
        'sc.pp.highly_variable_genes' parameter, valid when 'is_reduce' is True, default, 2000.
    :param n_neigh: int
        The number of neighbors selected when constructing a spatial neighbor graph. default, 15.
    :param is_undirected: bool
        Whether the constructed spatial neighbor graph is undirected graph, default, True.
    """
    def __init__(
        self,
        #  *data_path: str,
        merge_data: AnnData,
        #  min_genes: int = 20,
        #  min_cells: int = 20,
        batch_key: str = "batch",
        #  is_norm_log: bool = False,
        #  is_scale: bool = True,
        #  is_hvg: bool = False,
        is_reduce: bool = False,
        n_pcs: int = 100,
        #  n_hvg: int = 2000,
        n_neigh: int = 15,
        is_undirected: bool = True,
        spatial_key: str = "spatial"
    ):

        # self.data_path = data_path
        # self.is_norm_log = is_norm_log
        # self.is_scale = is_scale
        self.data_list = []
        self.batch_key = batch_key
        self.merge_data = merge_data
        self.spatial_key = spatial_key
        self.loader_list, self.batch_no = self._loader(
            merge_data=merge_data, is_reduce=is_reduce, n_pcs=n_pcs, n_neigh=n_neigh, is_undirected=is_undirected)
        # self.merge_data, self.loader_list = self.get_loader(
        #     min_genes=min_genes, min_cells=min_cells, is_reduce=is_reduce, is_hvg=is_hvg, n_pcs=n_pcs, n_hvg=n_hvg,
        #     n_neigh=n_neigh, is_undirected=is_undirected)
        self.n_domain = self.merge_data.obs[batch_key].cat.categories.size
        self.n_node = self.merge_data.shape[0]

        self.inner_dims = n_pcs if is_reduce else self.merge_data.shape[1]
        self.inner_genes = self.merge_data.var_names.tolist()

    # def get_loader(self,
    #                min_genes=20,
    #                min_cells=20,
    #                is_reduce=False,
    #                is_hvg=False,
    #                n_pcs=50,
    #                n_hvg=2000,
    #                n_neigh=30,
    #                is_undirected=True):
    #     merge_data = self._reader(min_genes, min_cells)
    #     if is_hvg:
    #         sc.pp.highly_variable_genes(merge_data, flavor="seurat_v3", n_top_genes=n_hvg)
    #         merge_data = merge_data[:, merge_data.var["highly_variable"]]

    #     dataset_list = self._loader(
    #         merge_data=merge_data, is_reduce=is_reduce, n_pcs=n_pcs, n_neigh=n_neigh, is_undirected=is_undirected)
    #     return merge_data, dataset_list

    # def _reader(self, min_genes, min_cells):
    #     print(f"{get_format_time()} Found Dataset: ")
    #     for path in self.data_path:
    #         data = sc.read_h5ad(path)
    #         data.var_names_make_unique()
    #         data.obs_names_make_unique()

    #         if self.is_norm_log:
    #             sc.pp.filter_cells(data, min_genes=min_genes)
    #             sc.pp.filter_genes(data, min_cells=min_cells)
    #             sc.pp.normalize_total(data, target_sum=1e4)
    #             sc.pp.log1p(data)
    #         if self.is_scale:
    #             sc.pp.scale(data, zero_center=False, max_value=10)

    #         self.data_list.append(data)
    #     [print(f"  cell nums: {data.shape[0]} gene nums: {data.shape[1]}") for data in self.data_list]
    #     if len(self.data_path) > 1:
    #         merge_data = AnnData.concatenate(*self.data_list, batch_key=self.batch_key)
    #     else:
    #         merge_data = self.data_list[0]
    #         merge_data.obs[self.batch_key] = 0
    #         merge_data.obs[self.batch_key] = merge_data.obs[self.batch_key].astype("category")
    #     return merge_data

    def convert_tensor(self, data, q=50, is_reduce=False):
        data = data if not sp.issparse(data) else data.toarray()
        x_tensor = torch.tensor(data)
        if not is_reduce:
            return x_tensor
        else:
            u, s, v = torch.pca_lowrank(x_tensor, q=q)
            pca_tensor = torch.matmul(x_tensor, v)
            return pca_tensor

    def _loader(self, merge_data: AnnData, is_reduce=False, n_pcs=100, n_neigh=15, is_undirected=True) -> Tuple[List[Data], List[str]]:
        dataset_list = []
        if self.spatial_key in merge_data.obsm_keys():
            # print(f"{get_format_time()}: Spatial coordinates are used to calculate nearest neighbor graphs")
            logger.info(f"The spatial coordinates specified by {self.spatial_key} are used to calculate nearest neighbor graphs")
            spatial_key = self.spatial_key
        else:
            # print(f"{get_format_time()}: PCA embedding are used to calculate nearest neighbor graphs")
            # logger.info(f"The spatial coordinates specified by {self.spatial_key} are not found, pca embedding is used to calculate nearest neighbor graphs")
            # spatial_key = "pca"
            raise KeyError(f"The spatial coordinates specified by {self.spatial_key} are not found")

        batch_no = []
        for d_idx, domain in enumerate(merge_data.obs[self.batch_key].cat.categories):
            batch_no.append(domain)
            data = merge_data[merge_data.obs[self.batch_key] == domain]
            feat_tensor = self.convert_tensor(data.X, q=n_pcs, is_reduce=is_reduce)

            if spatial_key == self.spatial_key:
                dataset = Data(x=feat_tensor, pos=torch.Tensor(data.obsm[spatial_key]))
            else:
                pos_tensor = self.convert_tensor(data=data.X, q=10, is_reduce=True)
                dataset = Data(x=feat_tensor, pos=pos_tensor)

            dataset: Data = torch_geometric.transforms.KNNGraph(k=n_neigh, loop=True)(dataset)
            dataset.edge_weight = torch.ones(dataset.edge_index.size(1))
            dataset.neigh_graph = torch.zeros((dataset.num_nodes, dataset.num_nodes), dtype=torch.float)
            dataset.neigh_graph[dataset.edge_index[0], dataset.edge_index[1]] = 1.
            if is_undirected:
                dataset.edge_index, dataset.edge_weight = to_undirected(dataset.edge_index, dataset.edge_weight)
                dataset.edge_weight = torch.ones_like(dataset.edge_weight)
            dataset.domain_idx = torch.tensor([d_idx] * data.shape[0], dtype=torch.int32)
            dataset.idx = torch.tensor(range(data.shape[0]), dtype=torch.int32)
            dataset_list.append(dataset)
        return dataset_list, batch_no
