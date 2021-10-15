#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua@genomics.cn
@time:2021/08/24
"""
from stereo.algorithm.pysctransform import get_hvg_residuals, vst
# from stereo.core.stereo_exp_data import StereoExpData
from scipy.sparse import issparse, csr_matrix
import numpy as np


def sc_transform(
        data,
        method="theta_ml",
        n_cells=5000,
        n_genes=2000,
        filter_hvgs=False,
        res_clip_range="seurat",
        var_features_n=3000,
        threads=4
):
    """
    python version sc transform

    :param data: stereoExpData object
    :param method: offset, theta_ml, theta_lbfgs, alpha_lbfgs
    :param n_cells: int
             Number of cells to use for estimating parameters in Step1: default is 5000
    :param n_genes: int
             Number of genes to use for estimating parameters in Step1; default is 2000
    :param filter_hvgs: bool
    :param res_clip_range: string or list
                    options: 1)"seurat": Clips residuals to -sqrt(ncells/30), sqrt(ncells/30)
                             2)"default": Clips residuals to -sqrt(ncells), sqrt(ncells)
                    only used when filter_hvgs is true
    :param var_features_n: int
                    Number of variable features to select (for calculating a subset of pearson residuals)

    :return: stereoExpData object
    """
    if not issparse(data.exp_matrix):
        data.exp_matrix = csr_matrix(data.exp_matrix)
    exclude_poisson = False
    vst_out = vst(
        data.exp_matrix.T,
        gene_names=data.gene_names.tolist(),
        cell_names=data.cell_names.tolist(),
        method=method,
        n_cells=n_cells,
        n_genes=n_genes,
        threads=4,
        exclude_poisson=exclude_poisson,
    )
    residuals = vst_out['residuals'].T
    if filter_hvgs:
        residuals = get_hvg_residuals(vst_out, var_features_n, res_clip_range)
    # return_data = StereoExpData(
    #     exp_matrix=residuals.values,
    #     position=data.partitions,
    #
    # )
    data.exp_matrix = residuals.values
    gene_index = np.isin(data.gene_names, np.array(residuals.columns))
    data.genes = data.genes.sub_set(gene_index)
    return data, vst_out
