#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@file: regulatory_network.py
@time: 2023/Jan/08
@description: inference gene regulatory networks
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI

change log:
    2023/01/08 init
'''

# python core modules
import os
import csv
import warnings
from typing import Union

# third party modules
import glob
import anndata
import scipy.sparse
import pandas as pd
import numpy as np
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from pyscenic.export import export2loom
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.prune import prune2df, df2regulons
from pyscenic.utils import modules_from_adjacencies
from pyscenic.cli.utils import load_signatures
from pyscenic.rss import regulon_specificity_scores
from pyscenic.aucell import aucell

# modules in self project
from ..log_manager import LogManager
from .algorithm_base import AlgorithmBase
from stereo.io.reader import read_gef
from ..plots.plot_base import PlotBase
from stereo.core.stereo_exp_data import StereoExpData


class InferenceRegulatoryNetwork(AlgorithmBase):
    """
    A gene regulatory network
    """

    logger = LogManager(log_path='project.log', level='debug').get_logger(name='Stereo')

    def __init__(self):
        # input
        self._data = None
        self._matrix = None  # pd.DataFrame
        # network calculated attributes
        self._regulons = None  # list, check
        self._auc_mtx = None  # check
        self._adjacencies = None  # pd.DataFrame

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @property
    def adjacencies(self):
        return self._adjacencies

    @adjacencies.setter
    def genes(self, value):
        self._adjacencies = value

    @property
    def regulons(self):
        return self._regulons

    @regulons.setter
    def regulons(self, value):
        self._regulons = value

    @property
    def auc_mtx(self):
        return self._auc_mtx

    @auc_mtx.setter
    def auc_mtx(self, value):
        self._auc_mtx = value

    @staticmethod
    def is_valid_exp_matrix(mtx: pd.DataFrame):
        """
        check if the exp matrix is valid for the grn pipeline
        :param mtx:
        :return:
        """
        return (all(isinstance(idx, str) for idx in mtx.index)
                and all(isinstance(idx, str) for idx in mtx.columns)
                and (mtx.index.nlevels == 1)
                and (mtx.columns.nlevels == 1))

    def set_data(self, data: Union[StereoExpData, anndata.AnnData]):
        """

        :param data:
        :return:
        """
        self.data = data
        if isinstance(data, StereoExpData):
            self.matrix = data.exp_matrix
            return data.exp_matrix, data.gene_names
        elif isinstance(data, anndata.AnnData):
            self.matrix = data.X
            return data.X, data.var_names

    @classmethod
    def load_anndata_by_cluster(cls, data: anndata.AnnData,
                                cluster_label: str,
                                target_clusters: list) -> anndata.AnnData:
        """

        :param data:
        :param cluster_label:
        :param target_clusters:
        :return:
        """
        if isinstance(data, anndata.AnnData):
            return data[data.obs[cluster_label].isin(target_clusters)]

    @classmethod
    def load_data_by_cluster(cls, data: StereoExpData,
                             meta: pd.DataFrame,
                             cluster_label: str,
                             target_clusters: list) -> scipy.sparse.csc_matrix:
        """

        :param cluster_label:
        :param data:
        :param meta:
        :param target_clusters:
        :return:
        """
        return data.exp_matrix[meta[cluster_label].isin(target_clusters)]

    @classmethod
    def read_file(cls, fn: str, bin_type='cell_bins'):
        """
        Loading input files, supported file formats:
            * gef
            * gem
            * loom
            * h5ad
            * csv
        Recommended formats:
            * h5ad
            * gef
        :param fn:
        :param bin_type:
        :return:
        """
        # self.logger.info('Loading expression data...')
        extension = os.path.splitext(fn)[1]
        if extension == '.csv':
            cls.mtx = pd.read_csv(fn)
            cls.genes = list(cls.mtx.columns)
            # self.logger.info(f'is valid expr matrix {RegulatoryNetwork.is_valid_exp_matrix(self.mtx)}')
            return cls.mtx, cls.genes
        elif extension == '.loom':
            cls.data = sc.read_loom(fn)
            cls.genes = list(cls.data.var_names)
            cls.mtx = cls.data.X
            return cls.mtx, cls.genes
        elif extension == 'h5ad':
            cls.data = sc.read_h5ad(fn)
            cls.genes = list(cls.data.var_names)
            cls.mtx = cls.data.X
            return cls.mtx, cls.genes
        elif extension == '.gef':
            cls.data = read_gef(file_path=fn, bin_type=bin_type)
            cls.genes = cls.data.gene_names
            cls.mtx = cls.data.to_df()
            return cls.mtx, cls.genes

    @staticmethod
    def _set_client(num_workers: int) -> Client:
        """

        :param num_workers:
        :return:
        """
        local_cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
        custom_client = Client(local_cluster)
        return custom_client

    @classmethod
    def grn_inference(cls,
                      matrix,
                      tf_names: list,
                      genes,
                      num_workers: int,
                      verbose: bool = True,
                      fn: str = 'adj.csv') -> pd.DataFrame:
        """
        Inference of co-expression modules

        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param tf_names:
        :param genes:
        :param num_workers:
        :param verbose:
        :param fn:
        :return:
        """
        if num_workers is None:
            num_workers = cpu_count()
        custom_client = InferenceRegulatoryNetwork._set_client(num_workers)
        adjacencies = grnboost2(matrix,
                                tf_names=tf_names,
                                gene_names=genes,
                                verbose=verbose,
                                client_or_address=custom_client)
        adjacencies.to_csv(fn, index=False)  # adj.csv, don't have to save into a file
        return adjacencies

    @staticmethod
    def _name(fname: str) -> str:
        """

        :param fname:
        :return:
        """
        return os.path.splitext(os.path.basename(fname))[0]

    @classmethod
    def load_database(cls, database_dir: str) -> list:
        """

        :param database_dir:
        :return:
        """
        # cls.logger.info('Loading ranked databases...')
        db_fnames = glob.glob(database_dir)
        dbs = [RankingDatabase(fname=fname, name=InferenceRegulatoryNetwork._name(fname)) for fname in db_fnames]
        return dbs

    @classmethod
    def load_tfs(cls, fn: str) -> list:
        """

        :param fn:
        :return:
        """
        with open(fn) as file:
            tfs_in_file = [line.strip() for line in file.readlines()]
        return tfs_in_file

    @classmethod
    def ctx_get_regulons(cls,
                         adjacencies,
                         matrix,
                         rho_mask_dropouts: bool = False):
        """
        Inference of co-expression modules

        :param adjacencies:
        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param rho_mask_dropouts:
        :return:
        """
        modules = list(
            modules_from_adjacencies(adjacencies, matrix, rho_mask_dropouts=rho_mask_dropouts)
        )
        return modules

    @classmethod
    def prune_modules(cls,
                      modules,
                      dbs: list,
                      motif_anno_fn,
                      num_workers: int,
                      is_prune: bool = True,
                      rgn: str = 'regulons.csv'):
        """

        :param dbs:
        :param motif_anno_fn:
        :param num_workers:
        :param is_prune:
        :param rgn:
        :return:
        """
        if num_workers is None:
            num_workers = cpu_count()
        if is_prune:
            with ProgressBar():
                df = prune2df(dbs, modules, motif_anno_fn, num_workers=num_workers)
                df.to_csv(rgn)  # motifs filename
            regulons = df2regulons(df)
            # alternative way of getting regulons, without creating df first
            # regulons = prune(dbs, modules, motif_anno_fn)
            return regulons
        else:
            warnings.warn('if prune_modules is set to False')

    @classmethod
    def regulons_to_csv(cls, regulons, fn: str = 'regulons.csv'):
        """
        Save regulons (df2regulons output) into a csv file.
        :param fn:
        :return:
        """
        rdict = {}
        for reg in regulons:
            targets = [target for target in reg.gene2weight]
            rdict[reg.name] = targets
        # Optional: join list of target genes
        for key in rdict.keys(): rdict[key] = ";".join(rdict[key])
        # Write to csv file
        with open(fn, 'w') as f:
            w = csv.writer(f)
            w.writerow(["Regulons", "Target_genes"])
            w.writerows(rdict.items())

    @classmethod
    def auc_activity_level(cls,
                           matrix,
                           regulons,
                           auc_threshold: float,
                           num_workers: int,
                           save: bool = True,
                           fn='auc.csv') -> pd.DataFrame:
        """

        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param regulons:
        :param auc_threshold:
        :param num_workers:
        :param save:
        :param fn:
        :return:
        """
        if num_workers is None:
            num_workers = cpu_count()
        auc_mtx = aucell(matrix, regulons, auc_threshold=auc_threshold, num_workers=num_workers)
        if save:
            auc_mtx.to_csv(fn)
        return auc_mtx

    @classmethod
    def save_to_loom(cls, matrix, auc_matrix, regulons, loom_fn: str = 'output.loom'):
        """

        :param matrix:
        :param auc_matrix:
        :param regulons:
        :param loom_fn:
        :return:
        """
        export2loom(ex_mtx=matrix, auc_mtx=auc_matrix,
                    regulons=[r.rename(r.name.replace('(+)', ' (' + str(len(r)) + 'g)')) for r in regulons],
                    out_fname=loom_fn)

    @classmethod
    def save_to_cytoscape(cls,
                          regulons: list,
                          adjacencies: pd.DataFrame,
                          tf: str,
                          fn: str = 'cyto.txt'):
        """

        :param regulons:
        :param adjacencies:
        :param tf:
        :param target_genes:
        :param fn:
        :return:
        """
        # get TF data
        sub_adj = adjacencies[adjacencies.TF == tf]
        targets = regulons[tf]
        # all the target genes of the TF
        sub_df = sub_adj[sub_adj.target.isin(targets)]
        sub_df.to_csv(fn, index=False, sep='\t')

    @classmethod
    def uniq_genes(cls, adjacencies, matrix: pd.DataFrame):
        """

        :param adjacencies:
        :param matrix:
        :return:
        """
        assert isinstance(matrix, pd.DataFrame)

        unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(matrix.columns)
        # RegulatoryNetwork.logger.info(f'find {len(unique_adj_genes) / len(set(cls.mtx.columns))} unique genes')
        return unique_adj_genes

    @classmethod
    def main(self,
             data: Union[StereoExpData, anndata.AnnData],
             databases: str,
             motif_anno_fn: str,
             tfs_fn,
             num_workers: int,
             target_genes):
        """"""
        matrix, genes = self.set_data(data)

        # assert isinstance(matrix, scipy.sparse.csc_matrix)

        if target_genes is None:
            target_genes = genes

        # 1. load TF list
        if tfs_fn is None:
            tfs = 'all'
        else:
            tfs = self.load_tfs(tfs_fn)

        # 2. load the ranking databases
        dbs = self.load_database(databases)
        # 3. GRN inference
        adjacencies = self.grn_inference(matrix, genes=target_genes, tf_names=tfs, num_workers=num_workers)
        # 4. Regulons prediction aka cisTarget
        modules = self.ctx_get_regulons(adjacencies, matrix)
        regulons = self.prune_modules(modules, dbs, motif_anno_fn, num_workers=24)
        # 5: Cellular enrichment (aka AUCell)
        auc_mtx = self.auc_activity_level(matrix, regulons, auc_threshold=0.5, num_workers=num_workers)
        return adjacencies, regulons, auc_mtx


class PlotRegulatoryNetwork(PlotBase):
    @staticmethod
    def _cal_percent_df(exp_matrix, cluster_meta, regulon, ct, cutoff=0):
        """
        Expression percent
        cell numbers
        :param exp_matrix:
        :param cluster_meta:
        :param regulon:
        :param ct:
        :param cutoff:
        :return:
        """
        cells = cluster_meta['cluster' == ct]['cell']
        ct_exp = exp_matrix.iloc(cells)
        g_ct_exp = ct_exp[regulon]
        regulon_cell_num = g_ct_exp[g_ct_exp > cutoff].count()
        total_cell_num = 0
        return regulon_cell_num / total_cell_num

    @staticmethod
    def _cal_exp_df(exp_matrix, cluster_meta, regulon, ct):
        """

        :param exp_matrix:
        :param cluster_meta:
        :param regulon:
        :param ct:
        :return:
        """
        cells = cluster_meta['cluster' == ct]['cell']
        ct_exp = exp_matrix.iloc(cells)
        g_ct_exp = ct_exp[regulon]
        return np.mean(g_ct_exp)

    @staticmethod
    def dotplot_stereo(StereoExpData, **kwargs):
        """
        Intuitive way of visualizing how feature expression changes across different
        identity classes (clusters). The size of the dot encodes the percentage of
        cells within a class, while the color encodes the AverageExpression level
        across all cells within a class (blue is high).

        :param StereoExpData:
        :param kwargs: features Input vector of features, or named list of feature vectors
        if feature-grouped panels are desired
        :return:
        """
        pass

    @staticmethod
    def dotplot_anndata(data: Union[anndata.AnnData, StereoExpData],
                        gene_names,
                        cluster_label: str,
                        save: bool = True):
        """
        create a dotplot for Anndata object.
        a dotplot contains percent (of cells that) expressed (the genes) and average exression (of genes).

        :param data: gene data
        :param gene_names: interested gene names
        :param cluster_label: label of clustering output
        :param save: if save plot into a file
        :return: plt axe object

        e.g.
            there is an anndata
        """
        if isinstance(data, anndata.AnnData):
            return sc.pl.dotplot(data, var_names=gene_names, groupby=cluster_label, save=save)
        elif isinstance(data, StereoExpData):
            print('for StereoExpData object, please use function: dotplot_stereo')

    @staticmethod
    def auc_heatmap(auc_mtx, fn='auc_heatmap.png'):
        """

        :param auc_mtx:
        :param fn:
        :return:
        """
        plt.figsize = (8, 8)
        sns.clustermap(auc_mtx)
        plt.tightlayout()
        plt.savefig(fn)

    @staticmethod
    def plot_reg_2d(auc_zscore, cell_coor, reg_name):
        """
        
        :param auc_zscore:
        :param cell_coor:
        :param reg_name:
        :return:
        """
        # prepare plotting data
        sub_zscore = auc_zscore[['Cell', reg_name]]
        # sort data points by zscore (low to high)
        zorder = np.argsort(sub_zscore[reg_name].values)
        # plot cell/bin dot, x y coor
        sc = plt.scatter(cell_coor['x'][zorder], cell_coor['y'][zorder], c=sub_zscore[reg_name][zorder], marker='.',
                         edgecolors='none', cmap='plasma', lw=0)
        plt.box(False)
        plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        plt.savefig(f'{reg_name.split("(")[0]}.png')
        plt.close()

    @staticmethod
    def multi_reg_2d(auc_zscore, cell_coor, target_regs):
        """

        :param auc_zscore:
        :param cell_coor:
        :param target_regs:
        :return:
        """
        for reg in target_regs:
            if PlotRegulatoryNetwork.is_regulon(reg):
                PlotRegulatoryNetwork.plot_reg_2d(auc_zscore, cell_coor, reg)

    @staticmethod
    def is_regulon(reg):
        """
        Decide if a string is a regulons name
        :param reg:
        :return:
        """
        if '(+)' in reg or '(-)' in reg:
            return True

    @staticmethod
    def rss_heatmap(adata: anndata.AnnData, auc_mtx: pd.DataFrame, meta, regulons_fn='regulons.csv'):
        """
        
        :param adata: 
        :param auc_mtx: 
        :param regulons_fn: 
        :param meta_data: 
        :return: 
        """
        # scenic output
        # lf = lp.connect('out.loom', mode='r', validate=False)  # validate must set to False
        # auc_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)
        # data = read_ann_h5ad('/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/StereopyData/Cellbin_deversion.h5ad')
        # meta = pd.read_csv('meta_mousebrain.csv', index_col=0).iloc[:, 0]

        # load the regulons from a file using the load_signatures function
        sig = load_signatures(regulons_fn)  # regulons_df -> list of regulons
        # TODO: adapt to StereoExpData
        # adata = add_scenic_metadata(adata, auc_mtx, sig)

        ### Regulon specificity scores (RSS) across predicted cell types
        ### Calculate RSS
        rss_cellType = regulon_specificity_scores(auc_mtx, meta)
        # rss_cellType.to_csv('regulon_specificity_scores.txt')

        func = lambda x: (x - x.mean()) / x.std(ddof=0)
        auc_mtx_Z = auc_mtx.transform(func, axis=0)
        auc_mtx_Z.to_csv('pyscenic_output.prune_modules.zscore.csv')

        ### Select the top 5 regulons from each cell type
        cats = sorted(list(set(meta)))
        topreg = []
        for i, c in enumerate(cats):
            topreg.extend(
                list(rss_cellType.T[c].sort_values(ascending=False)[:5].index)
            )
        topreg = list(set(topreg))

        colors = [
            '#d60000', '#e2afaf', '#018700', '#a17569', '#e6a500', '#004b00',
            '#6b004f', '#573b00', '#005659', '#5e7b87', '#0000dd', '#00acc6',
            '#bcb6ff', '#bf03b8', '#645472', '#790000', '#0774d8', '#729a7c',
            '#8287ff', '#ff7ed1', '#8e7b01', '#9e4b00', '#8eba00', '#a57bb8',
            '#5901a3', '#8c3bff', '#a03a52', '#a1c8c8', '#f2007b', '#ff7752',
            '#bac389', '#15e18c', '#60383b', '#546744', '#380000', '#e252ff',
        ]
        # colorsd = dict((f'c{i}', c) for i, c in enumerate(colors))
        # colormap = [colorsd[x] for x in clusters_series]

        sns.set(font_scale=1.2)
        g = sns.clustermap(auc_mtx_Z[topreg], annot=False, square=False, linecolor='gray', yticklabels=True,
                           xticklabels=True, vmin=-2, vmax=6, cmap="YlGnBu", figsize=(21, 16))
        g.cax.set_visible(True)
        g.ax_heatmap.set_ylabel('')
        g.ax_heatmap.set_xlabel('')
        plt.savefig("clusters-heatmap-top5.png")
