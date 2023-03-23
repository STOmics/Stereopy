#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: regulatory_network.py
@time: 2023/Jan/08
@description: inference gene regulatory networks
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI

change log:
    2023/01/08 init
"""

# python core modules
import os
import csv
import warnings
from typing import Union

# third party modules
import json
import glob
import anndata
import scipy.sparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from arboreto.utils import load_tf_names
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
from stereo.log_manager import logger
from stereo.plots.plot_base import PlotBase
from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.io.reader import read_gef
from stereo.core.stereo_exp_data import StereoExpData
from stereo.plots.scatter import base_scatter


def _name(fname: str) -> str:
    """
    Extract file name (without path and extension)
    :param fname:
    :return:
    """
    return os.path.splitext(os.path.basename(fname))[0]

# TODO 适配pandas1.5.3版本（目前只适配1.3.4版本）
class InferenceRegulatoryNetwork(AlgorithmBase):
    """
    Algorithms to inference Gene Regulatory Networks (GRN)
    """

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     # input
    #     self._data = self.stereo_exp_data
    #     self._matrix = None  # pd.DataFrame
    #     self._gene_names = []
    #     self._cell_names = []

    #     self.load_data_info()

    #     self._tfs = []

    #     # network calculated attributes
    #     self._regulon_list = None  # list
    #     self._auc_mtx = None
    #     self._adjacencies = None  # pd.DataFrame
    #     self._regulon_dict = None

    #     # other settings
    #     # self._num_workers = num_workers
    #     # self._thld = auc_thld
    
    # GRN pipeline main logic
    def main(self,
             database: str = None,
             motif_anno_fn: str= None,
             tfs_fn: Union[str, list]=None,
             target_genes: list=None,
             auc_threshold: float=0.5, 
             num_workers: int=None,
             res_key: str = 'inference_regulatory_network',
             seed: int = None,
             cache: bool = False,
             sn: str=None,
             save: bool=True):
        """
        :param database: the sequence of databases.
        :param motif_anno_fn: the name of the file that contains the motif annotations to use.
        :param tfs_fn: list of target transcription factors. If None or 'all', the list of gene_names will be used.
        :param target_genes: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
            'expression_data' instead of a DataFrame
        :param auc_threshold: the fraction of the ranked genome to take into account for the calculation of the Area Under the recovery Curve.
        :param num_workers: if not using a cluster, the number of workers to use for the calculation. None of all available CPUs need to be used.
        :param res_key: the key for storage of inference regulatory network result.
        :param seed: optional random seed for the regressors. Default None.
        :param cache: whether to use cache files. Need to provide adj.csv, motifs.csv and auc.csv.
        :param sn: sample name.
        :param save: whether to save the result as a file.
        :return: Computation result of inference regulatory network is stored in self.result where the result key is 'inference_regulatory_network'.
        """
        matrix = self.stereo_exp_data.to_df()
        df = self.stereo_exp_data.to_df()

        if num_workers is None:
            num_workers = cpu_count()

        if target_genes is None:
            target_genes = self.stereo_exp_data.gene_names

        # 1. load TF list
        if tfs_fn is None:
            tfs = 'all'
        elif tfs_fn == 'all':
            tfs = 'all'
        elif isinstance(tfs_fn,list):
            tfs = tfs_fn
        elif os.path.isfile(tfs_fn):
            tfs = self.load_tfs(tfs_fn)

        # 2. load the ranking database
        dbs = self.load_database(database)
        # 3. GRN inference
        adjacencies = self.grn_inference(matrix, genes=target_genes, tf_names=tfs, num_workers=num_workers, seed=seed, cache=cache, res_key=res_key)
        modules = self.get_modules(adjacencies, df)
        # 4. Regulons prediction aka cisTarget
        regulons = self.prune_modules(modules, dbs, motif_anno_fn, num_workers, cache=cache, sn=sn)
        self.regulon_dict = get_regulon_dict(regulons)
        # 5: Cellular enrichment (aka AUCell)
        auc_matrix = self.auc_activity_level(df, regulons, auc_threshold, num_workers, seed=seed, cache=cache, sn=sn)

        # save results
        # TODO 将结果保存到h5ad里
        self.pipeline_res[res_key] = {
            'regulons': self.regulon_dict, 
            'auc_matrix': auc_matrix, 
            'adjacencies': adjacencies
            }
        self.stereo_exp_data.tl.reset_key_record('inference_regulatory_network', res_key)

        if save:
            self.regulons_to_csv(regulons)
            #self.regulons_to_json(regulons)
            self.to_loom(df, auc_matrix, regulons)
            #self.to_cytoscape(regulons, adjacencies, 'Zfp354c')

    @staticmethod
    def read_motif_file(fname):
        """

        :param fname:
        :return:
        """
        df = pd.read_csv(fname, sep=',', index_col=[0, 1], header=[0, 1], skipinitialspace=True)
        df[('Enrichment', 'Context')] = df[('Enrichment', 'Context')].apply(lambda s: eval(s))
        df[('Enrichment', 'TargetGenes')] = df[('Enrichment', 'TargetGenes')].apply(lambda s: eval(s))
        return df

    @staticmethod
    def load_tfs(fn: str) -> list:
        """

        :param fn:
        :return:
        """
        tfs_in_file = load_tf_names(fn)

        return tfs_in_file

    # Gene Regulatory Network inference methods
    @staticmethod
    def _set_client(num_workers: int) -> Client:
        """

        :param num_workers:
        :return:
        """
        local_cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
        custom_client = Client(local_cluster)
        return custom_client

    def grn_inference(self,
                      matrix,
                      tf_names,
                      genes,
                      num_workers: int,
                      verbose: bool = True,
                      seed: int = None,
                      cache: bool = True,
                      res_key: str = 'inference_regulatory_network',
                      **kwargs) -> pd.DataFrame:
        """
        Inference of co-expression modules
        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param tf_names: list of target transcription factors. If None or 'all', the list of gene_names will be used.
        :param genes: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
            'expression_data' instead of a DataFrame
        :param num_workers: number of thread
        :param verbose: print info
        :param seed: optional random seed for the regressors. Default None.
        :param cache:
        :param save: if save adjacencies result into a file
        :param sn: sample name. Save adjacencies result in sn.adj.csv.
        :return:

        Example:

        """
        #TODO 需要将csv文件读取改为h5ad文件或result里读取

        if cache and (adjacencies in self.pipeline_res[res_key].keys()):
            logger.info(f'cached file {res_key}["adjacencies"] found')
            adjacencies = self.pipeline_res[res_key]['adjacencies']
            self.adjacencies = adjacencies
            return adjacencies
        else:
            logger.info('cached file not found, running grnboost2 now')

        if num_workers is None:
            num_workers = cpu_count()
        custom_client = InferenceRegulatoryNetwork._set_client(num_workers)
        adjacencies = grnboost2(matrix,
                                tf_names=tf_names,
                                gene_names=genes,
                                verbose=verbose,
                                client_or_address=custom_client,
                                seed=seed,
                                **kwargs)

        self.adjacencies = adjacencies
        return adjacencies

    # def uniq_genes(self, adjacencies):
    #     """
    #     Detect unique genes
    #     :param adjacencies:
    #     :return:
    #     """
    #     df = self._data.to_df()
    #     unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(df.columns)
    #     logger.info(f'find {len(unique_adj_genes) / len(set(df.columns))} unique genes')
    #     return unique_adj_genes

    @staticmethod
    def load_database(database_dir: str) -> list:
        """
        Load ranked database
        :param database_dir:
        :return:
        """
        logger.info('Loading ranked database...')
        db_fnames = glob.glob(database_dir)
        dbs = [RankingDatabase(fname=fname, name=_name(fname)) for fname in db_fnames]
        return dbs

    def get_modules(self,
                    adjacencies: pd.DataFrame,
                    matrix,
                    rho_mask_dropouts: bool = False,
                    **kwargs):
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
            modules_from_adjacencies(adjacencies, matrix, rho_mask_dropouts=rho_mask_dropouts, **kwargs)
        )
        return modules

    def prune_modules(self,
                      modules: list,
                      dbs: list,
                      motif_anno_fn: str,
                      num_workers: int = None,
                      cache: bool = True,
                      save: bool = True,
                      sn: str = None,
                      **kwargs):
        """
        First, calculate a list of enriched motifs and the corresponding target genes for all modules.
        Then, create regulon_list from this table of enriched motifs.
        :param modules: the sequence of modules.
        :param dbs: the sequence of databases.
        :param motif_anno_fn: the name of the file that contains the motif annotations to use.
        :param num_workers: if not using a cluster, the number of workers to use for the calculation. None of all available CPUs need to be used.
        :param cache: 
        :param save:
        :param fn:
        :return:
        """
        fn = sn + 'motifs.csv'

        if cache and os.path.isfile(fn):
            logger.info(f'cached file {fn} found')
            df = self.read_motif_file(fn)
            regulon_list = df2regulons(df)
            # alternative:
            # regulon_list = load_signatures(fn)
            self.regulon_list = regulon_list
            return regulon_list
        else:
            logger.info('cached file not found, running prune modules now')

        if num_workers is None:
            num_workers = cpu_count()

        with ProgressBar():
            df = prune2df(dbs, modules, motif_anno_fn, num_workers=num_workers, **kwargs)
            df.to_csv(fn)  # motifs filename
            
        regulon_list = df2regulons(df)
        self.regulon_list = regulon_list

        if save:
            self.regulons_to_json(regulon_list)

        # alternative way of getting regulon_list, without creating df first
        # regulon_list = prune(dbs, modules, motif_anno_fn)
        return regulon_list

    def auc_activity_level(self,
                           matrix,
                           regulons: list,
                           auc_threshold: float,
                           num_workers: int,
                           seed=None,
                           cache: bool = True,
                           save: bool = True,
                           sn: str=None,
                           **kwargs) -> pd.DataFrame:
        """

        :param the expression matrix (n_cells x n_genes):
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param regulons: list of ctxcore.genesig.Regulon objects. The gene signatures or regulons.
        :param auc_threshold: the fraction of the ranked genome to take into account for the calculation of the Area Under the recovery Curve.
        :param num_workers: the number of cores to use.
        :param cache:
        :param save:
        :param fn:
        :return:
        """
        fn = sn + 'auc.csv'

        if cache and os.path.isfile(fn):
            logger.info(f'cached file {fn} found')
            auc_mtx = pd.read_csv(fn, index_col=0)
            self.auc_mtx = auc_mtx
            return auc_mtx
        else:
            logger.info('cached file not found, calculating auc_activity_level now')

        if num_workers is None:
            num_workers = cpu_count()

        auc_mtx = aucell(matrix, regulons, auc_threshold=auc_threshold, num_workers=num_workers, seed=seed, **kwargs)
        self.auc_mtx = auc_mtx

        if save:
            auc_mtx.to_csv(fn)
        return auc_mtx

    # Results saving methods
    def regulons_to_json(self, regulon_list: list, fn='regulons.json'):
        """
        Write regulon dictionary into json file
        :param regulon_list:
        :param fn:
        :return:
        """
        regulon_dict = get_regulon_dict(regulon_list)
        with open(fn, 'w') as f:
            json.dump(regulon_dict, f, indent=4)

    def regulons_to_csv(self, regulon_list: list, fn: str = 'regulon_list.csv'):
        """
        Save regulon_list (df2regulons output) into a csv file.
        :param regulon_list:
        :param fn:
        :return:
        """
        regulon_dict = get_regulon_dict(regulon_list)
        # Optional: join list of target genes
        for key in regulon_dict.keys(): regulon_dict[key] = ";".join(regulon_dict[key])
        # Write to csv file
        with open(fn, 'w') as f:
            w = csv.writer(f)
            w.writerow(["Regulons", "Target_genes"])
            w.writerows(regulon_dict.items())

    def to_loom(self, matrix: pd.DataFrame, auc_matrix: pd.DataFrame, regulons: list, loom_fn: str = 'output.loom'):
        """
        Save GRN results in one loom file
        :param matrix:
        :param auc_matrix:
        :param regulons:
        :param loom_fn:
        :return:
        """
        export2loom(ex_mtx=matrix, auc_mtx=auc_matrix,
                    regulons=[r.rename(r.name.replace('(+)', ' (' + str(len(r)) + 'g)')) for r in regulons],
                    out_fname=loom_fn)

    def to_cytoscape(self,
                     regulons: list,
                     adjacencies: pd.DataFrame,
                     tf: str,
                     fn: str = 'cytoscape.txt'):
        """
        Save GRN result of one TF, into Cytoscape format for down stream analysis
        :param regulons: list of regulon objects, output of prune step
        :param adjacencies: adjacencies matrix
        :param tf: one target TF name
        :param fn: output file name
        :return:

        Example:
            grn.to_cytoscape(regulons, adjacencies, 'Gnb4', 'Gnb4_cytoscape.txt')
        """
        # get TF data
        regulon_dict = get_regulon_dict(regulons)
        sub_adj = adjacencies[adjacencies.TF == tf]
        targets = regulon_dict[f'{tf}(+)']
        # all the target genes of the TF
        sub_df = sub_adj[sub_adj.target.isin(targets)]
        sub_df.to_csv(fn, index=False, sep='\t')


class PlotRegulatoryNetwork(PlotBase):
    """
    Plot Gene Regulatory Networks related plots
    """

    # dotplot method for StereoExpData
    @staticmethod
    def _cal_percent_exp_df(exp_matrix: pd.DataFrame,
                        cluster_meta: pd.DataFrame,
                        regulon_genes: str,
                        celltype: list,
                        groupby: str='group',
                        cell_label: str='bins',
                        cutoff: float = 0):
        """
        Expression percent
        cell numbers
        :param exp_matrix:
        :param cluster_meta:
        :param regulon_genes:
        :param celltype:
        :param cutoff:
        :return:
        """
        # which cells are in cluster X
        cells = cluster_meta[cluster_meta[groupby] == celltype][cell_label]
        ncells = set(exp_matrix.index).intersection(set(cells))
        # get expression data for cells
        ct_exp = exp_matrix.loc[ncells]
        # input genes in regulon Y
        # get expression data for regulon Y genes in cluster X cells
        g_ct_exp = ct_exp[regulon_genes]
        # count regulon totol expression value
        g_ct_exp['total'] = g_ct_exp.sum(axis=1)
        # count the number of genes which expressed in cluster X cells
        regulon_cell_num = g_ct_exp['total'][g_ct_exp['total'] > cutoff].count()
        total_cell_num = g_ct_exp.shape[0]
        if total_cell_num == 0:
            return 0
        else:
            reg_ct_percent = regulon_cell_num / total_cell_num
            reg_ct_avg_exp = np.mean(g_ct_exp['total'])
            return round(reg_ct_percent,2), round(reg_ct_avg_exp,2)

    def grn_dotplot(self,
                       meta: pd.DataFrame,
                       regulon_names: Union[str, list] = None,
                       celltypes: Union[str, list] = None,
                       groupby: str = 'group',
                       cell_label: str = 'bins',
                       ign_res_key: str = 'inference_regulatory_network', 
                       palette: str = 'Reds',
                       **kwargs):
        """
        Intuitive way of visualizing how feature expression changes across different
        identity classes (clusters). The size of the dot encodes the percentage of
        cells within a class, while the color encodes the AverageExpression level
        across all cells within a class (red is high).

        :param meta: cell classification information.
        :param regulon_names: the regulon which would be shown on plot, defaults to None.
            If set it to None, it will be set to all regulon.
            1) string: only one cluster.
            2) list: an array contains the regulon which would be shown.
        :param celltypes: the celltypes in cluster pairs which would be shown on plot, defaults to None. 
            If set it to None, it will be set to all clusters.
            1) string: only one cluster.
            2) list: an array contains the clusters which would be shown.
        :param groupby: cell type label.
        :param cell_label: cell bin label.
        :param ign_res_key: the key which specifies inference regulatory network result
             in data.tl.result, defaults to 'inference_regulatory_network'
        :param palette: Color theme, defaults to 'Reds'
        :param kwargs: features Input vector of features, or named list of feature vectors
        
        :return: matplotlib.figure
        """
        if ign_res_key not in self.pipeline_res:
            logger.info(f"The result specified by {ign_res_key} is not exists.")

        expr_matrix = self.stereo_exp_data.to_df()
        dot_data = {'cell type': [], 'regulons': [], 'percentage': [], 'avg exp': []}

        regulon_dict = self.pipeline_res[ign_res_key]['regulons']

        if celltypes is None:
            meta_new = meta.drop_duplicates(subset='group')
            celltypes = sorted(meta_new['group'])
        elif isinstance(celltypes, str) and celltypes.upper() == 'ALL':
            meta_new = meta.drop_duplicates(subset='group')
            celltypes = sorted(meta_new['group'])
        elif isinstance(celltypes, str) and celltypes.upper() != 'ALL':
            celltypes = [celltypes]

        if regulon_names is None:
            regulon_names = regulon_dict.keys()
        elif isinstance(regulon_names, str) and regulon_names.upper() == 'ALL':
            regulon_names = regulon_dict.keys()
        elif isinstance(regulon_names, str) and regulon_names.upper() != 'ALL':
            regulon_names = [regulon_names]

        for reg in regulon_names:
            if '(+)' not in reg:
                reg = reg + '(+)'
            target_genes = regulon_dict[f'{reg}']
            for ct in celltypes:
                reg_ct_percent, reg_ct_avg_exp = PlotRegulatoryNetwork._cal_percent_exp_df(exp_matrix=expr_matrix,
                                                                       cluster_meta=meta,
                                                                       regulon_genes=target_genes,
                                                                       celltype=ct, 
                                                                       groupby=groupby,
                                                                       cell_label=cell_label)

                dot_data['regulons'].append(reg)
                dot_data['cell type'].append(ct)
                dot_data['percentage'].append(reg_ct_percent)
                dot_data['avg exp'].append(reg_ct_avg_exp)

        dot_df = pd.DataFrame(dot_data)

        # plotting
        ncols = len(dot_df['regulons'].unique())
        nrows = len(dot_df['cell type'].unique())

        width, height = int(5 + max(3, ncols * 0.8)), int(3 + max(5, nrows * 0.5))

        fig, ax = plt.subplots(figsize=(width, height))
        fig = sns.scatterplot(data=dot_df, size='percentage', hue='avg exp', x='regulons', y='cell type', sizes=(100, 300),
                            marker='o', palette=palette, legend='auto', ax=ax, **kwargs)
        ax.legend(fontsize=12, frameon=False, ncol=1, loc=(1.02, 0))
        ax.tick_params(axis='x', labelsize=12, labelrotation=90)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xlabel('Regulon')
        ax.set_ylabel('Cell type')
        return fig

    def auc_heatmap(
            self, 
            ign_res_key = 'inference_regulatory_network', 
            width=8, 
            height=8, 
            **kwargs):
        """
        Plot heatmap for auc value for regulons
        :param ign_res_key: the key which specifies inference regulatory network result
             in data.tl.result, defaults to 'inference_regulatory_network'
        :param height: height of drawing
        :param width: width of drawing

        :return: matplotlib.figure
        """
        logger.info('Generating auc heatmap plot')

        if ign_res_key not in self.pipeline_res:
            logger.info(f"The result specified by {ign_res_key} is not exists.")
        
        fig = sns.clustermap(
            self.pipeline_res[ign_res_key]['auc_matrix'], 
            figsize=(width,height),
            dendrogram_ratio=(.1, .2),
            cbar_pos=(-.05, .2, .03, .4),
            **kwargs)
        
        return fig

    def spatial_scatter_by_regulon(
            self, 
            ign_res_key: str='inference_regulatory_network', 
            reg_name: str=None, 
            dot_size: int=None,
            palette: str='CET_L4',
            **kwargs):
        """
        Plot genes of one regulon on a 2D map

        :param ign_res_key: the key which specifies inference regulatory network result
             in data.tl.result, defaults to 'inference_regulatory_network'
        :param reg_name: specify the regulon you want to draw, defaults to None, if none, will select randomly.
        :param dot_size: marker size, defaults to None
        :param palette: Color theme, defaults to 'CET_L4'

        :return: matplotlib.figure
        """
        logger.info(f'Please adjust the dot_size to prevent dots from covering each other')

        if ign_res_key not in self.pipeline_res:
            logger.info(f"The result specified by {ign_res_key} is not exists.")

        if reg_name is None:
            regulon_dict = self.pipeline_res[ign_res_key]['regulons']
            reg_name = list(regulon_dict.keys())[0]
        elif '(+)' not in reg_name:
            reg_name = reg_name + '(+)'
        cell_coor = self.stereo_exp_data.position
        # prepare plotting data
        auc_zscore = cal_zscore(self.pipeline_res[ign_res_key]['auc_matrix'][reg_name])
        # sort data points by zscore (low to high), because first dot will be covered by latter dots
        df = pd.DataFrame({'x':cell_coor[:, 0],'y':cell_coor[:, 1],'auc_zscore':auc_zscore})
        df.sort_values(by=['auc_zscore'],inplace=True)
        # plot cell/bin dot, x y coor
        if 'color_bar_reverse' in kwargs:
            color_bar_reverse = kwargs['color_bar_reverse']
            del kwargs['color_bar_reverse']
        else:
            color_bar_reverse = False
        
        fig = base_scatter(
            x = df['x'],
            y = df['y'],
            hue = df['auc_zscore'],
            title = reg_name,
            x_label='spatial1',
            y_label='spatial2',
            dot_size=dot_size,
            palette=palette,
            color_bar=True,
            color_bar_reverse=color_bar_reverse,
            **kwargs
        )

        return fig
        
    @staticmethod
    def plot_2d_reg_h5ad(data: anndata.AnnData, pos_label, auc_mtx, reg_name: str, **kwargs):
        """
        Plot genes of one regulon on a 2D map
        :param pos_label:
        :param data:
        :param auc_mtx:
        :param reg_name:
        :return:

        Example:
            plot_2d_reg_h5ad(data, 'spatial', auc_mtx, 'Zfp354c')
        """
        if '(+)' not in reg_name:
            reg_name = reg_name + '(+)'
        cell_coor = data.obsm[pos_label]
        auc_zscore = cal_zscore(auc_mtx)
        # prepare plotting data
        sub_zscore = auc_zscore[reg_name]
        # sort data points by zscore (low to high), because first dot will be covered by latter dots
        zorder = np.argsort(sub_zscore[reg_name].values)
        # plot cell/bin dot, x y coor
        sc = plt.scatter(cell_coor[:, 0][zorder], cell_coor[:, 1][zorder], c=sub_zscore[reg_name][zorder], marker='.',
                         edgecolors='none', cmap='plasma', lw=0, **kwargs)
        plt.box(False)
        plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        plt.savefig(f'{reg_name.split("(")[0]}.png')
        plt.close()

    def auc_heatmap_by_group(self,
                    ign_res_key: str = 'inference_regulatory_network', 
                    celltype_res_key: str = 'leiden',
                    top_n_feature: int=5,
                    width: int=18,
                    height: int=28,
                    **kwargs):
        """
        Plot heatmap for Regulon specificity scores (RSS) value
        :param auc_mtx: 
        :param regulons:
        :param meta:
        :param save:
        :param fn:
        :return: 
        """
    
        if ign_res_key not in self.pipeline_res:
            logger.info(f"The result specified by {ign_res_key} is not exists.")
        elif celltype_res_key not in self.pipeline_res:
            logger.info(f"The result specified by {celltype_res_key} is not exists.")

        auc_mtx = self.pipeline_res[ign_res_key]['auc_matrix']
        
        meta = self.pipeline_res[celltype_res_key]
        print(meta)
        # Regulon specificity scores (RSS) across predicted cell types
        rss_cellType = regulon_specificity_scores(auc_mtx, meta['group'])
        # rss_cellType.to_csv('regulon_specificity_scores.txt')
        # Select the top 5 regulon_list from each cell type
        cats = sorted(list(set(meta)))
        topreg = []
        for i, c in enumerate(cats):
            topreg.extend(
                list(rss_cellType.T[c].sort_values(ascending=False)[:top_n_feature].index)
            )
        topreg = list(set(topreg))

        # plot z-score
        auc_zscore = cal_zscore(auc_mtx)
        sns.set(font_scale=1.2)

        # set group color
        lut = dict(zip(meta['group'].unique(), ncolors(len(meta['group'].unique()))))
        row_colors = meta['group'].map(lut)
        meta['group'] = row_colors

        g = sns.clustermap(
            auc_zscore[topreg], 
            row_colors=meta.set_index(['bins']),
            figsize=(width,height),
            dendrogram_ratio=(.1, .2),
            cbar_pos=(-.05, .2, .03, .4),
            **kwargs
        )

        return g
    
def get_n_hls_colors(num):
    import random
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def ncolors(num):
    import colorsys
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append(rgb_to_hex(r,g,b))

        return rgb_colors
    
def get_regulon_dict(regulon_list: list) -> dict:
    """
    Form dictionary of { TF : Target } pairs from 'pyscenic ctx' output.
    :param regulon_list:
    :return:
    """
    regulon_dict = {}
    for reg in regulon_list:
        targets = [target for target in reg.gene2weight]
        regulon_dict[reg.name] = targets
    return regulon_dict


def cal_zscore(auc_mtx: pd.DataFrame) -> pd.DataFrame:
    """
    calculate z-score for each gene among cells
    :param auc_mtx:
    :return:
    """
    func = lambda x: (x - x.mean()) / x.std(ddof=0)
    auc_zscore = auc_mtx.transform(func, axis=0)
    return auc_zscore


def is_regulon(reg):
    """
    Decide if a string is a regulon_list name
    :param reg: the name of the regulon
    :return:
    """
    if '(+)' in reg or '(-)' in reg:
        return True
