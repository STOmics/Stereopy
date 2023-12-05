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

import csv
import glob
# third party modules
import json
# python core modules
import os
from multiprocessing import cpu_count
from typing import Union

import hotspot
import numpy as np
import pandas as pd
from arboreto.algo import grnboost2
from arboreto.utils import load_tf_names
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask.distributed import LocalCluster
from pyscenic.aucell import aucell
from pyscenic.export import export2loom
from pyscenic.prune import df2regulons
from pyscenic.prune import prune2df
from pyscenic.utils import modules_from_adjacencies

from stereo.algorithm.algorithm_base import AlgorithmBase
# modules in self project
from stereo.log_manager import logger


class RegulatoryNetworkInference(AlgorithmBase):
    """
    Algorithms to inference Gene Regulatory Networks (GRN)
    """

    # GRN pipeline main logic
    def main(self,
             database: str = None,
             motif_anno: str = None,
             tfs: Union[str, list] = None,
             target_genes: list = None,
             auc_threshold: float = 0.5,
             num_workers: int = None,
             res_key: str = 'regulatory_network_inference',
             seed: int = None,
             cache: bool = False,
             cache_res_key: str = 'regulatory_network_inference',
             save_regulons: bool = True,
             save_loom: bool = False,
             fn_prefix: str = None,
             method: str = 'grnboost',
             ThreeD_slice: bool = False,
             prune_kwargs: dict = {},
             hotspot_kwargs: dict = {},
             use_raw: bool = True
             ):
        """
        Enables researchers to infer transcription factors (TFs) and gene regulatory networks.

        :param database: the sequence of databases.
        :param motif_anno: the name of the file that contains the motif annotations to use.
        :param tfs: list of target transcription factors. If None or 'all', the list of gene_names will be used.
        :param target_genes: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
            'expression_data' instead of a DataFrame
        :param auc_threshold: the fraction of the ranked genome to take into account for the calculation of the Area Under the recovery Curve.
        :param num_workers: if not using a cluster, the number of workers to use for the calculation. None of all available CPUs need to be used.
        :param res_key: the key for storage of inference regulatory network result.
        :param seed: optional random seed for the regressors. Default None.
        :param cache: whether to use cache files. Need to provide adj.csv, motifs.csv and auc.csv.
        :param save_regulons: whether to save regulons into a csv file.
        :param save_loom: whether to save the result as a loom file.
        :param fn_prefix: the prefix of file name for saving regulons or loom.
        :param method: the method to inference GRN, 'grnboost' or 'hotspot'.
        :param ThreeD_slice: whether to use 3D slice data.
        :param prune_kwargs: dict, other parameters of pyscenic.prune.prune2df.
        :param hotspot_kwargs: dict, other parameters for 'hotspot' method.
        :return: Computation result of inference regulatory network is stored in self.result where the result key is 'regulatory_network_inference'.
        """  # noqa
        self.use_raw = use_raw
        if use_raw and self.stereo_exp_data.raw is None:
            raise Exception("The raw data is not found, you need to run 'raw_checkpoint()' first.")

        if use_raw:
            logger.info('the raw expression matrix will be used.')
            matrix = self.stereo_exp_data.raw.to_df()
        else:
            logger.info('if you have done some normalized processing, the normalized expression matrix will be used.')
            matrix = self.stereo_exp_data.to_df()
            # df = self.stereo_exp_data.to_df()
        df = matrix.copy(deep=True)

        if num_workers is None:
            num_workers = cpu_count()

        if target_genes is None:
            target_genes = self.stereo_exp_data.gene_names

        # 1. load TF list
        if tfs is None:
            tfsf = 'all'
        elif tfs == 'all':
            tfsf = 'all'
        elif isinstance(tfs, list):
            tfsf = tfs
        elif os.path.isfile(tfs):
            tfsf = self.load_tfs(tfs)

        # 2. load the ranking database
        dbs = self.load_database(database)

        # 2.5 check if data is 3D slice
        if ThreeD_slice:
            logger.info('If data belongs to 3D, it only can be runned as hotspot method now')
            method = 'hotspot'

        # 3. GRN inference
        if method == 'grnboost':
            adjacencies = self.grn_inference(matrix, genes=target_genes, tf_names=tfsf, num_workers=num_workers,
                                             seed=seed, cache=cache, cache_res_key=cache_res_key)
        elif method == 'hotspot':
            hotspot_kwargs_adjusted = {}
            for key, value in hotspot_kwargs.items():
                if key in ('tf_list', 'jobs', 'cache', 'cache_res_key', 'ThreeD_slice'):
                    continue
                hotspot_kwargs_adjusted[key] = value
            adjacencies = self.hotspot_matrix(tf_list=tfsf, jobs=num_workers, cache=cache, cache_res_key=cache_res_key,
                                              ThreeD_slice=ThreeD_slice, **hotspot_kwargs_adjusted)

        modules = self.get_modules(adjacencies, df)
        # 4. Regulons prediction aka cisTarget
        regulons, motifs = self.prune_modules(modules, dbs, motif_anno, num_workers, cache=cache,
                                              cache_res_key=cache_res_key, **prune_kwargs)
        self.regulon_dict = get_regulon_dict(regulons)
        # 5: Cellular enrichment (aka AUCell)
        auc_matrix = self.auc_activity_level(df, regulons, auc_threshold, num_workers, seed=seed, cache=cache,
                                             cache_res_key=cache_res_key)

        # save results
        self.pipeline_res[res_key] = {
            'regulons': self.regulon_dict,
            'auc_matrix': auc_matrix,
            'adjacencies': adjacencies,
            # 'motifs': motifs
        }
        self.stereo_exp_data.tl.reset_key_record('regulatory_network_inference', res_key)

        if save_regulons:
            self.regulons_to_csv(regulons, fn_prefix=fn_prefix)
            # self.regulons_to_json(regulons)
        if save_loom:
            self.to_loom(df, auc_matrix, regulons, fn_prefix=fn_prefix)
            # self.to_cytoscape(regulons, adjacencies, 'Zfp354c')

    @staticmethod
    def input_hotspot(counts, position):
        """
        Extract needed information to construct a Hotspot instance from StereoExpData data
        :param data:
        :return: a dictionary
        """
        # 3. use dataframe and position array, StereoExpData as well
        # counts = data.to_df().T  # gene x cell
        # position = data.position
        num_umi = counts.sum(axis=0)  # total counts for each cell
        # Filter genes
        gene_counts = (counts > 0).sum(axis=1)
        valid_genes = gene_counts >= 50
        counts = counts.loc[valid_genes]
        return {'counts': counts, 'num_umi': num_umi, 'position': position}

    def hotspot_matrix(self,
                       model='bernoulli',
                       distances: pd.DataFrame = None,
                       tree=None,
                       weighted_graph: bool = False,
                       n_neighbors=30,
                       fdr_threshold: float = 0.05,
                       tf_list: list = None,
                       jobs=None,
                       cache: bool = True,
                       cache_res_key: str = 'regulatory_network_inference',
                       ThreeD_slice: bool = False,
                       **kwargs) -> pd.DataFrame:
        """
        Inference of co-expression modules via hotspot method
        :param data: Count matrix (shape is cells by genes)
        :param model: Specifies the null model to use for gene expression.
            Valid choices are:
                * 'danb': Depth-Adjusted Negative Binomial
                * 'bernoulli': Models probability of detection
                * 'normal': Depth-Adjusted Normal
                * 'none': Assumes data has been pre-standardized
        :param distances: Distances encoding cell-cell similarities directly
            Shape is (cells x cells)
        :param tree: Root tree node.  Can be created using ete3.Tree
        :param weighted_graph: Whether or not to create a weighted graph
        :param n_neighbors: Neighborhood size
        :param neighborhood_factor: Used when creating a weighted graph.  Sets how quickly weights decay
            relative to the distances within the neighborhood.  The weight for
            a cell with a distance d will decay as exp(-d/D) where D is the distance
            to the `n_neighbors`/`neighborhood_factor`-th neighbor.
        :param approx_neighbors: Use approximate nearest neighbors or exact scikit-learn neighbors. Only
            when hotspot initialized with `latent`.
        :param fdr_threshold: Correlation theshold at which to stop assigning genes to modules
        :param tf_list: predefined TF names
        :param jobs: Number of parallel jobs to run
        :paran ThreeD_slice: whether to use 3D slice data.
        :return: A dataframe, local correlation Z-scores between genes (shape is genes x genes)
        """

        if cache and ('adjacencies' in self.pipeline_res[cache_res_key].keys()):
            logger.info(f'cached file {cache_res_key}["adjacencies"] found')
            adjacencies = self.pipeline_res[cache_res_key]['adjacencies']
            self.adjacencies = adjacencies
            return adjacencies
        else:
            logger.info('cached file not found, running hotspot now')

        global hs
        # data = self.stereo_exp_data
        if self.use_raw:
            counts = self.stereo_exp_data.raw.to_df().T
        else:
            counts = self.stereo_exp_data.to_df().T
        hotspot_data = RegulatoryNetworkInference.input_hotspot(counts, self.stereo_exp_data.position)

        if ThreeD_slice:
            arr2 = self.stereo_exp_data.position_z
            position_3D = np.concatenate((self.stereo_exp_data.position, arr2), axis=1)
            hotspot_data['position'] = position_3D

        hs = hotspot.Hotspot.legacy_init(hotspot_data['counts'],
                                         model=model,
                                         latent=hotspot_data['position'],
                                         umi_counts=hotspot_data['num_umi'],
                                         distances=distances,
                                         tree=tree)

        hs.create_knn_graph(weighted_graph=weighted_graph, n_neighbors=n_neighbors)

        # the most? time consuming step
        logger.info('compute_autocorrelations()')
        hs_results = hs.compute_autocorrelations(jobs=jobs)
        logger.info('compute_autocorrelations() done')

        hs_genes = hs_results.loc[hs_results.FDR < fdr_threshold].index  # Select genes
        logger.info('compute_local_correlations')
        # nope, THIS is the most time consuming step
        local_correlations = hs.compute_local_correlations(hs_genes, jobs=jobs)  # jobs for parallelization
        logger.info('Network Inference DONE')
        logger.info(f'Hotspot: create {local_correlations.shape[0]} features')
        logger.info(local_correlations.shape)

        # subset by TFs
        if tf_list and tf_list != 'all':
            common_tf_list = list(set(tf_list).intersection(set(local_correlations.columns)))
            logger.info(f'detected {len(common_tf_list)} predefined TF in data')
            assert len(common_tf_list) > 0, 'predefined TFs not found in data'

        # reshape matrix
        local_correlations['TF'] = local_correlations.columns
        local_correlations = local_correlations.melt(id_vars=['TF'])
        local_correlations.columns = ['TF', 'target', 'importance']
        # remove if TF = target
        local_correlations = local_correlations[local_correlations.TF != local_correlations.target]

        self.adjacencies = local_correlations
        return local_correlations

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
        local_cluster = LocalCluster(n_workers=num_workers, dashboard_address=None, threads_per_worker=4)
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
                      cache_res_key: str = 'regulatory_network_inference',
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
        :return:

        Example:

        """

        if cache and ('adjacencies' in self.pipeline_res[cache_res_key].keys()):
            logger.info(f'cached file {cache_res_key}["adjacencies"] found')
            adjacencies = self.pipeline_res[cache_res_key]['adjacencies']
            self.adjacencies = adjacencies
            return adjacencies
        else:
            logger.info('cached file not found, running grnboost2 now')

        if num_workers is None:
            num_workers = cpu_count()
        custom_client = RegulatoryNetworkInference._set_client(num_workers)
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
                      motif_anno: str,
                      num_workers: int = None,
                      cache: bool = True,
                      cache_res_key: str = 'regulatory_network_inference',
                      **kwargs):
        """
        First, calculate a list of enriched motifs and the corresponding target genes for all modules.
        Then, create regulon_list from this table of enriched motifs.
        :param modules: the sequence of modules.
        :param dbs: the sequence of databases.
        :param motif_anno: the name of the file that contains the motif annotations to use.
        :param num_workers: if not using a cluster, the number of workers to use for the calculation. None of all available CPUs need to be used. # noqa
        :param cache:
        :param save:
        :param fn:
        :return:
        """

        if cache and ('regulons_full' in self.pipeline_res[cache_res_key].keys()):
            logger.info(f'cached file {cache_res_key}["regulons_full"] found')
            regulon_list = self.pipeline_res[cache_res_key]['regulons_full']
            self.regulon_list = regulon_list
            return regulon_list
        else:
            logger.info('cached file not found, running prune modules now')

        if num_workers is None:
            num_workers = cpu_count()

        with ProgressBar():
            df = prune2df(dbs, modules, motif_anno, num_workers=num_workers, **kwargs)

        regulon_list = df2regulons(df)
        self.regulon_list = regulon_list

        # alternative way of getting regulon_list, without creating df first
        return regulon_list, df

    def auc_activity_level(self,
                           matrix,
                           regulons: list,
                           auc_threshold: float,
                           num_workers: int,
                           seed=None,
                           cache: bool = True,
                           cache_res_key: str = 'regulatory_network_inference',
                           **kwargs) -> pd.DataFrame:
        """

        :param the expression matrix (n_cells x n_genes):
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param regulons: list of ctxcore.genesig.Regulon objects. The gene signatures or regulons.
        :param auc_threshold: the fraction of the ranked genome to take into account for the calculation of the Area Under the recovery Curve. # noqa
        :param num_workers: the number of cores to use.
        :param cache:
        :param save:
        :param fn:
        :return:
        """

        if cache and ('auc_matrix' in self.pipeline_res[cache_res_key].keys()):
            logger.info(f'cached file {cache_res_key}["auc_matrix"] found')
            auc_mtx = self.pipeline_res[cache_res_key]['auc_matrix']
            self.auc_mtx = auc_mtx
            return auc_mtx
        else:
            logger.info('cached file not found, calculating auc_activity_level now')

        if num_workers is None:
            num_workers = cpu_count()

        auc_mtx = aucell(matrix, regulons, auc_threshold=auc_threshold, num_workers=num_workers, seed=seed, **kwargs)
        self.auc_mtx = auc_mtx

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

    def regulons_to_csv(self, regulon_list: list, fn: str = 'regulon_list.csv', fn_prefix: str = None):
        """
        Save regulon_list (df2regulons output) into a csv file.
        :param regulon_list:
        :param fn:
        :return:
        """
        regulon_dict = get_regulon_dict(regulon_list)
        # Optional: join list of target genes
        for key in regulon_dict.keys():
            regulon_dict[key] = ";".join(regulon_dict[key])
        # Write to csv file
        if fn_prefix is not None:
            fn = f"{fn_prefix}_{fn}"
        with open(fn, 'w') as f:
            w = csv.writer(f)
            w.writerow(["Regulons", "Target_genes"])
            w.writerows(regulon_dict.items())

    def to_loom(
            self,
            matrix: pd.DataFrame,
            auc_matrix: pd.DataFrame,
            regulons: list,
            loom_fn: str = 'grn_output.loom',
            fn_prefix: str = None
    ):
        """
        Save GRN results in one loom file
        :param matrix:
        :param auc_matrix:
        :param regulons:
        :param loom_fn:
        :return:
        """
        if fn_prefix is not None:
            loom_fn = f"{fn_prefix}_{loom_fn}"
        export2loom(
            ex_mtx=matrix,
            auc_mtx=auc_matrix,
            regulons=[r.rename(r.name.replace('(+)', ' (' + str(len(r)) + 'g)')) for r in regulons],
            out_fname=loom_fn
        )

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
    func = lambda x: (x - x.mean()) / x.std(ddof=0)  # noqa
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


def _name(fname: str) -> str:
    """
    Extract file name (without path and extension)
    :param fname:
    :return:
    """
    return os.path.splitext(os.path.basename(fname))[0]
