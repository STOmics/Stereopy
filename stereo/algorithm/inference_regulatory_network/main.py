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
from typing import Union

# third party modules
import json
import glob
import pandas as pd
from arboreto.utils import load_tf_names
from multiprocessing import cpu_count
from pyscenic.export import export2loom
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.prune import prune2df, df2regulons
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell

# modules in self project
from stereo.log_manager import logger
from stereo.algorithm.algorithm_base import AlgorithmBase


class InferenceRegulatoryNetwork(AlgorithmBase):
    """
    Algorithms to inference Gene Regulatory Networks (GRN)
    """
    
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
             cache_res_key: str = 'inference_regulatory_network',
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
        adjacencies = self.grn_inference(matrix, genes=target_genes, tf_names=tfs, num_workers=num_workers, seed=seed, cache=cache, cache_res_key=cache_res_key)
        modules = self.get_modules(adjacencies, df)
        # 4. Regulons prediction aka cisTarget
        regulons = self.prune_modules(modules, dbs, motif_anno_fn, num_workers, cache=cache, cache_res_key=cache_res_key)
        self.regulon_dict = get_regulon_dict(regulons)
        # 5: Cellular enrichment (aka AUCell)
        auc_matrix = self.auc_activity_level(df, regulons, auc_threshold, num_workers, seed=seed, cache=cache, cache_res_key=cache_res_key)

        # save results
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
                      cache_res_key: str = 'inference_regulatory_network',
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
        #TODO remove cached

        if cache and ('adjacencies' in self.pipeline_res[cache_res_key].keys()):
            logger.info(f'cached file {cache_res_key}["adjacencies"] found')
            adjacencies = self.pipeline_res[cache_res_key]['adjacencies']
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
                      cache_res_key: str = 'inference_regulatory_network',
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
            df = prune2df(dbs, modules, motif_anno_fn, num_workers=num_workers, **kwargs)
            
        regulon_list = df2regulons(df)
        self.regulon_list = regulon_list


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
                           cache_res_key: str = 'inference_regulatory_network',
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

    def to_loom(self, matrix: pd.DataFrame, auc_matrix: pd.DataFrame, regulons: list, loom_fn: str = 'grn_output.loom'):
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

def _name(fname: str) -> str:
    """
    Extract file name (without path and extension)
    :param fname:
    :return:
    """
    return os.path.splitext(os.path.basename(fname))[0]