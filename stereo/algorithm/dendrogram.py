from typing import Sequence, Dict, Any, Optional, Union, Literal
import pandas as pd
import numpy as np
from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.log_manager import logger

class Dendrogram(AlgorithmBase):
    def main(
        self,
        cluster_res_key: str,
        pca_res_key: Optional[str] = 'pca',
        use_raw: Optional[bool] = False,
        cor_method: str = 'pearson',
        linkage_method: str = 'complete',
        optimal_ordering: bool = False,
        res_key: str = 'dendrogram',
    ):
        """
        Computes a hierarchical clustering for the given `cluster_res_key` categories.

        Alternatively, a list of `var_names` (e.g. genes) can be given.

        Average values of either `var_names` or components are used
        to compute a correlation matrix.

        .. note::
            The computation of the hierarchical clustering is based on predefined
            groups and not per cell. The correlation matrix is computed using by
            default pearson but other methods are available.

        :param cluster_res_key: a key or a list of keys which specify the cluster result in data.tl.result.
        :param pca_res_key: a key which specify the pca result in data.tl.result, if None, using exp_matrix instead.
        :param use_raw: whether to use raw exp_matrix, defaults to False.
        :param cor_method: correlation method to use, options are 'pearson', 'kendall', and 'spearman'.
        :param linkage_method: linkage method to use. See :func:`scipy.cluster.hierarchy.linkage` for more information.
        :param optimal_ordering: Same as the optimal_ordering argument of :func:`scipy.cluster.hierarchy.linkage`
                            which reorders the linkage matrix so that the distance between successive
                            leaves is minimal.
        :param res_key: a key to store dendrogram result in data.tl.result.
        """
        if isinstance(cluster_res_key, str):
            # if not a list, turn into a list
            cluster_res_key = [cluster_res_key]
        cluster_res_list = []
        for key in cluster_res_key:
            if key not in self.pipeline_res:
                raise ValueError(f"there is no cluster result specified by key '{key}' in data.tl.result")
            if key in self.stereo_exp_data.cells._obs:
                cluster_res_list.append(self.stereo_exp_data.cells._obs[key].astype('category'))
            else:
                cluster_res_list.append(self.pipeline_res[key]['group'].astype('category'))

        # if gene_names is None:
        #     rep_df: pd.DataFrame = self._choose_representation(pca_res_key, use_raw)
        # else:
        #     if gene_names == 'all':
        #         gene_names = self.stereo_exp_data.genes.gene_name
        #     elif isinstance(gene_names, str):
        #         gene_names = [gene_names]
        #     rep_df: pd.DataFrame = self._prepare_dataframe(gene_names, use_raw)
        rep_df: pd.DataFrame = self._choose_representation(pca_res_key, use_raw)
        categorical:pd.Series = cluster_res_list[0]
        if len(cluster_res_list) > 1:
            for group in cluster_res_list[1:]:
                # create new category by merging the given groupby categories
                categorical = (
                    categorical.astype(str) + "_" + group.astype(str)
                ).astype('category')
        categorical.name = "_".join(cluster_res_key)

        rep_df.set_index(categorical, inplace=True)
        categories = rep_df.index.categories

        # aggregate values within categories using 'mean'
        mean_df = rep_df.groupby(level=0).mean()

        import scipy.cluster.hierarchy as sch
        from scipy.spatial import distance

        corr_matrix = mean_df.T.corr(method=cor_method)
        corr_condensed = distance.squareform(1 - corr_matrix)
        z_var = sch.linkage(
            corr_condensed, method=linkage_method, optimal_ordering=optimal_ordering
        )
        dendro_info = sch.dendrogram(z_var, labels=list(categories), no_plot=True)

        dat = dict(
            linkage=z_var,
            cluster_res_key=cluster_res_key,
            pca_res_key=pca_res_key,
            cor_method=cor_method,
            linkage_method=linkage_method,
            categories_ordered=dendro_info['ivl'],
            categories_idx_ordered=dendro_info['leaves'],
            dendrogram_info=dendro_info,
            correlation_matrix=corr_matrix.values,
        )

        self.pipeline_res[res_key] = dat
    
    def _choose_representation(self, pca_res_key, use_raw):
        if pca_res_key is not None:
            if pca_res_key not in self.pipeline_res:
                raise ValueError(f"Can not get PCA result from data.tl.result by key '{pca_res_key}'.")
            return self.pipeline_res[pca_res_key]
        else:
            exp_matrix = self.stereo_exp_data.exp_matrix if not use_raw else self.stereo_exp_data.raw.exp_matrix
            return pd.DataFrame(exp_matrix.toarray())
    
    def _prepare_dataframe(self, gene_names, use_raw):
        all_gene_names = pd.Index(self.stereo_exp_data.genes.gene_name)
        gene_idx = all_gene_names.get_indexer(gene_names)
        if use_raw and self.stereo_exp_data.raw is None:
            logger.warning(f"there is no raw data, using current data instead.")
            use_raw = False
        sub_exp_matrix = self.stereo_exp_data.exp_matrix[:, gene_idx] if not use_raw else self.stereo_exp_data.raw.exp_matrix[:, gene_idx]
        sub_exp_matrix_df = pd.DataFrame(sub_exp_matrix.toarray(), columns=gene_names)
        return sub_exp_matrix_df
    