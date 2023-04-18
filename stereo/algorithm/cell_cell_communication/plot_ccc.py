import os
from typing import Union
from pathlib import Path
import natsort

# third part module
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# module in self project
from stereo.log_manager import logger
from stereo.plots.plot_base import PlotBase

from stereo.algorithm.cell_cell_communication.exceptions import PipelineResultInexistent

class PlotCellCellCommunication(PlotBase):
    # TODO: change default paths
    def ccc_dot_plot(
            self,
            interacting_pairs: Union[str, list, np.ndarray] = None,
            clusters1: Union[str, list, np.ndarray] = None,
            clusters2: Union[str, list, np.ndarray] = None,
            separator_cluster: str = '|',
            palette: str = 'Reds',
            res_key: str = 'cell_cell_communication',
            width: int = None,
            height: int = None
            # **kw_args
    ):
        """Generate dot plot based on the result of CellCellCommunication.

        :param interacting_pairs: path, string, list or ndarray.
                        specify the interacting pairs which would be shown on plot, defaults to None.
                        1) path: the path of file in which saves the interacting pairs which would be shown, one line one pair.
                        2) string: only one interacting pair.
                        3) list or ndarray: an array contains the interacting pairs which would be shown.
        :param clusters1: path, string, list or ndarray.
                        the first clusters in cluster pairs which would be shown on plot, defaults to None.
                        1) path: the path of file in which saves the clusters which would be shown, one line one cluster.
                        2) string: only one cluster.
                        3) list or ndarray: an array contains the clusters which would be shown.
        :param clusters2: path, string, list or ndarray.
                        the second clusters in cluster pairs which would be shown on plot, defaults to None.
                        clusters1 and clusters2 together form cluster pairs
                        each cluster in cluster1 will join with each one in cluster2 to form the cluster pairs.
                        if set it to None, it will be set to all clusters.
                        1) path: the path of file in which saves the clusters which would be shown, one line one cluster.
                        2) string: only one cluster.
                        3) list or ndarray: an array contains the clusters which would be shown.
        :param separator_cluster: the symbol for joining the clusters1 and clusters2, defaults to '|'
        :param palette: plot palette, defaults to 'Reds'
        :param res_key: the key which specifies the cell cell communication result in data.tl.result, defaults to 'cell_cell_communication'
        :return: matplotlib.figure
        """
        logger.info('Generating dot plot')

        if res_key not in self.pipeline_res:
            PipelineResultInexistent(res_key)

        if self.pipeline_res[res_key]['parameters']['analysis_type'] != 'statistical':
            logger.warning("This plot just only support analysis type 'statistical'")
            return None

        means_df = self.pipeline_res[res_key]['means']
        pvalues_df = self.pipeline_res[res_key]['pvalues']

        interacting_pairs = self._parse_interacting_pairs_or_clusters(interacting_pairs)
        if interacting_pairs is None:
            interacting_pairs = means_df['interacting_pair'].tolist()
        else:
            if all(np.isin(interacting_pairs, means_df['interacting_pair']) == False):
                raise Exception("there is no interacting pairs to show, maybe the parameter 'interacting_pairs' you set is not in result.")

        clusters1 = self._parse_interacting_pairs_or_clusters(clusters1)
        clusters2 = self._parse_interacting_pairs_or_clusters(clusters2)
        if clusters1 is None:
            cluster_pairs = natsort.natsorted([x for x in means_df.columns if separator_cluster in x])
        else:
            if clusters2 is None:
                cluster_res_key = self.pipeline_res[res_key]['parameters']['cluster_res_key']
                clusters2 = self.pipeline_res[cluster_res_key]['group'].unique()
            cluster_pairs = [f'{c1}{separator_cluster}{c2}' for c1 in clusters1 for c2 in clusters2]
            cluster_pairs = natsort.natsorted([x for x in cluster_pairs if x in means_df.columns])
            if len(cluster_pairs) == 0:
                raise Exception("there is no cluster pairs to show, maybe the parameter 'clusters' you set is not in result.")

        means_selected: pd.DataFrame = means_df[means_df['interacting_pair'].isin(interacting_pairs)][['interacting_pair'] + cluster_pairs]
        pvalues_selected: pd.DataFrame = pvalues_df[pvalues_df['interacting_pair'].isin(interacting_pairs)][['interacting_pair'] + cluster_pairs]

        nrows, ncols = means_selected.shape

        means = means_selected.melt(id_vars='interacting_pair', value_vars=cluster_pairs, value_name='mean')
        means['log2(mean+1)'] = np.log2(means['mean'] + 1)

        pvalues = pvalues_selected.melt(id_vars='interacting_pair', value_vars=cluster_pairs, value_name='pvalue')
        pvalues['-log10(pvalue)'] = pvalues['pvalue'].apply(lambda x: -np.log10(0.000001) if x == 0 else (-np.log10(x) if x != 1 else 0))
        # pvalues['log10(pvalue+1)'] = np.log10(pvalues['pvalue'] + 1)

        result = pd.merge(means, pvalues, on=["interacting_pair", "variable"])
        result = result.rename(columns={'variable': 'cluster_pair'})

        # plotting
        if width is None or height is None:
            width, height = int(5 + max(3, ncols * 0.8)), int(3 + max(5, nrows * 0.5))
        else:
            width = width / 100 if width >= 100 else int(5 + max(3, ncols * 0.8))
            height = height / 100 if height >= 100 else int(3 + max(5, nrows * 0.5))
        fig, ax = plt.subplots(figsize=(width, height))
        # fig.subplots_adjust(bottom=0.2, left=0.18, right=0.85)
        sns.scatterplot(data=result, x="cluster_pair", y="interacting_pair", palette=palette, 
                        hue='log2(mean+1)', size='-log10(pvalue)', sizes=(100, 300), legend='auto', ax=ax)
        # legend_position = kw_args.get('legend_position', 'lower right')
        # legend_coodinate = kw_args.get('legend_coodinate')
        # ax.legend(fontsize=12, frameon=False, ncol=1, loc=legend_position, bbox_to_anchor=legend_coodinate)
        ax.legend(fontsize=12, frameon=False, ncol=1, loc=(1.02, 0))
        ax.tick_params(axis='x', labelsize=12, labelrotation=90)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')
        return fig

    def ccc_heatmap(
            self,
            pvalue: float = 0.05,
            separator_cluster: str = '|',
            res_key: str = 'cell_cell_communication',
            width: int = None,
            height: int = None
    ):
        """
        Heatmap of number of interactions in each cluster pairs.
        Each off-diagonal cell value equals the number of interactions from A to B + the number of interactions from B to A

        :param pvalue: the threshold of pvalue, defaults to 0.05
        :param separator_cluster: the symbol for joining the first and second cluster in cluster pairs, defaults to '|'
        :param res_key: the key which specifies the cell cell communication result in data.tl.result, defaults to 'cell_cell_communication'
        :return: _description_
        """
        logger.info('Generating heatmap plot')

        if res_key not in self.pipeline_res:
            PipelineResultInexistent(res_key)

        if self.pipeline_res[res_key]['parameters']['analysis_type'] != 'statistical':
            logger.warning("This plot just only support analysis type 'statistical'")
            return None

        cluster_res_key = self.pipeline_res[res_key]['parameters']['cluster_res_key']
        meta_df = self.pipeline_res[cluster_res_key]
        clusters_all = natsort.natsorted(meta_df['group'].unique())
        n_cluster: int = len(clusters_all)

        pvalues_df = self.pipeline_res[res_key]['pvalues']

        cluster_pairs = np.array(np.meshgrid(clusters_all, clusters_all)).T.reshape(-1, 2)
        network = pd.DataFrame(cluster_pairs, columns=['source', 'target'])

        for index, row in network.iterrows():
            col1 = row['source'] + separator_cluster + row['target']
            col2 = row['target'] + separator_cluster + row['source']
            if col1 in pvalues_df.columns and col2 in pvalues_df.columns:
                if col1 == col2:
                    network.loc[index, 'number'] = (pvalues_df[col1] <= pvalue).sum()
                else:
                    network.loc[index, 'number'] = (pvalues_df[col1] <= pvalue).sum() + (pvalues_df[col2] <= pvalue).sum()
            else:
                network.loc[index, 'number'] = 0

        network: pd.DataFrame = network.pivot("source", "target", "number")
        network = network.loc[clusters_all][clusters_all]
        rows = network.index.tolist()
        rows.reverse()
        network = network[rows]
        log_network = np.log1p(network)

        if width is None or height is None:
            width, height = int(3 + max(3, n_cluster * 0.5)) * 2, int(3 + max(3, n_cluster * 0.5))
        else:
            width = width / 100 if width >= 100 else int(3 + max(3, n_cluster * 0.5)) * 2
            height = height / 100 if height >= 100 else int(3 + max(3, n_cluster * 0.5))
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(width, height), gridspec_kw={'wspace': 0})

        sns.heatmap(data=network, square=True, cmap='coolwarm', cbar_kws={'pad': 0.1, 'shrink': 0.5, 'location': 'bottom', 'orientation': 'horizontal'}, ax=axes[0])
        axes[0].yaxis.set_ticks_position('right')
        axes[0].invert_yaxis()
        axes[0].tick_params(axis='x', labelsize=13, labelrotation=90)
        axes[0].tick_params(axis='y', labelsize=13)
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
        axes[0].set_title('count')

        sns.heatmap(data=log_network, square=True, cmap='coolwarm', cbar_kws={'pad': 0.1, 'shrink': 0.5, 'location': 'bottom', 'orientation': 'horizontal'}, ax=axes[1])
        axes[1].yaxis.set_ticks_position('right')
        axes[1].invert_yaxis()
        axes[1].tick_params(axis='x', labelsize=13, labelrotation=90)
        axes[1].tick_params(axis='y', labelsize=13)
        axes[1].set_xlabel('')
        axes[1].set_ylabel('')
        axes[1].set_title('log_count')

        return fig

    def _parse_interacting_pairs_or_clusters(self, interacting_pairs_or_clusters: Union[str, list, np.ndarray]):
        if isinstance(interacting_pairs_or_clusters, list) or isinstance(interacting_pairs_or_clusters, np.ndarray):
            return np.unique(interacting_pairs_or_clusters)
        
        if isinstance(interacting_pairs_or_clusters, str):
            path = Path(interacting_pairs_or_clusters)
            if not path.is_file() and not path.is_dir():
                return [interacting_pairs_or_clusters]
            if path.is_file() and path.exists():
                with path.open('r') as fp:
                    return np.unique([line.strip() for line in fp.readlines()])
        
        return None

    
    def _ensure_path_exists(self, path: str):
        expanded_path = os.path.expanduser(path)

        if not os.path.exists(expanded_path):
            os.makedirs(expanded_path)