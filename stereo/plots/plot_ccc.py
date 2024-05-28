import json
import os
from pathlib import Path
from typing import (
    Union,
    List,
    Dict,
    Optional
)

import matplotlib.pyplot as plt
import natsort
import networkx as nx
# third part module
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from pycirclize import Circos

from stereo.algorithm.cell_cell_communication.analysis_helper import mouse2human
from stereo.algorithm.cell_cell_communication.exceptions import PipelineResultInexistent
# module in self project
from stereo.log_manager import logger
from stereo.plots.plot_base import PlotBase
from stereo.stereo_config import stereo_conf


class PlotCellCellCommunication(PlotBase):
    def __init__(self, *args, **kwargs):
        super(PlotCellCellCommunication, self).__init__(*args, **kwargs)
        PlotCellCellCommunication.ccc_sankey_plot.__download__ = False

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
            height: int = None,
            **kw_args
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
        :param width: the figure width in pixels, defaults to None
        :param height: the figure height in pixels, defaults to None
        :return: matplotlib.figure
        """  # noqa
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
            if all(np.isin(interacting_pairs, means_df['interacting_pair']) == False):  # noqa
                raise Exception("there is no interacting pairs to show, maybe the parameter 'interacting_pairs' "
                                "you set is not in result.")

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
                raise Exception(
                    "there is no cluster pairs to show, maybe the parameter 'clusters' you set is not in result.")

        means_selected: pd.DataFrame = means_df[means_df['interacting_pair'].isin(interacting_pairs)][
            ['interacting_pair'] + cluster_pairs]
        pvalues_selected: pd.DataFrame = pvalues_df[pvalues_df['interacting_pair'].isin(interacting_pairs)][
            ['interacting_pair'] + cluster_pairs]

        nrows, ncols = means_selected.shape

        means = means_selected.melt(id_vars='interacting_pair', value_vars=cluster_pairs, value_name='mean')
        means['log2(mean+1)'] = np.log2(means['mean'] + 1)

        pvalues = pvalues_selected.melt(id_vars='interacting_pair', value_vars=cluster_pairs, value_name='pvalue')
        pvalues['-log10(pvalue)'] = pvalues['pvalue'].apply(
            lambda x: -np.log10(0.000001) if x == 0 else (-np.log10(x) if x != 1 else 0))

        result = pd.merge(means, pvalues, on=["interacting_pair", "variable"])
        result = result.rename(columns={'variable': 'cluster_pair'})

        # plotting
        if width is None or height is None:
            width, height = int(5 + max(3, ncols * 0.8)), int(3 + max(5, nrows * 0.5))
        else:
            width = width / 100 if width >= 100 else int(5 + max(3, ncols * 0.8))
            height = height / 100 if height >= 100 else int(3 + max(5, nrows * 0.5))
        fig, ax = plt.subplots(figsize=(width, height))
        sns.scatterplot(data=result, x="cluster_pair", y="interacting_pair", palette=palette,
                        hue='log2(mean+1)', size='-log10(pvalue)', sizes=(100, 300), legend='auto', ax=ax, **kw_args)
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
            height: int = None,
            palette: str = 'coolwarm',
            **kwargs
    ):
        """
        Heatmap of number of interactions in each cluster pairs.
        Each off-diagonal cell value equals the number of interactions from A to B + the number of interactions from B to A

        :param pvalue: the threshold of pvalue, defaults to 0.05
        :param separator_cluster: the symbol for joining the first and second cluster in cluster pairs, defaults to '|'
        :param res_key: the key which specifies the cell cell communication result in data.tl.result, defaults to 'cell_cell_communication'
        :return: _description_
        """  # noqa
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
                    network.loc[index, 'number'] = (pvalues_df[col1] <= pvalue).sum() + (
                            pvalues_df[col2] <= pvalue).sum()
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

        sns.heatmap(
            data=network,
            square=True,
            cmap=palette,
            cbar_kws={'pad': 0.1, 'shrink': 0.5, 'location': 'top', 'orientation': 'horizontal'},
            ax=axes[0], **kwargs)
        axes[0].yaxis.set_ticks_position('left')
        axes[0].invert_yaxis()
        axes[0].tick_params(axis='x', labelsize=13, labelrotation=90)
        axes[0].tick_params(axis='y', labelsize=13, labelrotation=0)
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
        axes[0].set_title('count')

        sns.heatmap(
            data=log_network,
            square=True,
            cmap=palette,
            cbar_kws={'pad': 0.1, 'shrink': 0.5, 'location': 'top', 'orientation': 'horizontal'},
            ax=axes[1], **kwargs)
        axes[1].yaxis.set_ticks_position('right')
        axes[1].invert_yaxis()
        axes[1].tick_params(axis='x', labelsize=13, labelrotation=90)
        axes[1].tick_params(axis='y', labelsize=13, labelrotation=0)
        axes[1].set_xlabel('')
        axes[1].set_ylabel('')
        axes[1].set_title('log_count')

        return fig

    def ccc_circos_plot(
            self,
            separator_cluster: str = '|',
            cluster_pair: list = None,
            palette: str = "RdYlBu_r",
            width: int = 8,
            height: int = 8,
            res_key: str = 'cell_cell_communication'
    ):
        # TODO: change default significant_path to the significant_means_xxx file generated by the main CCC analysis
        """
        Circos plot of number of interactions in each cluster pairs.

        :param separator_cluster: separator used for cell cluster pairs.
        :param cluster_pair: if None, use all cluster pairs in the significant result; else list selected cluster pairs used in the plot. # noqa
        :param palette: colormap used.
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.
        :param res_key: the key which specifies the cell cell communication result in data.tl.result, defaults to 'cell_cell_communication'. # noqa
        """
        logger.info('Generating circos plot')

        if res_key not in self.pipeline_res:
            PipelineResultInexistent(res_key)

        significant_df: pd.DataFrame = self.pipeline_res[res_key]['significant_means']

        if cluster_pair is None:
            significant_df = significant_df.drop(
                ['id_cp_interaction', 'interacting_pair', 'partner_a', 'partner_b', 'gene_a', 'gene_b', 'secreted',
                 'receptor_a', 'receptor_b', 'annotation_strategy', 'is_integrin', 'rank'], axis=1)
            significant_df = significant_df.dropna(how='all').reset_index(drop=True)
            cluster_pair = significant_df.columns
        else:
            significant_df = significant_df[cluster_pair]
            significant_df = significant_df.dropna(how='all').reset_index(drop=True)

        cell_types = [x.split(separator_cluster) for x in cluster_pair]
        cell_types = list(set([item for sublist in cell_types for item in sublist]))
        cell_types.sort()

        number_matrix = pd.DataFrame(index=cell_types, columns=cell_types)
        for source in cell_types:
            for target in cell_types:
                if source + separator_cluster + target in cluster_pair:
                    number_matrix.loc[source, target] = significant_df[source + separator_cluster + target].count()
        number_matrix = number_matrix.fillna(0)

        circos = Circos.initialize_from_matrix(
            number_matrix,
            space=5,
            cmap=palette,
            label_kws=dict(size=12),
            link_kws=dict(ec="black", lw=0.5, direction=1),
        )
        fig, ax = plt.subplots(figsize=(width, height), subplot_kw={"projection": "polar"})
        circos.plotfig(ax=ax)
        return fig

    def ccc_sankey_plot(
            self,
            sender_cluster: str,
            receiver_cluster: str,
            homo_transfer: bool = False,
            homogene_path: Optional[str] = None,
            separator_cluster: str = '|',
            separator_interaction: str = '_',
            weighted_network_path: str = None,
            regulons: Union[str, Dict[str, list]] = None,
            tfs: Optional[List[str]] = None,
            pct_expressed: float = 0.05,
            max_path_length: int = 4,
            width: int = 600,
            height: int = 880,
            out_path: Optional[str] = None,
            receptor_tf_paths_out_path: Optional[str] = None,
            res_key: str = 'cell_cell_communication',
    ):
        """
        Sankey-plot showing inter- and/or intra-cellular gene interaction. Left pillar is ligands, middle pillar receptors, # noqa
        right pillar TFs. The width of each band is the average expression of the two genes.

        :param sender_cluster: sender cell type
        :param receiver_cluster: receiver cell type
        :param homo_transfer: If species is 'MOUSE' but database is 'cellphonedb' or 'liana', the gene names in the
                                significant result have been transferred to 'HUMAN', we need to transfer them back
                                in order to match the gene names in counts.
        :param homogene_path: path to the file storing mouse-human homologous genes ralations.
        :param separator_cluster: separator used for cell cluster pairs
        :param separator_interaction: separator used for LR interaction
        :param weighted_network_path: path to the weighted network
        :param regulons: path or dict to the spaGRN regulon output
        :param tfs: list of TFs used in the Sankey plot, if it is gave, the parameter `regulons` is ignored.
        :param pct_expressed: threshold used to detect expressed path between receptor and TF.
        :param max_path_length: the max path length between receptor and TF,
                                paths longer than max_path_length is not considered as a potential pathway
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.
        :param out_path: the path to save the plot.
        :param res_key: the key which specifies the cell cell communication result in data.tl.result, defaults to 'cell_cell_communication'. # noqa

        """

        if res_key not in self.pipeline_res:
            PipelineResultInexistent(res_key)

        cluster_res_key = self.pipeline_res[res_key]['parameters']['cluster_res_key']
        if cluster_res_key not in self.pipeline_res:
            PipelineResultInexistent(cluster_res_key)

        assert weighted_network_path is not None

        assert regulons is not None or tfs is not None

        cluster_pair_key = sender_cluster + separator_cluster + receiver_cluster
        significant_df: pd.DataFrame = self.pipeline_res[res_key]['significant_means']
        significant_df = significant_df[['interacting_pair', 'partner_a', 'partner_b', cluster_pair_key]]

        def __clean_significant_df(row, cluster_pair_key):
            partner_a, partner_b = row['partner_a'], row['partner_b']
            if row[cluster_pair_key] == -1:
                row[cluster_pair_key] = pd.NaT
            if partner_a.startswith('complex') or partner_b.startswith('complex'):
                return pd.NaT
            else:
                return row

        significant_df = significant_df.apply(__clean_significant_df, axis=1, result_type='broadcast',
                                              cluster_pair_key=cluster_pair_key)
        significant_df.dropna(
            how='any',
            subset=['partner_a', 'partner_b', cluster_pair_key],
            axis=0,
            inplace=True
        )

        assert significant_df.size > 0, 'No significant LR pairs detected.'

        significant_df.reset_index(drop=True, inplace=True)

        ligands = list(set([lr.split('_')[0] for lr in significant_df['interacting_pair'].values]))
        ligands.sort()
        receptors = list(set([lr.split('_')[1] for lr in significant_df['interacting_pair'].values]))
        receptors.sort()
        significant_pairs = significant_df['interacting_pair'].values

        if homo_transfer:
            if homogene_path is None:
                homogene_path = Path(stereo_conf.data_dir,
                                     'algorithm/cell_cell_communication/database/mouse2human.csv').absolute().as_posix()
            genes_mouse = self.stereo_exp_data.gene_names
            genes_human, human_genes_to_mouse = mouse2human(genes_mouse, homogene_path)
            ligands = [human_genes_to_mouse[x] for x in ligands]
            receptors = [human_genes_to_mouse[x] for x in receptors]
            significant_pairs_tmp = []
            for lr in significant_pairs:
                interaction1, interaction2 = lr.split(separator_interaction)
                if interaction1 not in human_genes_to_mouse or interaction2 not in human_genes_to_mouse:
                    continue
                significant_pairs_tmp.append(
                    human_genes_to_mouse[interaction1] + separator_interaction + human_genes_to_mouse[interaction2])
            significant_pairs = significant_pairs_tmp

        # Construct expressed weighted gene regulatory network
        counts_receiver = self._get_cell_counts(cluster_res_key, receiver_cluster)
        expressed_genes_receiver = self._get_expressed_genes(counts_receiver, pct_expressed)

        weighted_network_lr_sig = pd.read_csv(weighted_network_path, sep='\t')
        weighted_network_lr_sig_expressed = self._get_expressed_network(weighted_network_lr_sig,
                                                                        expressed_genes_receiver)
        weighted_network_lr_sig_expressed['distance'] = 1 / weighted_network_lr_sig_expressed['weight']

        # return genes_mouse_obtained, counts_receiver, expressed_genes_receiver, weighted_network_lr_sig_expressed

        G = nx.DiGraph()
        for idx, row in weighted_network_lr_sig_expressed.iterrows():
            G.add_edge(row['from'], row['to'], weight=row['weight'], distance=row['distance'])

        # Get TFs from the json file of GRN analysis
        if tfs is None:
            if isinstance(regulons, dict):
                tfs = list(map(lambda x: x[:-3], regulons.keys()))
            elif regulons.endswith('json'):
                with open(regulons, 'r') as fp:
                    rgl = json.load(fp)
                tfs = list(map(lambda x: x[:-3], rgl.keys()))
            elif regulons.endswith('csv'):
                regulons_df = pd.read_csv(regulons, sep=',', header=0)
                tfs = regulons_df['Regulons'].apply(lambda x: x[:-3]).tolist()

        # Get paths between Receptors and TFs
        source_rtf = []
        target_rtf = []
        length_rtf = []
        paths = []
        for receptor in receptors:
            for tf in tfs:
                try:
                    path, weight = self._get_shortest_path(G, source=receptor, target=tf, distance='distance',
                                                           weight='weight')
                    paths.append(path)
                    source_rtf.append(receptor)
                    target_rtf.append(tf)
                    length_rtf.append(len(path))
                except Exception:
                    source_rtf.append(receptor)
                    target_rtf.append(tf)
                    paths.append([])
                    length_rtf.append(999)

        result_path = pd.DataFrame({'receptor': source_rtf, 'TF': target_rtf, 'path': paths, 'path_length': length_rtf})
        if receptor_tf_paths_out_path:
            result_path.to_csv(receptor_tf_paths_out_path)
        result_path = result_path[result_path['path_length'] <= max_path_length]

        tfs = list(set(result_path['TF']))
        tfs.sort()

        assert len(tfs) > 0, 'Can not find any path from Receptor to TF.'

        # Generate final data for plotting
        label = ligands + receptors + tfs
        counts_sender = self._get_cell_counts(cluster_res_key, sender_cluster)

        # The left part of Ligand-Receptor interaction
        source_lr = []
        target_lr = []
        value_lr = []
        for i, ligand in enumerate(ligands):
            for j, receptor in enumerate(receptors):
                current_pair = ligand + separator_interaction + receptor
                if current_pair in significant_pairs:
                    source_lr.append(i)
                    target_lr.append(len(ligands) + j)
                    mean_expression = (self._calculate_mean_expression(counts_sender, ligand) +
                                       self._calculate_mean_expression(counts_receiver, receptor)) / 2
                    value_lr.append(mean_expression)

        # The right part of Receptor-TF interaction
        source_rtf = []
        target_rtf = []
        value_rtf = []
        for i, receptor in enumerate(receptors):
            # weight_r = []
            for j, tf in enumerate(tfs):
                df = result_path[(result_path['receptor'] == receptor) & (result_path['TF'] == tf)]
                if df.empty:
                    continue
                else:
                    source_rtf.append(len(ligands) + i)
                    target_rtf.append(len(ligands) + len(receptors) + j)
                    mean_expression = (self._calculate_mean_expression(counts_receiver, receptor) +
                                       self._calculate_mean_expression(counts_receiver, tf)) / 2
                    value_rtf.append(mean_expression)

        # Generate sankey plot
        node = dict(thickness=8, pad=2, label=label)
        link = dict(source=source_lr + source_rtf, target=target_lr + target_rtf, value=value_lr + value_rtf)

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=node,
                    link=link)
            ])
        fig.update_layout(height=height, width=width, font_size=12, font_family='Arial')
        if out_path is not None:
            fig.write_image(out_path)
        return fig

    def _get_cell_counts(self, cluster_res_key, cluster):
        cluster_res: pd.DataFrame = self.pipeline_res[cluster_res_key]
        isin = cluster_res['group'].isin([cluster]).to_numpy()
        cell_counts = self.stereo_exp_data.exp_matrix[isin]
        cell_list = self.stereo_exp_data.cell_names[isin]
        if self.stereo_exp_data.issparse():
            cell_counts = cell_counts.toarray()
        cell_counts = cell_counts.T
        return pd.DataFrame(cell_counts, columns=cell_list, index=self.stereo_exp_data.gene_names)

    def _get_expressed_genes(self, count_df, pct):
        count_df = count_df.loc[(count_df != 0).mean(axis=1) >= pct, :]
        return list(count_df.index)

    def _get_expressed_network(self, network, expressed_genes):
        sub_network = network[network['from'].isin(expressed_genes) & network['to'].isin(expressed_genes)].copy()
        return sub_network

    def _get_shortest_path(self, graph, source, target, distance, weight):
        shortest_path = nx.shortest_path(graph, source=source, target=target, weight=distance)
        path_weight_sum = 0
        for i in range(len(shortest_path) - 1):
            path_weight_sum = path_weight_sum + graph.get_edge_data(shortest_path[i], shortest_path[i + 1])[weight]
        return shortest_path, path_weight_sum

    def _calculate_mean_expression(self, counts, gene):
        counts_gene = counts.loc[gene, :]
        return np.mean(counts_gene)

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
