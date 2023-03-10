# python core module
import os
from typing import Tuple, Union
from pathlib import Path
import natsort

# third part module
import numpy as np
import pandas as pd
import seaborn as sns
import numpy_groupies as npg
from functools import partial
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from multiprocessing.pool import Pool

# module in self project
from stereo.log_manager import logger
from stereo.stereo_config import stereo_conf
from stereo.plots.plot_base import PlotBase
from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.algorithm.cell_cell_communication.utils.sqlalchemy_model import Base
from stereo.algorithm.cell_cell_communication.analysis_helper import (
    Subsampler,
    write_to_file,
    get_separator,
    mouse2human
)
from stereo.algorithm.cell_cell_communication.utils.database_utils import Database, DatabaseManager
from stereo.algorithm.cell_cell_communication.utils.sqlalchemy_repository import (
    ComplexRepository,
    GeneRepository,
    InteractionRepository,
    MultidataRepository,
    ProteinRepository
)
from stereo.algorithm.cell_cell_communication.exceptions import (
    ProcessMetaException,
    ParseCountsException,
    ThresholdValueException,
    AllCountsFilteredException,
    NoInteractionsFound,
    InvalidDatabase,
    PipelineResultInexistent
)

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
                        each cluster in cluster1 will join with each one in cluster2 to form ther cluster pairs.
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
            logger.warn("This plot just only support analysis type 'statistical'")
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
        # pvalues['log10_pvalue'] = pvalues['pvalue'].apply(lambda x: -np.log10(0.0009) if x == 0 else (-np.log10(x) if x != 1 else 0))
        pvalues['log10(pvalue+1)'] = np.log10(pvalues['pvalue'] + 1)

        result = pd.merge(means, pvalues, on=["interacting_pair", "variable"])
        result = result.rename(columns={'variable': 'cluster_pair'})

        # plotting
        width, height = int(5 + max(3, ncols * 0.8)), int(3 + max(5, nrows * 0.5))
        fig, ax = plt.subplots(figsize=(width, height))
        # fig.subplots_adjust(bottom=0.2, left=0.18, right=0.85)
        sns.scatterplot(data=result, x="cluster_pair", y="interacting_pair", palette=palette, 
                        hue='log2(mean+1)', size='log10(pvalue+1)', sizes=(100, 300), legend='auto', ax=ax)
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
            res_key: str = 'cell_cell_communication'
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
            logger.warn("This plot just only support analysis type 'statistical'")
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

        width, height = int(3 + max(3, n_cluster * 0.5)) * 2, int(3 + max(3, n_cluster * 0.5))
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(width, height), gridspec_kw={'wspace': 0})

        sns.heatmap(data=network, square=True, cmap='coolwarm', cbar_kws={'pad': 0.1, 'shrink': 0.5, 'location': 'bottom', 'orientation': 'horizontal'}, ax=axes[0])
        axes[0].yaxis.set_ticks_position('right')
        axes[0].invert_yaxis()
        axes[0].tick_params(axis='x', labelsize=13, labelrotation=90)
        axes[0].tick_params(axis='y', labelsize=13)
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
        axes[0].set_title('network')

        sns.heatmap(data=log_network, square=True, cmap='coolwarm', cbar_kws={'pad': 0.1, 'shrink': 0.5, 'location': 'bottom', 'orientation': 'horizontal'}, ax=axes[1])
        axes[1].yaxis.set_ticks_position('right')
        axes[1].invert_yaxis()
        axes[1].tick_params(axis='x', labelsize=13, labelrotation=90)
        axes[1].tick_params(axis='y', labelsize=13)
        axes[1].set_xlabel('')
        axes[1].set_ylabel('')
        axes[1].set_title('log1p_network')

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


class CellCellCommunication(AlgorithmBase):
    # FIXME: change the default output_path in linux, change default homogene_path
    def main(
            self,
            analysis_type: str = 'statistical',
            cluster_res_key: str = 'cluster',
            micro_envs: pd.DataFrame = None,
            species: str = "HUMAN",
            database: str = 'cellphonedb',
            homogene_path: str = None,
            counts_identifiers: str = "hgnc_symbol",
            subsampling: bool = False,
            subsampling_log: bool = False,
            subsampling_num_pc: int = 1000,
            subsampling_num_cells: int = None,
            pca_res_key: str = None,
            separator_cluster: str = "|",
            separator_interaction: str = "_",
            iterations: int = 500,
            threshold: float = 0.1,
            processes: int = 1,
            pvalue: float = 0.05,
            result_precision: int = 3,
            output_path: str = None,
            means_filename: str = 'means',
            pvalues_filename: str = 'pvalues',
            significant_means_filename: str = 'significant_means',
            deconvoluted_filename: str = 'deconvoluted',
            output_format: str = 'csv',
            res_key: str = 'cell_cell_communication'
    ):
        """
        Cell-cell communication analysis main functon.

        :param analysis_type: type of analysis: "simple", "statistical".
        :param cluster_res_key: the key which specifies the clustering result in data.tl.result.
        :param micro_envs: two columns, column names should be "cell_type" and "microenvironment".
        :param species: 'HUMAN' or 'MOUSE'
        :param database: if species is HUMAN, choose from 'cellphonedb' or 'liana';
                        if MOUSE, use 'cellphonedb' or 'liana' or 'celltalkdb'
        :param homogene_path: path to the file storing mouse-human homologous genes ralations.
                        if species is MOUSE but database is 'cellphonedb' or 'liana', we need to use the human
        homologous genes for the input mouse genes.
        :param counts_identifiers: type of gene identifiers in the Counts data: "ensembl", "gene_name" or "hgnc_symbol".
        :param subsampling: flag of subsampling.
        :param subsampling_log: flag of doing log1p transformation before subsampling.
        :param subsampling_num_pc: number of pcs used when doing subsampling, <= min(m,n).
        :param subsampling_num_cells: size of the subsample.
        :param pca_res_key: the key which specifies the pca result in data.tl.result
                        if set subsampling to True and set it to None, this function will run the pca.
        :param separator_cluster: separator of cluster names used in the result and plots, e.g. '|'.
        :param separator_interaction: separator of interactions used in the result and plots, e.g. '_'.
        :param iterations: number of samples used for generating the null distribution.
        :param threshold: threshold of percentage of gene expression, above which being considered as significant.
        :param processes: number of processes used for doing the statistical analysis, on notebook just only support one process.
        :param pvalue: the cut-point of p-value, below which being considered significant.
        :param result_precision: result precision for the results, default=3.
        :param output_path: the directory to save the result file, set it to output the result to files.
        :param means_filename: name of the means result file.
        :param pvalues_filename: name of the pvalues result file.
        :param significant_means_filename: name of the significant mean result file.
        :param deconvoluted_filename: name of the convoluted result file.
        :param output_format: format of result, 'txt', 'csv', 'tsv', 'tab'.
        :return:
        """
        if subsampling and pca_res_key is not None and pca_res_key not in self.pipeline_res:
            raise PipelineResultInexistent(pca_res_key)

        database_dir = os.path.join(stereo_conf.data_dir, "algorithm/cell_cell_communication/database")
        if homogene_path is None:
            homogene_path = os.path.join(database_dir, 'mouse2human.csv')
        # 0. prepare the reference database
        # FIXME: change the default database path
        if database == 'cellphonedb':
            db_path = os.path.join(database_dir, 'cellphonedb.db')
        elif database == 'liana':
            db_path = os.path.join(database_dir, 'liana.db')
        elif database == 'celltalkdb':
            db_path = os.path.join(database_dir, 'celltalkdb.db')
        else:
            raise InvalidDatabase()

        interactions, genes, complex_composition, complex_expanded = self._get_ref_database(db_path)

        counts, meta = self._prepare_data(cluster_res_key)

        # 1. preprocess and validate input data

        # 1.1. preprocess and validate meta data (cell name as index, cell type as the only column).
        # meta = self._check_meta_data(meta)

        # 1.2. preprocess and validate counts data
        self._check_counts_data(counts, counts_identifiers)
        counts = self._counts_validations(counts, meta)

        # 1.3. if species is mouse, get the homologous genes.
        if species.upper() == 'MOUSE' and (database == 'cellphonedb' or database == 'liana'):
            genes_mouse = counts.index.tolist()
            genes_human = mouse2human(genes_mouse, homogene_path)
            counts.index = genes_human
            counts = counts.drop('NotAvailable')
            counts = counts.groupby(counts.index, as_index=True).sum()

        # 1.4. preprocess and validate micro_env data
        if micro_envs is None:
            micro_envs = pd.DataFrame()
        else:
            micro_envs = self._check_microenvs_data(micro_envs, meta)

        # 1.5. preprocess and validate other parameters
        threshold = float(threshold)
        if threshold < 0 or threshold > 1:
            raise ThresholdValueException(threshold)

        # 2. subsampling if required
        if subsampling:
            subsampler = Subsampler(subsampling_log, subsampling_num_pc, subsampling_num_cells)
        else:
            subsampler = None

        if subsampler is not None:
            if pca_res_key is not None:
                counts = subsampler.subsample(counts, self.pipeline_res[pca_res_key])
            else:
                counts = subsampler.subsample(counts)
            meta = meta.filter(items=list(counts), axis=0)

        # 3. do the analysis
        # 3.1. filter input and database data
        if analysis_type == 'statistical':
            logger.info(
                '[{} analysis] Threshold:{} Precision:{} Iterations:{} Threads:{}'.format(analysis_type,
                                                                                          threshold,
                                                                                          result_precision,
                                                                                          iterations,
                                                                                          processes))
        if analysis_type == 'simple':
            logger.info(
                '[{} analysis] Threshold:{} Precision:{}'.format(analysis_type, threshold, result_precision))

        interactions_reduced = interactions[['multidata_1_id', 'multidata_2_id']].drop_duplicates()

        # add id_multidata as index to counts, calculate mean grouped by id_multidata:
        # counts is the grouped means, counts_relations includes 'id_multidata', 'ensembl', 'gene_name', 'hgnc_symbol'.
        counts, counts_relations = self.add_multidata_and_means_to_counts(counts, genes, counts_identifiers)
        # filter the complex_composition, interactions_reduced, counts data
        complex_composition_filtered, interactions_filtered, counts_filtered = self.prefilters(interactions_reduced,
                                                                                               counts,
                                                                                               complex_composition)
        if interactions_filtered.empty:
            raise NoInteractionsFound()

        meta = meta.loc[counts.columns]

        # 3.2. build the cluster (means, percentages) and do the analysis

        # dict: cluster names, cluster means of proteins and complexes (min),
        # cluster percentages of proteins and complexes (min).
        clusters = self.build_clusters(meta, counts_filtered, complex_composition_filtered, skip_percent=False)

        logger.info('Running Real Analysis')
        cluster_interactions = self.get_cluster_combinations(clusters['names'], micro_envs)  # arrays

        base_result = self.build_result_matrix(interactions_filtered, cluster_interactions, separator_cluster)

        # (x > 0) * (y > 0) * (x + y) / 2
        real_mean_analysis = self.mean_analysis(interactions_filtered, clusters, cluster_interactions,
                                                separator_cluster)

        # ((x > threshold) * (y > threshold)).astype(int)
        real_percents_analysis = self.percent_analysis(clusters, threshold, interactions_filtered, cluster_interactions,
                                                       separator_cluster)

        if analysis_type == 'statistical':
            logger.info('Running Statistical Analysis')
            statistical_mean_analysis = self.shuffled_analysis(iterations,
                                                               meta,
                                                               counts_filtered,
                                                               interactions_filtered,
                                                               cluster_interactions,
                                                               complex_composition_filtered,
                                                               real_mean_analysis,
                                                               processes,
                                                               separator_cluster)
            result_pvalues = self.build_pvalue_result(real_mean_analysis,
                                                      real_percents_analysis,
                                                      statistical_mean_analysis,
                                                      base_result)
        else:
            result_pvalues = pd.DataFrame()

        # 3.3. output results

        pvalues_result = None
        means_result = None
        significant_means = None
        deconvoluted_result = None

        if analysis_type == 'simple':
            pvalues_result, means_result, significant_means, deconvoluted_result = self.build_results(
                analysis_type,
                interactions_filtered,
                interactions,
                counts_relations,
                real_mean_analysis,
                real_percents_analysis,
                clusters['means'],
                complex_composition_filtered,
                counts,
                genes,
                result_precision,
                pvalue,
                counts_identifiers,
                separator_interaction
            )
        if analysis_type == 'statistical':
            pvalues_result, means_result, significant_means, deconvoluted_result = self.build_results(
                analysis_type,
                interactions_filtered,
                interactions,
                counts_relations,
                real_mean_analysis,
                result_pvalues,
                clusters['means'],
                complex_composition_filtered,
                counts,
                genes,
                result_precision,
                pvalue,
                counts_identifiers,
                separator_interaction
            )
        max_rank = significant_means['rank'].max()
        significant_means['rank'] = significant_means['rank'].apply(
            lambda rank: rank if rank != 0 else (1 + max_rank))
        significant_means.sort_values('rank', inplace=True)  # min to max, 0s at the bottom

        self.pipeline_res[res_key] = {
            'means': means_result,
            'significant_means': significant_means,
            'deconvoluted': deconvoluted_result,
            # 'interactions_filtered': interactions_filtered,
            # 'interactions': interactions
        }
        if analysis_type == "statistical":
            self.pipeline_res[res_key]['pvalues'] = pvalues_result
        self.pipeline_res[res_key]['parameters'] = {
            'analysis_type': analysis_type,
            'cluster_res_key': cluster_res_key
        }
        self.stereo_exp_data.tl.reset_key_record('cell_cell_communication', res_key)

        if output_path is not None:
            logger.info('Writing results to files')
        # Todo: Test output_path in linux
            write_to_file(means_result, means_filename, output_path=output_path, output_format=output_format)
            write_to_file(significant_means, significant_means_filename, output_path=output_path, output_format=output_format)
            write_to_file(deconvoluted_result, deconvoluted_filename, output_path=output_path, output_format=output_format)
            if analysis_type == "statistical":
                write_to_file(pvalues_result, pvalues_filename, output_path=output_path, output_format=output_format)

    def _prepare_data(self, cluster_res_key):
        if cluster_res_key not in self.pipeline_res:
            raise PipelineResultInexistent(cluster_res_key)
        cluster: pd.DataFrame = self.pipeline_res[cluster_res_key].copy()
        cluster['bins'] = cluster['bins'].astype(str)
        cluster.rename({'group': 'cell_type'}, axis=1, inplace=True)
        cluster.set_index('bins', drop=True, inplace=True)
        cluster.index.name = 'cell'
        data = pd.DataFrame(self.stereo_exp_data.exp_matrix.T.toarray())
        data.columns = self.stereo_exp_data.cell_names.astype(str)
        data.index = self.stereo_exp_data.gene_names
        return data, cluster

    def _get_ref_database(self, db_path):
        """
        preprocessing the reference database
        """
        url = 'sqlite:///{}'.format(db_path)
        engine = create_engine(url)
        database = Database(engine)
        database.base_model = Base
        database_manager = DatabaseManager(None, database)
        # load repositories
        database_manager.add_repository(ComplexRepository)
        database_manager.add_repository(GeneRepository)
        database_manager.add_repository(InteractionRepository)
        database_manager.add_repository(MultidataRepository)
        database_manager.add_repository(ProteinRepository)

        # get data form database
        interactions = database_manager.get_repository('interaction').get_all_expanded(include_gene=False)
        genes = database_manager.get_repository('gene').get_all_expanded()  # join gene, protein, multidata
        complex_composition = database_manager.get_repository('complex').get_all_compositions()
        complex_expanded = database_manager.get_repository('complex').get_all_expanded()

        # index interactions and complex dataframes
        interactions.set_index('id_interaction', drop=True, inplace=True)
        complex_composition.set_index('id_complex_composition', inplace=True, drop=True)

        return interactions, genes, complex_composition, complex_expanded

    
    def _check_meta_data(self, meta_raw: pd.DataFrame):
        """
        Preprocess the meta dataframe:
        When the dataframe has both "cell" and "cell_type" columns, take "cell" as the index and "cell_type" as the
        only column.
        When the dataframe does not have a "cell" column and the index type is range, take the 1st column of meta as
        "cell" and use it as the index.
        When the dataframe does not have a "cell" column and the index type is base, rename the index as "cell".
        When the dataframe has no "cell" and "cell_type" at all, take the 1st column as the index and the 2nd column
        as "cell_type".
        """
        meta_raw.columns = map(str.lower, meta_raw.columns)
        try:
            if 'cell' in meta_raw and 'cell_type' in meta_raw:
                meta = meta_raw[['cell', 'cell_type']]
                meta.set_index('cell', inplace=True, drop=True)
                return meta

            if type(meta_raw.index) == pd.core.indexes.multi.MultiIndex:
                raise ProcessMetaException

            elif 'cell_type' in meta_raw:
                meta = meta_raw[['cell_type']]
                if type(meta_raw.index) == pd.core.indexes.range.RangeIndex:
                    meta.set_index(meta_raw.iloc[:, 0], inplace=True)
                    meta.index.name = 'cell'
                    return meta

                if type(meta_raw.index) == pd.core.indexes.base.Index:
                    meta.index.name = 'cell'
                    return meta

            meta = pd.DataFrame(data={'cell_type': meta_raw.iloc[:, 1]})
            meta.set_index(meta_raw.iloc[:, 0], inplace=True)
            meta.index.name = 'cell'
            meta.index = meta.index.astype(str)
            return meta
        except:
            raise ProcessMetaException

    
    def _check_counts_data(self, counts: pd.DataFrame, counts_data: str) -> None:
        """Naive check count data against counts gene names.

        This method quickly checks if count_data matches the all gene names and
        gives a comprehensive warning.

        Parameters
        ----------
        counts: pd.DataFrame
            Counts data
        counts_data: str
            Gene identifier expected in counts data
        """
        if ~np.all(counts.index.str.startswith(("ENSG0", "ENSMUSG0"))) and counts_data == "ensembl":
            logger.warning(f"Gene format missmatch. Using gene type '{counts_data}' "
                           f"expects gene names to start with 'ENSG' (human) or 'ENSMUSG0' (mouse) but "
                           f"some genes seem to be in another format. "
                           f"Try using hgnc_symbol if all counts are filtered.")

    
    def _counts_validations(self, counts: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        """
        Change counts type to np.float32.
        Check if the column names of counts matches the indexes of meta.
        """
        if not len(counts.columns):
            raise ParseCountsException('Counts values are not decimal values', 'Incorrect file format')
        try:
            if np.any(counts.dtypes.values != np.dtype('float32')):
                counts = counts.astype(np.float32)
        except:
            raise ParseCountsException("Counts values cannot be changed to np.float32")

        meta.index = meta.index.astype(str)

        if np.any(~meta.index.isin(counts.columns)):
            raise ParseCountsException("Some cells in meta did not exist in counts",
                                       "Maybe incorrect file format")

        if np.any(~counts.columns.isin(meta.index)):
            logger.debug("Dropping counts cells that are not present in meta")
            counts = counts.loc[:, counts.columns.isin(meta.index)].copy()
        return counts

    
    def _check_microenvs_data(self, microenvs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        """
        Runs validations to make sure the file has enough columns and that all the cell types in the microenvironment
        are included in meta.
        Rename the two columns as "cell_type" and "microenvironment".
        """
        microenvs.drop_duplicates(inplace=True)
        len_columns = len(microenvs.columns)
        if len_columns < 2:
            raise Exception(f"Missing columns in microenvironments: 2 required but {len_columns} provieded")
        elif len_columns > 2:
            logger.warning(f"Microenvrionemnts expects 2 columns and got {len_columns}. Droppoing extra columns.")
        microenvs = microenvs.iloc[:, 0:2]
        # if any(~microenvs.iloc[:, 0].isin(meta['cell_type'])):
        if not all(microenvs.iloc[:, 0].isin(meta['cell_type'])):
            raise Exception("Some clusters/cell_types in microenvironments are not present in meta")
        microenvs.columns = ["cell_type", "microenvironment"]
        return microenvs

    
    def add_multidata_and_means_to_counts(self, counts: pd.DataFrame, genes: pd.DataFrame, counts_identifiers: str):
        """Adds multidata and means to counts.

        This method merges multidata ids into counts data using counts_identifiers as column name for the genes.
        Then sorts the counts columns based on the cell names, makes sure count data is of type float32 and finally
        calculates the means grouped by id_multidata.

        Returns
        -------
        Tuple: A tuple containing:
            - counts: counts data merged with mutidata and indexsed by id_multidata
            - counts_relations: a subset of counts with only id_multidata and all gene identifiers
        """
        # sort cell names
        cells_names = sorted(counts.columns)

        # add id multidata to counts input, INNER join, new index is the range index in genes.
        counts = counts.merge(
            genes[['id_multidata', 'ensembl', 'gene_name', 'hgnc_symbol']],
            left_index=True,
            right_on=counts_identifiers
        )

        if counts.empty:
            raise AllCountsFilteredException(hint='Are you using human data?')

        counts_relations = counts[['id_multidata', 'ensembl', 'gene_name', 'hgnc_symbol']].copy()

        counts.set_index('id_multidata', inplace=True, drop=True)  # id_multidata not unique
        counts = counts[cells_names]
        if np.any(counts.dtypes.values != np.dtype('float32')):
            counts = counts.astype(np.float32)
        # one protein could correspond to multiple genes
        # e.g. genes.loc[[21,22]]
        counts = counts.groupby(counts.index).mean()

        return counts, counts_relations

    def prefilters(
            self,
            interactions: pd.DataFrame,
            counts: pd.DataFrame,
            complex_composition: pd.DataFrame
        ):
        """
        Filter complex_composition, interaction and counts.
        """
        # Remove rows with all zero values
        if counts.empty:
            counts_filtered = counts
        else:
            counts_filtered = counts[counts.apply(lambda row: row.sum() > 0, axis=1)]
        # Filter complex_composition, keeping only complexes whose composing proteins are all in counts.
        # Also filter the counts, keep only proteins in the filtered complex_composition.
        complex_composition_filtered, counts_complex = self._filter_complex_composition_by_counts(counts_filtered,
                                                                                                  complex_composition)
        # Filter interactions, keeping only interactions whose two parts are both in the filtered complex or counts.
        interactions_filtered = self._filter_interactions_by_counts(interactions,
                                                                    counts_filtered,
                                                                    complex_composition_filtered)
        # Filter counts, keeping only simple proteins in the interactions.
        counts_simple = self._filter_counts_by_interactions(counts_filtered, interactions_filtered)
        # Combine counts of proteins in the interaction and proteins in the complexes.
        counts_filtered = counts_simple.append(counts_complex, sort=False)
        counts_filtered = counts_filtered[~counts_filtered.index.duplicated()]

        return complex_composition_filtered, interactions_filtered, counts_filtered

    
    def _filter_complex_composition_by_counts(
            self,
            counts: pd.DataFrame,
            complex_composition: pd.DataFrame
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter the counts and complex_composition:
        - keep only complexes whose composing proteins are all in the counts.
        - keep only the above proteins in counts.

        Returns
        -------
        Tuple: A tuple containing:
            - complex_composition filtered
            - counts filtered
        """
        proteins_in_complexes = complex_composition['protein_multidata_id'].drop_duplicates().tolist()

        # Remove counts that can't be part of a complex
        counts_filtered = counts[counts.apply(lambda count: count.name in proteins_in_complexes, axis=1)]

        # Find complexes with all components defined in counts
        multidata_protein = list(counts_filtered.index)
        composition_filtered = complex_composition[complex_composition['protein_multidata_id'].apply(
            lambda protein_multidata: protein_multidata in multidata_protein)]

        if composition_filtered.empty:
            complex_composition_filtered = pd.DataFrame(columns=complex_composition.columns)
        else:
            def all_protein_involved(current_complex: pd.Series) -> bool:
                number_proteins_in_counts = len(composition_filtered[
                                                    composition_filtered['complex_multidata_id'] == current_complex[
                                                        'complex_multidata_id']])
                if number_proteins_in_counts < current_complex['total_protein']:
                    return False
                return True

            complex_composition_filtered = composition_filtered[
                composition_filtered.apply(all_protein_involved, axis=1)]

        if complex_composition_filtered.empty:
            return complex_composition_filtered, pd.DataFrame(columns=counts.columns)

        available_complex_proteins = complex_composition_filtered['protein_multidata_id'].drop_duplicates().to_list()

        # Remove counts that are not defined in selected complexes
        counts_filtered = counts_filtered[
            counts_filtered.apply(lambda count: count.name in available_complex_proteins, axis=1)]

        return complex_composition_filtered, counts_filtered

    
    def _filter_interactions_by_counts(
            self,
            interactions: pd.DataFrame,
            counts: pd.DataFrame,
            complex_composition: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Use filtered complex_composition and unfiltered counts to filter interactions.
        - keep only interactions that both parts are in the complex or counts.
        """
        multidatas = list(counts.index)

        if not complex_composition.empty:
            multidatas += complex_composition['complex_multidata_id'].to_list() + complex_composition[
                'protein_multidata_id'].to_list()

        multidatas = list(set(multidatas))

        def filter_interactions(interaction: pd.Series) -> bool:
            if interaction['multidata_1_id'] in multidatas and interaction['multidata_2_id'] in multidatas:
                return True
            return False

        interactions_filtered = interactions[interactions.apply(filter_interactions, axis=1)]

        return interactions_filtered

    
    def _filter_counts_by_interactions(
            self,
            counts: pd.DataFrame,
            interactions: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Removes counts if is not defined in interactions components.
        """
        multidata_ids = interactions['multidata_1_id'].append(
            interactions['multidata_2_id']).drop_duplicates().tolist()

        counts_filtered = counts.filter(multidata_ids, axis=0)

        return counts_filtered

    
    def build_clusters(
            self,
            meta: pd.DataFrame,
            counts: pd.DataFrame,
            complex_composition: pd.DataFrame,
            skip_percent: bool
        ) -> dict:
        """
        Build the means and percent values for each cluster and stores the results in a dictionary with the
        following keys: 'names', 'means' and 'percents'.

        Parameters
        ----------
        meta: pd.DataFrame
            Meta data.
        counts: pd.DataFrame
            Counts data
        complex_composition: pd.DataFrame
            Complex data.
        skip_percent: bool
            Calculate percent values for each cluster or not.
        Returns
        -------
        dict: Dictionary containing the following:
            - names: cluster names
            - means: cluster means
            - percents: cluster percents
        """
        meta['cell_type'] = meta['cell_type'].astype('category')
        cluster_names = meta['cell_type'].cat.categories

        # cell counts to cluster counts
        cluster_means = pd.DataFrame(
            npg.aggregate(meta['cell_type'].cat.codes, counts.values, func='mean', axis=1),
            index=counts.index,
            columns=cluster_names.to_list()
        )
        if not skip_percent:
            cluster_pcts = pd.DataFrame(
                npg.aggregate(meta['cell_type'].cat.codes, (counts > 0).astype(int).values, func='mean', axis=1),
                index=counts.index,
                columns=cluster_names.to_list()
            )
        else:
            cluster_pcts = pd.DataFrame(index=counts.index, columns=cluster_names.to_list())

        # Complex genes cluster counts
        if not complex_composition.empty:
            complexes = complex_composition.groupby('complex_multidata_id').apply(
                lambda x: x['protein_multidata_id'].values).to_dict()
            # complex_id as rows, clusters as columns
            complex_cluster_means = pd.DataFrame(
                {complex_id: cluster_means.loc[protein_ids].min(axis=0).values
                 for complex_id, protein_ids in complexes.items()},
                index=cluster_means.columns
            ).T
            cluster_means = cluster_means.append(complex_cluster_means)
            if not skip_percent:
                complex_cluster_pcts = pd.DataFrame(
                    {complex_id: cluster_pcts.loc[protein_ids].min(axis=0).values
                     for complex_id, protein_ids in complexes.items()},
                    index=cluster_pcts.columns
                ).T
                cluster_pcts = cluster_pcts.append(complex_cluster_pcts)

        return {'names': cluster_names, 'means': cluster_means, 'percents': cluster_pcts}

    
    def get_cluster_combinations(self, cluster_names: np.array, microenvs: pd.DataFrame = pd.DataFrame()) -> np.array:
        """
        Calculates and sorts combinations of clusters.

        Generates all possible combinations between the 'cluster_names' provided.
        Combinations include each cluster with itself.
        If `microenvs` is provided then the combinations are limited to the
        clusters within each microenvironment as specified.

        Parameters
        ----------
        cluster_names: np.array
            Array of cluster names.
        microenvs: pd.DataFrame
            Microenvironments data.

        Example
        -------
        INPUT
        cluster_names = ['cluster1', 'cluster2', 'cluster3']

        RESULT
        [('cluster1','cluster1'),('cluster1','cluster2'),('cluster1','cluster3'),
         ('cluster2','cluster1'),('cluster2','cluster2'),('cluster2','cluster3'),
         ('cluster3','cluster1'),('cluster3','cluster2'),('cluster3','cluster3')]

        if microenvironments are provided combinations are performed only within each microenv

        INPUT
        cluster_names = ['cluster1', 'cluster2', 'cluster3']
        microenvs = [
            ('cluster1', 'env1'),
            ('cluster2', 'env1'),
            ('cluster3', 'env2')]

        RESULT
        [('cluster1','cluster1'),('cluster1','cluster2'),
         ('cluster2','cluster1'),('cluster2','cluster2'),
         ('cluster3','cluster3')]

        Returns
        -------
        np.array
            An array of arrays representing cluster combinations. Each inner array
            represents the combination of two clusters.
        """
        result = np.array([])
        if microenvs.empty:
            result = np.array(np.meshgrid(cluster_names.values, cluster_names.values)).T.reshape(-1, 2)
        else:
            logger.info('Limiting cluster combinations using microenvironments')
            cluster_combinations = []
            for me in microenvs["microenvironment"].unique():
                me_cell_types = microenvs[microenvs["microenvironment"] == me]["cell_type"]
                combinations = np.array(np.meshgrid(me_cell_types, me_cell_types))
                cluster_combinations.extend(combinations.T.reshape(-1, 2))
            result = pd.DataFrame(cluster_combinations).drop_duplicates().to_numpy()
        logger.debug(f'Using {len(result)} cluster combinations for analysis')
        return result

    
    def build_result_matrix(self, interactions: pd.DataFrame, cluster_interactions: list, separator: str) -> pd.DataFrame:
        """
        builds an empty cluster matrix to fill it later, index is id_interaction
        """
        columns = []

        for cluster_interaction in cluster_interactions:
            columns.append('{}{}{}'.format(cluster_interaction[0], separator, cluster_interaction[1]))

        result = pd.DataFrame(index=interactions.index, columns=columns, dtype=float)

        return result

    
    def mean_analysis(
            self,
            interactions: pd.DataFrame,
            clusters: dict,
            cluster_interactions: list,
            separator: str
        ) -> pd.DataFrame:
        """
        Calculates the mean for the list of interactions and for each cluster interaction

        Based on the interactions from CellPhoneDB database (gene1|gene2) and each
        cluster means (gene|cluser) this method calculates the mean of an interaction
        (gene1|gene2) and a cluster combination (cluster1|cluster2). When any of the
        values is 0, the result is set to 0, otherwise the mean is used. The following
        expression is used to get the result `(x > 0) * (y > 0) * (x + y) / 2` where
        `x = mean(gene1|cluster1)` and `y = mean(gene2|cluster2)` and the output is
        expected to be mean(gene1|gene2, cluster1|cluster2).

        Parameters
        ----------
        interactions: pd.DataFrame
            Interactions from CellPhoneDB database. Gene names will be taken from
            here and interpret as 'multidata_1_id' for gene1 and 'multidata_2_id'
            for gene2.
        clusters: dict
            Clusters information. 'means' key will be used to get the means of a
            gene/cluster combination/
        cluster_interactions: list
            List of cluster interactions obtained from the combination of the cluster
            names and possibly filtered using microenvironments.
        separator: str
            Character to use as a separator when joining cluster as column names.


        Example
        ----------
            cluster_means
                       cluster1    cluster2    cluster3
            ensembl1     0.0         0.2         0.3
            ensembl2     0.4         0.5         0.6
            ensembl3     0.7         0.0         0.9

            interactions:

            ensembl1,ensembl2
            ensembl2,ensembl3

            RESULT:
                                  cluster1_cluster1   cluster1_cluster2   ...   cluster3_cluster2   cluster3_cluster3
            ensembl1_ensembl2     mean(0.0,0.4)*      mean(0.0,0.5)*            mean(0.3,0.5)       mean(0.3,0.6)
            ensembl2_ensembl3     mean(0.4,0.7)       mean(0.4,0.0)*            mean(0.6,0.0)*      mean(0.6,0.9)


            results with * are 0 because one of both components is 0.

        Returns
        -------
        DataFrame
            A DataFrame where each column is a cluster combination (cluster1|cluster2)
            and each row represents an interaction (gene1|gene2). Values are the mean
            for that interaction and that cluster combination.
        """
        GENE_ID1 = 'multidata_1_id'
        GENE_ID2 = 'multidata_2_id'

        cluster1_names = cluster_interactions[:, 0]
        cluster2_names = cluster_interactions[:, 1]
        gene1_ids = interactions[GENE_ID1].values
        gene2_ids = interactions[GENE_ID2].values

        x = clusters['means'].loc[gene1_ids, cluster1_names].values
        y = clusters['means'].loc[gene2_ids, cluster2_names].values

        result = pd.DataFrame(
            (x > 0) * (y > 0) * (x + y) / 2,
            index=interactions.index,
            columns=(pd.Series(cluster1_names) + separator + pd.Series(cluster2_names)).values)

        return result

    
    def percent_analysis(
            self,
            clusters: dict,
            threshold: float,
            interactions: pd.DataFrame,
            cluster_interactions: list,
            separator: str
        ) -> pd.DataFrame:
        """
        Calculates the percents for cluster interactions and for each gene
        interaction.

        This method builds a gene1|gene2,cluster1|cluster2 table of percent values.
        As the first step, calculates the percents for each gene|cluster. The cluster
        percent is 0 if the number of positive cluster cells divided by total of
        cluster cells is greater than threshold and 1 if not. If one of both is NOT 0
        then sets the value to 0 else sets the value to 1. Then it calculates the
        percent value of the interaction.

        Parameters
        ----------
        clusters: dict
            Clusters information. 'percents' key will be used to get the precent of a
            gene/cell combination.
        threshold: float
            Cutoff value for percentages (number of positive cluster cells divided
            by total of cluster cells).
        interactions: pd.DataFrame
            Interactions from CellPhoneDB database. Gene names will be taken from
            here and interpret as 'multidata_1_id' for gene1 and 'multidata_2_id'
            for gene2.
        cluster_interactions: list
            List of cluster interactions obtained from the combination of the cluster
            names and possibly filtered using microenvironments.
        separator: str
            Character to use as a separator when joining cluster as column names.

        Returns
        ----------
        pd.DataFrame:
            A DataFrame where each column is a cluster combination (cluster1|cluster2)
            and each row represents an interaction (gene1|gene2). Values are the percent
            values calculated for each interaction and cluster combination.
        """
        GENE_ID1 = 'multidata_1_id'
        GENE_ID2 = 'multidata_2_id'

        cluster1_names = cluster_interactions[:, 0]
        cluster2_names = cluster_interactions[:, 1]
        gene1_ids = interactions[GENE_ID1].values
        gene2_ids = interactions[GENE_ID2].values

        x = clusters['percents'].loc[gene1_ids, cluster1_names].values
        y = clusters['percents'].loc[gene2_ids, cluster2_names].values

        result = pd.DataFrame(
            ((x > threshold) * (y > threshold)).astype(int),
            index=interactions.index,
            columns=(pd.Series(cluster1_names) + separator + pd.Series(cluster2_names)).values)

        return result

    def shuffled_analysis(
            self,
            iterations: int,
            meta: pd.DataFrame,
            counts: pd.DataFrame,
            interactions: pd.DataFrame,
            cluster_interactions: list,
            complex_composition: pd.DataFrame,
            real_mean_analysis: pd.DataFrame,
            processes: int,
            separator: str
        ) -> list:
        """
        Shuffles meta and calculates the means for each and saves it in a list.

        Runs it in a multiple processes to run it faster

        Note that on notebook just only support one process
        """
        statistical_analysis_thread = partial(self._statistical_analysis,
                                            cluster_interactions,
                                            counts,
                                            interactions,
                                            meta,
                                            complex_composition,
                                            separator,
                                            real_mean_analysis)
        if processes > 1:
            with Pool(processes=processes) as pool:
                results = pool.map(statistical_analysis_thread, range(iterations))
        else:
            results = [statistical_analysis_thread(i) for i in range(iterations)]
        return results

    def _statistical_analysis(
            self,
            cluster_interactions: list,
            counts: pd.DataFrame,
            interactions: pd.DataFrame,
            meta: pd.DataFrame,
            complex_composition: pd.DataFrame,
            separator: str,
            real_mean_analysis: pd.DataFrame,
            iteration_number: int
    ):
        """
        Shuffles meta dataset and calculates the means
        """

        def shuffle_meta(meta: pd.DataFrame) -> pd.DataFrame:
            """
            Get a randomly shuffled copy of the input meta.
            """
            meta_copy = meta.copy()
            labels = list(meta_copy['cell_type'].values)
            np.random.shuffle(labels)
            meta_copy['cell_type'] = labels
            return meta_copy

        shuffled_meta = shuffle_meta(meta)
        shuffled_clusters = self.build_clusters(shuffled_meta, counts, complex_composition, skip_percent=True)
        shuffled_mean_analysis = self.mean_analysis(interactions,
                                                    shuffled_clusters,
                                                    cluster_interactions,
                                                    separator)

        result_mean_analysis = np.packbits(shuffled_mean_analysis.values > real_mean_analysis.values, axis=None)
        return result_mean_analysis

    
    def build_pvalue_result(
            self,
            real_mean_analysis: pd.DataFrame,
            real_percents_analysis: pd.DataFrame,
            statistical_mean_analysis: list,
            base_result: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates the pvalues after statistical analysis.

        If real_percent or real_mean are zero, result_pvalue is 1

        If not:
        Calculates how many shuffled means are bigger than real mean and divides it for the number of
        the total iterations

        Parameters
        ----------
        real_mean_analysis: pd.DataFrame
            Means cluster analyisis
        real_percents_analysis: pd.DataFrame
            Percents cluster analyisis
        statistical_mean_analysis: list
            Statitstical means analyisis
        base_result: pd.DataFrame
            Contains the index and columns that will be used by the returned object

        Returns
        -------
        pd.DataFrame
            A DataFrame with interactions as rows and cluster combinations as columns.
        """
        logger.info('Building Pvalues result')
        percent_result = np.zeros(real_mean_analysis.shape)
        result_size = percent_result.size
        result_shape = percent_result.shape

        for statistical_mean in statistical_mean_analysis:
            percent_result += np.unpackbits(statistical_mean, axis=None)[:result_size].reshape(result_shape)
        percent_result /= len(statistical_mean_analysis)

        mask = (real_mean_analysis.values == 0) | (real_percents_analysis == 0)

        percent_result[mask] = 1

        return pd.DataFrame(percent_result, index=base_result.index, columns=base_result.columns)

    def build_results(
            self,
            analysis_type,
            interactions: pd.DataFrame,
            interactions_original: pd.DataFrame,
            counts_relations: pd.DataFrame,
            real_mean_analysis: pd.DataFrame,
            result_percent: pd.DataFrame,
            clusters_means: pd.DataFrame,
            complex_compositions: pd.DataFrame,
            counts: pd.DataFrame,
            genes: pd.DataFrame,
            result_precision: int,
            pvalue: float = None,
            counts_data: str = None,
            separator: str = '|'
    ):
        """
        Sets the results data structure from method generated data. Results documents are defined by specs.
        """
        logger.info('Building results')
        interactions: pd.DataFrame = interactions_original.loc[interactions.index]  # get full interaction info
        interactions['interaction_index'] = interactions.index
        # add 'id_multidata', 'ensembl', 'gene_name', 'hgnc_symbol'
        interactions = interactions.merge(counts_relations, how='left', left_on='multidata_1_id',
                                          right_on='id_multidata', )
        interactions = interactions.merge(counts_relations, how='left', left_on='multidata_2_id',
                                          right_on='id_multidata',
                                          suffixes=('_1', '_2'))
        interactions.set_index('interaction_index', inplace=True, drop=True)

        interacting_pair = self._interacting_pair_build(interactions, separator)

        def simple_complex_indicator(interaction: pd.Series, suffix: str) -> str:
            """
            Add simple/complex prefixes to interaction components
            """
            if interaction['is_complex{}'.format(suffix)]:
                return 'complex:{}'.format(interaction['name{}'.format(suffix)])

            return 'simple:{}'.format(interaction['name{}'.format(suffix)])

        interactions['partner_a'] = interactions.apply(lambda interaction: simple_complex_indicator(interaction, '_1'),
                                                       axis=1)
        interactions['partner_b'] = interactions.apply(lambda interaction: simple_complex_indicator(interaction, '_2'),
                                                       axis=1)

        significant_means = None
        significant_mean_rank = None
        if analysis_type == 'simple':
            significant_mean_rank, significant_means = self.build_significant_means(real_mean_analysis, result_percent)
        if analysis_type == 'statistical':
            significant_mean_rank, significant_means = self.build_significant_means(real_mean_analysis,
                                                                                    result_percent,
                                                                                    pvalue)
        significant_means = significant_means.round(result_precision)

        gene_columns = ['{}_{}'.format(counts_data, suffix) for suffix in ('1', '2')]  # ['ensembl_1', 'ensembl_2']
        gene_renames = {column: 'gene_{}'.format(suffix) for column, suffix in zip(gene_columns, ['a', 'b'])}

        # Remove useless columns
        interactions_data_result = pd.DataFrame(
            interactions[['id_cp_interaction', 'partner_a', 'partner_b', 'receptor_1', 'receptor_2', *gene_columns,
                          'annotation_strategy']].copy())

        interactions_data_result = pd.concat([interacting_pair, interactions_data_result], axis=1, sort=False)

        interactions_data_result['secreted'] = (interactions['secreted_1'] | interactions['secreted_2'])
        interactions_data_result['is_integrin'] = (interactions['integrin_1'] | interactions['integrin_2'])

        interactions_data_result.rename(
            columns={**gene_renames, 'receptor_1': 'receptor_a', 'receptor_2': 'receptor_b'},
            inplace=True)

        # Dedupe rows and filter only desired columns
        interactions_data_result.drop_duplicates(inplace=True)

        means_columns = ['id_cp_interaction', 'interacting_pair', 'partner_a', 'partner_b', 'gene_a', 'gene_b',
                         'secreted',
                         'receptor_a', 'receptor_b', 'annotation_strategy', 'is_integrin']

        interactions_data_result = interactions_data_result[means_columns]

        real_mean_analysis = real_mean_analysis.round(result_precision)
        significant_means = significant_means.round(result_precision)

        # Round result decimals
        for key, cluster_means in clusters_means.items():
            clusters_means[key] = cluster_means.round(result_precision)

        # Document 1
        pvalues_result = pd.DataFrame()
        if analysis_type == 'statistical':
            pvalues_result = pd.merge(interactions_data_result, result_percent, left_index=True, right_index=True,
                                      how='inner')

        # Document 2
        means_result = pd.merge(interactions_data_result, real_mean_analysis, left_index=True, right_index=True,
                                how='inner')

        # Document 3
        significant_means_result = pd.merge(interactions_data_result, significant_mean_rank, left_index=True,
                                            right_index=True,
                                            how='inner')
        significant_means_result = pd.merge(significant_means_result, significant_means, left_index=True,
                                            right_index=True,
                                            how='inner')

        # Document 4
        deconvoluted_result = self.deconvoluted_complex_result_build(clusters_means,
                                                                     interactions,
                                                                     complex_compositions,
                                                                     counts,
                                                                     genes,
                                                                     counts_data)

        def fillna_func(column: pd.Series):
            if column.dtype == object:
                return column.fillna(value='')
            if column.dtype == bool:
                return column.fillna(value=False)
            return column.fillna(value=-1)
        
        pvalues_result = pvalues_result.apply(fillna_func, axis=0)
        means_result = means_result.apply(fillna_func, axis=0)
        significant_means_result = significant_means_result.apply(fillna_func, axis=0)
        deconvoluted_result = deconvoluted_result.apply(fillna_func, axis=0)

        return pvalues_result, means_result, significant_means_result, deconvoluted_result

    
    def _interacting_pair_build(self, interactions: pd.DataFrame, separator) -> pd.Series:
        """
        Returns the interaction result formated with name1_name2
        """

        def get_interactor_name(interaction: pd.Series, suffix: str) -> str:
            """
            If part of interaction is complex, return name; if not, return gene_name
            """
            if interaction['is_complex{}'.format(suffix)]:
                return interaction['name{}'.format(suffix)]

            return interaction['gene_name{}'.format(suffix)]

        interacting_pair = interactions.apply(
            lambda interaction: '{}{}{}'.format(get_interactor_name(interaction, '_1'), separator,
                                                get_interactor_name(interaction, '_2')), axis=1)

        interacting_pair.rename('interacting_pair', inplace=True)

        return interacting_pair

    def build_significant_means(
            self,
            real_mean_analysis: pd.DataFrame,
            result_percent: pd.DataFrame,
            min_significant_mean: float = None
        ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculates the significant means and adds rank (number of non-empty entries divided by total entries)
        :param real_mean_analysis: the real mean results
        :param result_percent: the real percent results if simple analysis, pvalue results if statistical analysis
        :param min_significant_mean:
        """
        significant_means = self._get_significant_means(real_mean_analysis, result_percent, min_significant_mean)
        significant_mean_rank = significant_means.count(axis=1)  # type: pd.Series
        number_of_clusters = len(significant_means.columns)
        significant_mean_rank = significant_mean_rank.apply(lambda rank: rank / number_of_clusters)
        significant_mean_rank = significant_mean_rank.round(3)
        significant_mean_rank.name = 'rank'
        return significant_mean_rank, significant_means

    
    def _get_significant_means(
            self,
            real_mean_analysis: pd.DataFrame,
            result_percent: pd.DataFrame,
            min_significant_mean: float = None
        ) -> pd.DataFrame:
        """
        Get the significant means for gene1_gene2|cluster1_cluster2.

        For statistical_analysis `min_signigicant_mean` needs to be provided
        and if `result_percent > min_significant_mean` then sets the value to
        NaN otherwise uses the mean.
        For simple analysis `min_signigicant_mean` is NOT provided
        and uses `result_percent == 0` to set NaN, otherwise uses the mean.

        Parameters
        ----------
        real_mean_analysis : pd.DataFrame
            Mean results for each gene|cluster combination
        result_percent : pd.DataFrame
            Percent results for each gene|cluster combination
            - Simple analysis: real percent result
            - Statistical analysis: p-value result
        min_significant_mean : float,optional
            - Simple analysis, 0.
            - Statistical analysis: Filter p-value > min_significant_mean.

        Returns
        -------
        pd.DataFrame
            Significant means data frame. Columns are cluster interactions (cluster1|cluster2)
            and rows are NaN if there is no significant interaction or the mean value of the
            interaction if it is a relevant interaction.
        """
        significant_means = real_mean_analysis.values.copy()
        if min_significant_mean:
            mask = result_percent > min_significant_mean
        else:
            mask = result_percent == 0
        significant_means[mask] = np.nan
        return pd.DataFrame(significant_means,
                            index=real_mean_analysis.index,
                            columns=real_mean_analysis.columns)

    def deconvoluted_complex_result_build(
            self,
            clusters_means: pd.DataFrame,
            interactions: pd.DataFrame,
            complex_compositions: pd.DataFrame,
            counts: pd.DataFrame,
            genes: pd.DataFrame,
            counts_data: str
        ) -> pd.DataFrame:
        genes_counts = list(counts.index)
        genes_filtered = genes[genes['id_multidata'].apply(lambda gene: gene in genes_counts)]

        deconvoluted_complex_result_1 = self._deconvolute_complex_interaction_component(complex_compositions,
                                                                                        genes_filtered,
                                                                                        interactions,
                                                                                        '_1',
                                                                                        counts_data)
        deconvoluted_simple_result_1 = self._deconvolute_interaction_component(interactions,
                                                                               '_1',
                                                                               counts_data)

        deconvoluted_complex_result_2 = self._deconvolute_complex_interaction_component(complex_compositions,
                                                                                        genes_filtered,
                                                                                        interactions,
                                                                                        '_2',
                                                                                        counts_data)
        deconvoluted_simple_result_2 = self._deconvolute_interaction_component(interactions,
                                                                               '_2',
                                                                               counts_data)

        deconvoluted_result = deconvoluted_complex_result_1.append(
            [deconvoluted_simple_result_1, deconvoluted_complex_result_2, deconvoluted_simple_result_2], sort=False)

        deconvoluted_result.set_index('multidata_id', inplace=True, drop=True)

        deconvoluted_columns = ['gene_name', 'name', 'is_complex', 'protein_name', 'complex_name', 'id_cp_interaction',
                                'gene']

        deconvoluted_result = deconvoluted_result[deconvoluted_columns]
        deconvoluted_result.rename({'name': 'uniprot'}, axis=1, inplace=True)
        deconvoluted_result = pd.concat([deconvoluted_result, clusters_means.reindex(deconvoluted_result.index)],
                                        axis=1, join='inner', sort=False)
        deconvoluted_result.set_index('gene', inplace=True, drop=True)
        deconvoluted_result.drop_duplicates(inplace=True)

        return deconvoluted_result

    
    def _deconvolute_interaction_component(self, interactions, suffix, counts_data):
        interactions = interactions[~interactions['is_complex{}'.format(suffix)]]
        deconvoluted_result = pd.DataFrame()
        deconvoluted_result['gene'] = interactions['{}{}'.format(counts_data, suffix)]

        deconvoluted_result[
            ['multidata_id', 'protein_name', 'gene_name', 'name', 'is_complex', 'id_cp_interaction', 'receptor']] = \
            interactions[
                ['multidata{}_id'.format(suffix), 'protein_name{}'.format(suffix), 'gene_name{}'.format(suffix),
                 'name{}'.format(suffix),
                 'is_complex{}'.format(suffix), 'id_cp_interaction', 'receptor{}'.format(suffix)]]
        deconvoluted_result['complex_name'] = np.nan

        return deconvoluted_result

    
    def _deconvolute_complex_interaction_component(
            self,
            complex_compositions,
            genes_filtered,
            interactions,
            suffix,
            counts_data
        ):
        return_properties = [counts_data, 'protein_name', 'gene_name', 'name', 'is_complex', 'id_cp_interaction',
                             'receptor', 'complex_name']
        if complex_compositions.empty:
            return pd.DataFrame(
                columns=return_properties)

        deconvoluted_result = pd.DataFrame()
        component = pd.DataFrame()
        component[counts_data] = interactions['{}{}'.format(counts_data, suffix)]
        component[[counts_data, 'protein_name', 'gene_name', 'name', 'is_complex', 'id_cp_interaction', 'id_multidata',
                   'receptor']] = \
            interactions[
                ['{}{}'.format(counts_data, suffix), 'protein_name{}'.format(suffix), 'gene_name{}'.format(suffix),
                 'name{}'.format(suffix), 'is_complex{}'.format(suffix), 'id_cp_interaction',
                 'multidata{}_id'.format(suffix), 'receptor{}'.format(suffix)]]

        deconvolution_complex = pd.merge(complex_compositions,
                                         component,
                                         left_on='complex_multidata_id',
                                         right_on='id_multidata')
        deconvolution_complex = pd.merge(deconvolution_complex,
                                         genes_filtered,
                                         left_on='protein_multidata_id',
                                         right_on='protein_multidata_id',
                                         suffixes=['_complex', '_simple'])

        deconvoluted_result['gene'] = deconvolution_complex['{}_simple'.format(counts_data)]

        deconvoluted_result[
            ['multidata_id', 'protein_name', 'gene_name', 'name', 'is_complex', 'id_cp_interaction', 'receptor',
             'complex_name']] = \
            deconvolution_complex[
                ['complex_multidata_id', 'protein_name_simple', 'gene_name_simple', 'name_simple',
                 'is_complex_complex', 'id_cp_interaction', 'receptor_simple', 'name_complex']]

        return deconvoluted_result


if __name__ == "__main__":
    # counts = pd.read_csv(r'E:\Stereopy\小试答辩\测试\test_counts_mouse.txt', sep='\t', index_col=0)
    # meta = pd.read_csv(r'E:\Stereopy\小试答辩\测试\test_meta.txt', sep='\t')
    # micro_envs = pd.read_csv(r'E:\Stereopy\小试答辩\测试\test_microenviroments.txt', sep='\t')
    from stereo.io.reader import read_ann_h5ad
    data_mousebrain = read_ann_h5ad(r'C:\Users\liuxiaobin\Desktop\Cellbin_deversion.h5ad')
    cell_names = [str(x) for x in data_mousebrain.cell_names]
    mousebrain = data_mousebrain.exp_matrix.log1p()
    mousebrain = pd.DataFrame(mousebrain.T.toarray())
    mousebrain.columns = cell_names
    mousebrain.index = data_mousebrain.gene_names

    meta = pd.read_csv(r'C:\Users\liuxiaobin\Desktop\meta_mousebrain.csv')
    #
    ccc = CellCellCommunication()
    ccc.main('simple', meta, counts=mousebrain, means_filename='means', threads=8,
             significant_means_filename='significant_means', deconvoluted_filename='deconvoluted',
             output_format='csv', species='MOUSE', database='cellphonedb', output_path=r'E:\Stereopy\小试答辩\小试答辩2',
             counts_identifiers='hgnc_symbol')

    # pltt = PlotCellCellCommunication(None)
    # pltt.dot_plot(means_path=r'E:\Stereopy\out\means_statistical.csv',
    #               pvalues_path=r'E:\Stereopy\out\pvalues.csv',
    #               output_path=r'E:\Stereopy\out',
    #               output_name=r'dotplot.pdf',
    #               rows_path=r'E:\Stereopy\小试答辩\测试\rows.txt',
    #               columns_path=r'E:\Stereopy\小试答辩\测试\columns.txt', palette='coolwarm')
    #
    # pltt.heatmap(meta_path=r'E:\Stereopy\CellphoneDB\in\example_data\test_meta.txt',
    #              pvalues_path=r'E:\Stereopy\out\pvalues.csv',
    #              separator_cluster='|',
    #              count_network_path=r'E:\Stereopy\out\network.txt',
    #              pvalue=0.05,
    #              output_path=r'E:\Stereopy\out',
    #              count_name=r'heatmap_count.pdf',
    #              log_count_name=r'heatmap_log.pdf')
