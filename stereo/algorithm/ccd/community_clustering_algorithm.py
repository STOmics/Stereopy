import os
from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from skimage import color

from stereo.log_manager import logger
from .utils import plot_spatial
from .utils import set_figure_params
from .utils import timeit

cluster_palette = ["#1f77b4", "#ff7f0e", "#279e68", "#d62728", "#aa40fc", "#8c564b",
                   "#e377c2", "#b5bd61", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
                   "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#ad494a", "#8c6d31",
                   "#b4d2b1", "#568f8b", "#1d4a60", "#cd7e59", "#ddb247", "#d15252",
                   "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#ef476f",
                   "#ffd166", "#06d6a0", "#118ab2", "#073b4c", "#fbf8cc", "#fde4cf",
                   "#ffcfd2", "#f1c0e8", "#cfbaf0", "#a3c4f3", "#90dbf4", "#8eecf5",
                   '#8359A3', '#5e503f', '#33CC99', '#F2C649', '#B94E48', '#0095B7',
                   '#FF681F', '#e0aaff', '#FED85D', '#0a0908', '#C32148', '#98f5e1',
                   "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
                   "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
                   "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
                   "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
                   "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
                   "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
                   "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
                   "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
                   "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
                   "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
                   "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
                   "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
                   "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
                   "#00B7FF", "#004DFF", "#00FFFF", "#826400", "#580041", "#FF00FF", "#00FF00", "#C500FF",
                   "#B4FFD7", "#FFCA00", "#969600", "#B4A2FF", "#C20078", "#0000C1", "#FF8B00", "#FFC8FF",
                   "#666666", "#FF0000", "#CCCCCC", "#009E8F", "#D7A870", "#8200FF", "#960000", "#BBFF00",
                   "#FFFF00", "#006F00"]


class CommunityClusteringAlgo(ABC):
    """Abstract base class for community detection algorithms."""

    def __init__(self, adata, slice_id, input_file_path, **params):
        """
        Initialize the CommunityClusteringAlgo.

        Parameters:
        - adata (AnnData): Annotated data object which represents the sample to which the algorithm will be applied.
        - slice_id (int): ID of the slice (important when applying the algorithm to more than one slices).
        - input_file_path (str): Path to the input file.
        - **params: Additional parameters, important for a specific algorithm.

        """

        self.slice_id = slice_id
        for key, value in params.items():
            setattr(self, key, value)

        self.dpi = params['dpi']
        set_figure_params(dpi=self.dpi, facecolor='white')
        self.adata = adata
        self.adata.uns['algo_params'] = params
        self.adata.uns['sample_name'] = os.path.basename(input_file_path.rsplit(".", 1)[0])

        self.tissue = None

        cell_count_limit = (self.min_count_per_type * len(self.adata)) // 100
        cell_over_limit = []
        for cell_tp in sorted(self.adata.obs[self.annotation].unique()):
            cell_num = sum(self.adata.obs[self.annotation] == cell_tp)
            if cell_num > cell_count_limit:
                cell_over_limit.append(cell_tp)
            else:
                logger.info(
                    f'{cell_tp} cell type excluded, due to insufficient cells of that type: {cell_num} cells '
                    f'< {int(cell_count_limit)} ({self.min_count_per_type} % of {len(self.adata)})')

        self.adata = self.adata[self.adata.obs[self.annotation].isin(cell_over_limit), :]
        self.unique_cell_type = list(sorted(self.adata.obs[self.annotation].unique()))
        self.annotation_palette = {ct: self.adata.uns[f'{self.annotation}_colors'][i] for i, ct in
                                   enumerate(self.unique_cell_type)}
        self.cluster_palette = {str(i): color for i, color in enumerate(cluster_palette)}
        self.cluster_palette['unknown'] = '#CCCCCC'

    @abstractmethod
    def run(self):
        """Run the algorithm."""
        pass

    @abstractmethod
    def calc_feature_matrix(self):
        """Calculate the feature matrix and generate the tissue object."""
        pass

    @abstractmethod
    def community_calling(self):
        """Perform community calling step."""
        pass

    def get_tissue(self):
        """
        Get the tissue object.

        Returns:
        - tissue (AnnData)

        """
        return self.tissue

    def get_adata(self):
        """
        Get the annotated data object which represents sample.

        Returns:
        - adata (AnnData)

        """
        return self.adata

    def get_community_labels(self):
        """
        Get the community labels of all the cells.

        Returns:
        - community_labels (pandas.Series)

        """
        return self.adata.obs[f'tissue_{self.method_key}']

    def get_cell_mixtures(self):
        """
        Get the cell mixtures table.

        Returns:
        - cell_mixtures (pandas.DataFrame)

        """
        return self.tissue.uns['cell mixtures']

    def set_clustering_labels(self, labels):
        """
        Set the clustering labels.

        Parameters:
        - labels (pandas.Series): The clustering labels.

        """
        self.tissue.obs.loc[:, self.cluster_algo] = labels

    @timeit
    def plot_annotation(self):
        """
        Plot cells using the info of their spatial coordinates and cell type annotations.

        Saves the figure as 'cell_type_annotation.png' in the directory path.

        """

        figure, ax = plt.subplots(figsize=(10, 6))
        plot_spatial(self.adata, annotation=self.annotation, spot_size=self.spot_size, palette=self.annotation_palette,
                     ax=ax, title=f'{self.adata.uns["sample_name"]}')
        legend_ncols = 1 if len(self.unique_cell_type) <= 12 else 2
        plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1), ncol=legend_ncols, prop={"size": 6}, frameon=False)
        plt.tight_layout()
        figure.savefig(os.path.join(self.dir_path, 'cell_type_annotation.png'), dpi=self.dpi, bbox_inches='tight')
        if not self.hide_plots:
            plt.show()
        plt.close()

    def plot_histogram_cell_sum_window(self):
        """
        Plot a histogram of the number of cells in windows.

        This method generates a histogram plot using the window cell sum values
        from the 'obs' column of the tissue object. The resulting plot is saved as
        an image file in the directory specified by 'dir_path'.

        """

        figure, ax = plt.subplots(nrows=1, ncols=1)
        plt.hist(self.tissue.obs['window_cell_sum'].values)
        figure.savefig(os.path.join(self.dir_path,
                                    f'window_cell_num_hist_ws_{"_".join([str(i) for i in self.win_sizes_list])}.png'),
                       dpi=self.dpi, bbox_inches='tight')
        if not self.hide_plots:
            plt.show()
        plt.close()

    def cell_type_filtering(self):
        """
        Perform cell type filtering based on entropy and scatteredness thresholds.

        This function filters the cells in `self.tissue` based on entropy and scatteredness thresholds.
        The filtered cells are stored in `self.tissue` and the raw data is preserved in `self.tissue.raw`.

        """
        # extract binary image of cell positions for each cell type in the slice
        var_use = self.tissue.var.loc[(self.tissue.var['entropy'] <= self.entropy_thres) & (
                self.tissue.var['scatteredness'] <= self.scatter_thres)].index
        self.tissue.raw = self.tissue
        self.tissue = self.tissue[:, var_use]

    @timeit
    def plot_celltype_images(self):
        """
        Plot and save cell type images.

        This method iterates over each cell type in the sample and plots cells of that type.
        The figure is saved as a PNG file in the directory specified by 'dir_path'.

        """

        for cell_t in self.unique_cell_type:
            plt.imsave(fname=os.path.join(self.dir_path, f'tissue_window_{cell_t}_{self.params_suffix}.png'),
                       arr=self.tissue.uns['cell_t_images'][cell_t], vmin=0, vmax=1, cmap='gray', dpi=self.dpi)

    @timeit
    def plot_clustering(self):
        """
        Plot results of the community detection algorithm.

        This function generates a plot using the spatial coordinates of cells,
        where each cell belongs to a certain detected community.
        The plot is saved as an image file.

        """

        # plot initial clustering for each window
        figure, ax = plt.subplots(figsize=(10, 6))
        labels = np.unique(self.adata.obs[f'tissue_{self.method_key}'].values)
        if 'unknown' in labels:
            labels = labels[labels != 'unknown']
        if len(labels) > len(self.cluster_palette):
            logger.warning(f"Number of clusters ({len(labels)}) is larger than pallette size. All clusters will be "
                           f"colored gray.")
            self.cluster_palette = {label: '#CCCCCC' for label in labels}
            self.cluster_palette['unknown'] = '#CCCCCC'
        plot_spatial(self.adata, annotation=f'tissue_{self.method_key}', palette=self.cluster_palette,
                     spot_size=self.spot_size, ax=ax, title=f'{self.adata.uns["sample_name"]}')
        handles, labels = ax.get_legend_handles_labels()
        order = [el[0] for el in
                 sorted(enumerate(labels), key=lambda x: float(x[1]) if x[1] != "unknown" else float('inf'))]
        legend_ncols = 1 if len(labels) <= 12 else 2
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left',
                   bbox_to_anchor=(1.04, 1), ncol=legend_ncols, prop={"size": 6}, frameon=False)
        plt.tight_layout()
        figure.savefig(os.path.join(self.dir_path, f'clusters_cellspots_{self.params_suffix}.png'), dpi=self.dpi,
                       bbox_inches='tight')
        if not self.hide_plots:
            plt.show()
        plt.close()

    def calculate_spatial_cell_type_metrics(self):
        """Calculate spatial cell type metrics."""
        pass

    @timeit
    def calculate_cell_mixture_stats(self):
        """
        Calculate cell type percentages per cluster - community and save it in pandas.DataFrame object.

        Percentages are calculated globaly for all cells with single class label.
        This is saved in self.tissue.uns['cell mixtures'] for further use by plot fn.
        Columns of total cell count per class and percentage of tissue per cluster are added.
        Row of total cell type count is added. DataFrame with additional columns and row is saved in adata.uns['cell mixture stats']  # noqa
        """

        # extract information on self.cluster_algo clustering labels and cell types to create cell communities statistics # noqa
        clustering_labels = f'tissue_{self.method_key}'
        cell_types_communities = self.adata.obs[[clustering_labels, self.annotation]]
        # remove cells with unknown cell community label
        if 'unknown' in cell_types_communities[clustering_labels].cat.categories:
            cell_types_communities = cell_types_communities[cell_types_communities[clustering_labels] != 'unknown']
            cell_types_communities[clustering_labels] = cell_types_communities[clustering_labels].cat.remove_categories(
                'unknown')

        stats_table = {}
        # calculate cell type mixtures for every cluster
        for label, cluster_data in cell_types_communities.groupby(clustering_labels):
            # cell_type_dict = {ct: 0 for ct in self.unique_cell_type}
            # for cell in cluster_data[self.annotation]:
            #     cell_type_dict[cell] += 1
            cell_type_dict = {ct: np.sum(cluster_data[self.annotation] == ct) for ct in self.unique_cell_type}

            # remove excluded cell types
            cell_type_dict = {k: cell_type_dict[k] for k in self.tissue.var.index.sort_values()}

            # create a dictionary of cluster cell type distributions
            stats_table[label] = {k: cell_type_dict[k] for k in cell_type_dict}

        stats = pd.DataFrame(stats_table).T
        stats.columns.name = "cell types"

        stats.index = stats.index.astype(int)
        stats = stats.sort_index()
        stats.index = stats.index.astype(str)

        # [TODO] Condsider doing this in some other place
        # if there are cell types with 0 cells in every cluster remove them
        for col in stats.columns:
            if sum(stats.loc[:, col]) == 0:
                stats = stats.drop(labels=col, axis=1)
        # if there are clusters with 0 cells remove them
        for row in stats.index:
            if sum(stats.loc[row, :]) == 0:
                stats = stats.drop(labels=row, axis=0)

        # save absolute cell mixtures to tissue
        self.tissue.uns['cell mixtures'] = stats.iloc[:, :].copy()

        # add column with total cell count per cluster
        stats['total_counts'] = np.array([sum(stats.loc[row, :]) for row in stats.index]).astype(int)

        # add row with total counts per cell types
        cell_type_counts = {ct: [int(sum(stats[ct]))] for ct in stats.columns}
        stats = pd.concat([stats, pd.DataFrame(cell_type_counts, index=['total_cells'])])

        # divide each row with total sum of cells per cluster and mul by 100 to get percentages
        stats.iloc[:-1, :-1] = stats.iloc[:-1, :-1].div(stats['total_counts'][:-1], axis=0).mul(100).astype(int)

        # add column with percentage of all cells belonging to a cluster
        stats['perc_of_all_cells'] = np.around(stats['total_counts'] / stats['total_counts'][-1] * 100, decimals=1)

        # save cell mixture statistics to tissue
        self.tissue.uns['cell mixtures stats'] = stats.iloc[:, :]

    @timeit
    def plot_stats(self):
        """
        Plot cell mixture statistics as a heatmap.

        The heatmap is saved as a PNG file in the directory specified by `self.dir_path`.

        """
        stats = self.tissue.uns['cell mixtures stats']
        set_figure_params(dpi=self.dpi, facecolor='white')
        sns.set(font_scale=1.5)

        ncols = len(stats.columns)  # we want to separately print the total_counts column
        fig, axes = plt.subplots(ncols=ncols, figsize=(30, 20))

        # no space between columns
        fig.subplots_adjust(wspace=0)

        # put colormaps of your choice in a list:
        vmax_perc = np.max(np.ravel(stats.iloc[:-1, :-2]))
        for i, ax in enumerate(axes[:-2]):
            sns.heatmap(pd.DataFrame(stats.iloc[:, i]), vmin=0.0, vmax=vmax_perc, linewidths=0, linecolor=None,
                        annot=True, cbar=False, ax=ax, cmap="Greys", fmt='4.0f', xticklabels=True,
                        yticklabels=True if i == 0 else False, square=True)
        # total_counts column - sum of all cells per cluster
        sns.heatmap(pd.DataFrame(stats.iloc[:, -2]), annot=True, vmin=0, vmax=np.max(stats.iloc[:-1, -2]), linewidths=0,
                    linecolor=None, cbar=False, cmap='Greens', ax=axes[-2], fmt='4.0f', xticklabels=True,
                    yticklabels=False, square=True)
        # perf_of_all_cells column - perc of all tissue cells in each cluster
        sns.heatmap(pd.DataFrame(stats.iloc[:, -1]), annot=True, vmin=0, vmax=np.max(stats.iloc[:-1, -1]), linewidths=0,
                    linecolor=None, cbar=False, cmap='Greens', ax=axes[-1], fmt='4.0f', xticklabels=True,
                    yticklabels=False, square=True)

        # x axis on top
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
            ax.xaxis.tick_top()
        plt.savefig(os.path.join(self.dir_path, f'cell_mixture_table_{self.params_suffix}.png'), bbox_inches='tight',
                    dpi=self.dpi)
        if not self.hide_plots:
            plt.show()
        plt.close()

    @timeit
    def plot_cluster_mixtures(self, cluster_index=None):
        """
        Plot cell mixtures for each cluster (community). Only cell types which have more than min_perc_to_show abundance will be shown. # noqa

        The cell mixtures are obtained from `self.tissue.uns['cell mixtures stats']`.
        The resulting plots are saved as PNG files in the directory specified by `self.dir_path`.

        """
        # plot each cluster and its cells mixture
        set_figure_params(dpi=self.dpi, facecolor='white')
        stats = self.tissue.uns['cell mixtures stats']

        new_stats = stats.copy()
        new_stats = new_stats.drop(labels=['total_counts', 'perc_of_all_cells'], axis=1)
        new_stats = new_stats.drop(labels='total_cells', axis=0)

        cl_palette = {}
        for cluster in new_stats.index:
            cl_palette[cluster] = '#dcdcdc'
        cl_palette['unknown'] = '#dcdcdc'

        ind = 0
        for cluster in new_stats.iterrows():
            if cluster_index != None and cluster_index != ind:  # noqa
                ind += 1
                continue
            elif cluster_index != None and cluster_index == ind:  # noqa
                ind += 1
            # only display clusters with more than min_cells_in_cluster cells
            if stats.loc[cluster[0]]['total_counts'] > self.min_cluster_size:
                # sort cell types by their abundnce in the cluster
                ct_perc = cluster[1].sort_values(ascending=False)
                # only cell types which have more than min_perc_to_show abundance will be shown
                ct_show = ct_perc.index[ct_perc > self.min_perc_to_show]
                ct_palette = {x: self.annotation_palette[x] for x in ct_show}
                for y in self.annotation_palette.keys():
                    if y not in ct_show:
                        ct_palette[y] = '#dcdcdc'

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
                fig.subplots_adjust(wspace=0.35)

                plot_spatial(self.adata, annotation=self.annotation, palette=ct_palette, spot_size=self.spot_size,
                             ax=ax[0])
                ax[0].set_title('Cell types')
                handles, labels = ax[0].get_legend_handles_labels()
                handles, labels = zip(*filter(lambda hl: hl[1] in ct_show, zip(handles, labels)))
                labels = [f'{ctype} ({ct_perc[ctype]}%)' for ctype in labels]
                ax[0].legend(handles=handles, labels=labels, bbox_to_anchor=(1.0, 0.5), loc='center left',
                             frameon=False, fontsize=8)
                cl_palette[cluster[0]] = cluster_palette[int(cluster[0])]

                plot_spatial(self.adata, annotation=f'tissue_{self.method_key}', palette=cl_palette,
                             spot_size=self.spot_size, ax=ax[1])
                ax[1].set_title(f'Cell community {cluster[0]} ({self.adata.uns["sample_name"]})')
                ax[1].get_legend().remove()
                fig.savefig(os.path.join(self.dir_path, f'cmixtures_{self.params_suffix}_c{cluster[0]}.png'),
                            bbox_inches='tight')
                if not self.hide_plots:
                    plt.show()
                plt.close()
                cl_palette[cluster[0]] = '#dcdcdc'
        return plt.figure()

    @timeit
    def boxplot_stats(self, cluster_index=None, stripplot=False):
        """
        Generate a box plot of cell type percentages distribution per cluster.

        Args:
            cluster_index (int, optional):
            stripplot (bool, optional): Whether to overlay a stripplot of specific percentage values.

        """

        # box plot per cluster of cell type percentages distribution
        set_figure_params(dpi=self.dpi, facecolor='white')

        cluster_list = np.unique(self.tissue.obs[self.cluster_algo])
        cluster_list = np.sort(cluster_list.astype(np.int32)).astype(str)

        ind = 0
        for cluster in cluster_list:
            if cluster_index != None and cluster_index != ind:  # noqa
                ind += 1
                continue
            elif cluster_index != None and cluster_index == ind:  # noqa
                ind += 1
            # for each window size a box plot is provided per cluster
            cl_win_cell_distrib = self.tissue[self.tissue.obs[self.cluster_algo] == cluster]
            for window_size, sliding_step in zip(self.win_sizes_list, self.sliding_steps_list):
                # extract only windows of specific size
                win_cell_distrib = cl_win_cell_distrib[cl_win_cell_distrib.obsm['spatial'][:, 3] == window_size]
                # check if subset anndata object is empty
                if win_cell_distrib.X.size > 0:
                    # a DataFrame with cell percentages instead of normalized cel number is created
                    win_cell_distrib_df = pd.DataFrame(win_cell_distrib.X / (self.total_cell_norm / 100),
                                                       columns=win_cell_distrib.var.index)

                    # Reshape data into long format
                    cell_type_distrib = pd.melt(win_cell_distrib_df, var_name='Cell Type', value_name='Percentage')
                    cell_type_distrib = cell_type_distrib.sort_values(by='Cell Type')

                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
                    # plot boxplot of cell type percentages per mixture
                    ax = sns.boxplot(x='Cell Type', y='Percentage', data=cell_type_distrib,
                                     palette=self.annotation_palette)
                    if stripplot:
                        # overlap with a plot of specific percentage values.
                        # Jitter allows dots to move left and right for better visibility of all points
                        ax = sns.stripplot(x='Cell Type', y='Percentage', data=cell_type_distrib, jitter=True,
                                           color='black', size=2)
                    # remove top and right frame of the plot
                    sns.despine(top=True, right=True)
                    ax.set_title(f'Cell community {cluster} win size {window_size} step {sliding_step} '
                                 f'({self.adata.uns["sample_name"]})')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    ax.xaxis.tick_bottom()  # x axis on the bottom
                    fig.savefig(os.path.join(self.dir_path, f'boxplot_c{cluster}_ws{window_size}.png'),
                                bbox_inches='tight')
                    if not self.hide_plots:
                        plt.show()
                    plt.close()
        return plt.figure()

    @timeit
    def colorplot_stats(self, color_system='rgb', cluster_index=None):
        """
        For each cluster (community) plot percentage of cell types. Plotting is done on windows level.

        Parameters:
        - color_system (str, optional): Color system to use. Supported values are 'hsv' and 'rgb'.

        """

        supported_color_systems = ['hsv', 'rgb']
        if color_system in supported_color_systems:
            stats = self.tissue.uns['cell mixtures'].copy()
            # extract total counts per cluster
            total_counts = np.sum(stats, axis=1)
            # divide each row with total sum of cells per cluster and mul by 100 to get percentages
            stats.iloc[:, :] = stats.iloc[:, :].div(np.array([sum(stats.loc[row, :]) for row in stats.index]),
                                                    axis=0).mul(100).astype(int)

            cx_min = int(np.min(self.adata.obsm['spatial'][:, 0]))
            cy_min = int(np.min(self.adata.obsm['spatial'][:, 1]))
            cx_max = int(np.max(self.adata.obsm['spatial'][:, 0]))
            cy_max = int(np.max(self.adata.obsm['spatial'][:, 1]))

            ind = 0
            for cluster in stats.iterrows():
                if cluster_index != None and cluster_index != ind:  # noqa
                    ind += 1
                    continue
                elif cluster_index != None and cluster_index == ind:  # noqa
                    ind += 1
                # only display clusters with more than min_cells_in_cluster cells
                if total_counts[cluster[0]] > self.min_cluster_size:
                    ct_perc = cluster[1].sort_values(ascending=False)
                    top_three_ct = ct_perc.index.values[0:3]

                    cl_win_cell_distrib = self.tissue[self.tissue.obs[self.cluster_algo] == cluster[0]]
                    cl_win_cell_distrib = cl_win_cell_distrib[:, top_three_ct]

                    # for each pair of window size and sliding step a separate color plot should be made
                    for window_size, sliding_step in zip(self.win_sizes_list, self.sliding_steps_list):
                        # extract data for a specfic window size
                        win_cell_distrib = cl_win_cell_distrib[cl_win_cell_distrib.obsm['spatial'][:, 3] == window_size]
                        # data is a DataFrame with rows for each window, columns of 3 top most cell types for
                        # current cell mixture cluster, with data on cell type in percentages [0.00-1.00]
                        # The last is achieved by dividing the features with self.total_cell_norm
                        data_df = pd.DataFrame(win_cell_distrib.X / self.total_cell_norm,
                                               columns=win_cell_distrib.var.index, index=win_cell_distrib.obs.index)
                        # init image
                        mixture_image = np.zeros(shape=(cy_max - cy_min + 1, cx_max - cx_min + 1, 3), dtype=np.float32)

                        for window in data_df.iterrows():
                            wx = int(window[0].split("_")[0])
                            wy = int(window[0].split("_")[1])
                            mixture_image[
                            int(wy * sliding_step - cy_min): int(wy * sliding_step + window_size - cy_min),  # noqa
                            int(wx * sliding_step - cx_min): int(wx * sliding_step + window_size - cx_min),  # noqa
                            :] = 1 - window[1].values.astype(np.float32)  # noqa

                        # convert image of selected color representation to rgb
                        if color_system == 'hsv':
                            # if hsv display the 1 - percentage since the colors will be too dark
                            rgb_image = color.hsv2rgb(mixture_image)
                        elif color_system == 'rgb':
                            rgb_image = mixture_image
                        # plot the colored window image of the cell scatterplot
                        fig, ax = plt.subplots(nrows=1, ncols=1)
                        # cell scatterplot for visual spatial reference
                        plt.scatter(x=self.adata.obsm['spatial'][:, 0] - cx_min,
                                    y=self.adata.obsm['spatial'][:, 1] - cy_min, c='#CCCCCC', marker='.', s=0.5,
                                    zorder=1)
                        # mask of window positions
                        window_mask = rgb_image[:, :, 1] != 0
                        # mask adjusted to alpha channel and added to rgb image
                        window_alpha = (window_mask == True).astype(int)[..., np.newaxis]  # noqa
                        rgba_image = np.concatenate([rgb_image, window_alpha], axis=2)
                        # plot windows, where empty areas will have alpha=0, making them transparent
                        plt.imshow(rgba_image, zorder=2)
                        plt.axis('off')
                        plt.gca().invert_yaxis()
                        ax.grid(visible=False)
                        ax.set_title(
                            f'{color_system.upper()} of community {cluster[0]} win size {window_size}, step'
                            f' {sliding_step} - top 3 cell types\n({self.adata.uns["sample_name"]})')

                        if color_system == 'hsv':
                            plane_names = ['H\'', 'S\'', 'V\'']
                        elif color_system == 'rgb':
                            plane_names = ['R\'', 'G\'', 'B\'']

                        ax.text(1.05, 0.5, f'{plane_names[0]} - {top_three_ct[0]} '
                                           f'({ct_perc[top_three_ct[0]]}%)\n{plane_names[1]} - '
                                           f'{top_three_ct[1]} ({ct_perc[top_three_ct[1]]}%)\n{plane_names[2]} - '
                                           f'{top_three_ct[2]} ({ct_perc[top_three_ct[2]]}%)', transform=ax.transAxes,
                                fontsize=12, va='center', ha='left')
                        fig.savefig(os.path.join(self.dir_path, f'colorplot_{color_system}_c{cluster[0]}_ws'
                                                                f'{window_size}_ss{sliding_step}.png'),
                                    bbox_inches='tight', dpi=self.dpi)
                        if not self.hide_plots:
                            plt.show()
                        plt.close()
            return plt.figure()
        else:
            logger.warning(f'Unsupported color system: {color_system}.')

    @timeit
    def colorplot_stats_per_cell_types(self):
        """
        For each cell type plot its percentage in each window as the red channel of RGB.
        Green and blue channels values are 128.

        """

        cx_min = int(np.min(self.adata.obsm['spatial'][:, 0]))
        cy_min = int(np.min(self.adata.obsm['spatial'][:, 1]))
        cx_max = int(np.max(self.adata.obsm['spatial'][:, 0]))
        cy_max = int(np.max(self.adata.obsm['spatial'][:, 1]))

        for window_size, sliding_step in zip(self.win_sizes_list, self.sliding_steps_list):
            windows_mixture = self.tissue[self.tissue.obsm['spatial'][:, 3] == window_size]
            data_df = pd.DataFrame(windows_mixture.X / self.total_cell_norm, columns=windows_mixture.var.index,
                                   index=windows_mixture.obs.index)
            rgb_image = np.zeros(shape=(cy_max - cy_min + 1, cx_max - cx_min + 1, 3), dtype=np.float32)

            for cell_type in self.tissue.var.index:
                for window in data_df.iterrows():
                    wx = int(window[0].split("_")[0])
                    wy = int(window[0].split("_")[1])
                    rgb_image[int(wy * sliding_step - cy_min): int(wy * sliding_step + window_size - cy_min),
                    int(wx * sliding_step - cx_min): int(wx * sliding_step + window_size - cx_min), :] = [  # noqa
                        window[1][cell_type], 0.5, 0.5]

                fig, ax = plt.subplots(nrows=1, ncols=1)
                plt.scatter(x=self.adata.obsm['spatial'][:, 0] - cx_min, y=self.adata.obsm['spatial'][:, 1] - cy_min,
                            c='#CCCCCC', marker='.', s=0.5, zorder=1)

                window_mask = rgb_image[:, :, 0] != 0
                window_alpha = (window_mask == True).astype(int)[..., np.newaxis]  # noqa
                rgba_image = np.concatenate([rgb_image, window_alpha], axis=2)

                plt.imshow(rgba_image, zorder=2)
                plt.axis('off')
                plt.gca().invert_yaxis()
                ax.grid(visible=False)
                ax.set_title(
                    f'Percentage of {cell_type} (red channel of RGB) in:\n{self.adata.uns["sample_name"]}, win size'
                    f' {window_size}, step {sliding_step}')
                fig.savefig(
                    os.path.join(self.dir_path, f'ct_colorplot_rgb_{cell_type}_ws{window_size}_ss{sliding_step}.png'),
                    bbox_inches='tight', dpi=self.dpi)
                if not self.hide_plots:
                    plt.show()
                plt.close()

    @timeit
    def plot_celltype_table(self):
        """Plot a table showing cell type abundance per cluster."""

        sns.set(font_scale=1)
        set_figure_params(dpi=self.dpi, facecolor='white')

        stats = self.tissue.uns['cell mixtures'].copy()

        # calculate percentage of all cells of one cell type belonging to each cluster
        ct_perc_per_celltype = stats.iloc[:, :].div(np.array([sum(stats.loc[:, col]) for col in stats.columns]),  # noqa
                                                    axis=1).mul(100).astype(int)
        ct_perc_per_cluster = stats.iloc[:, :].div(np.array([sum(stats.loc[row, :]) for row in stats.index]),
                                                   axis=0).mul(100).astype(int)

        # divide cell numbers with total number of cells per cluster to obtain ct perc per cluster
        for cluster in stats.iterrows():
            ct_perc_per_cluster_sorted = ct_perc_per_cluster.loc[cluster[0], :].sort_values(ascending=False)
            if (ct_perc_per_cluster_sorted[self.min_num_celltype - 1] < self.min_perc_celltype) or (
                    sum(stats.loc[cluster[0], :]) < self.min_cluster_size):
                # remove all clusters that have low number of cells, or high abundance of single cell type
                stats = stats.drop(labels=cluster[0], axis=0)
                ct_perc_per_celltype = ct_perc_per_celltype.drop(labels=cluster[0], axis=0)
                ct_perc_per_cluster = ct_perc_per_cluster.drop(labels=cluster[0], axis=0)

        # remove cell types that are not a significant part of any heterogeneous cluster
        for celltype in stats.columns:
            if (max(ct_perc_per_celltype.loc[:, celltype]) < self.min_perc_to_show):  # noqa
                stats = stats.drop(labels=celltype, axis=1)
                ct_perc_per_celltype = ct_perc_per_celltype.drop(labels=celltype, axis=1)
                ct_perc_per_cluster = ct_perc_per_cluster.drop(labels=celltype, axis=1)

        ncols = len(stats)
        # table will have a clumn for each cluster and first column for cell types
        fig, axes = plt.subplots(nrows=1, ncols=ncols + 1, figsize=(30, 25))
        # no space between columns
        fig.subplots_adjust(wspace=0, hspace=0)

        # create a dictionary mapping each cluster to its corresponding color
        cluster_color = dict(zip(stats.index, [cluster_palette[int(x)] for x in stats.index]))

        # cell type colors from adata.uns['annotation_colors'] if exists
        row_cmap = ["#FFFFFF"] + list(self.annotation_palette.values())
        # inner area of the table is of white background
        column_cmap = ["#FFFFFF" for _ in range(stats.shape[1])]

        for i, ax in enumerate(axes):
            if i == 0:
                g = sns.heatmap(np.array(range(len(stats.columns) + 1))[:, np.newaxis], linewidths=0.5,
                                linecolor='gray', square=None,
                                annot=np.array([''] + [column for column in stats.columns])[:, np.newaxis],
                                ax=ax, cbar=False, cmap=row_cmap, fmt="", xticklabels=False, yticklabels=False,
                                )
            else:
                table_annotation = np.array([f'cluster {stats.index[i - 1]}'] + [
                    f'{ct_perc_per_cluster.iloc[i - 1, int(x)]}%\n({ct_perc_per_celltype.iloc[i - 1, int(x)]}%)' for x
                    in range(len(stats.columns))])[:, np.newaxis]
                column_cmap[0] = cluster_color[stats.index[i - 1]]
                g = sns.heatmap(np.array(range(stats.shape[1] + 1))[:, np.newaxis], linewidths=0.5,  # noqa
                                linecolor='gray', annot=table_annotation, cbar=False, cmap=column_cmap,
                                ax=ax, fmt='', xticklabels=False,
                                yticklabels=False, square=None)
        axes[i // 2].set_title('Cell type abundance per cluster (and per cel type set)')
        axes[i // 2].title.set_size(20)
        fig.savefig(os.path.join(self.dir_path, f'celltype_table_{self.params_suffix}.png'), bbox_inches='tight')
        if not self.hide_plots:
            plt.show()
        plt.close()
        return plt.figure()

    def save_metrics(self):
        """Save metrics results in CSV format."""

        self.tissue.var[['entropy', 'scatteredness']].to_csv(
            os.path.join(self.dir_path, f'spatial_metrics_{self.params_suffix}.csv'))

    def save_tissue(self, suffix=''):
        """
        Save self.tissue anndata object.

        Parameters:
        - suffix (str): A suffix to add to the filename (default: '').

        """

        self.tissue.write_h5ad(os.path.join(self.dir_path, f'tissue_{self.filename}{suffix}.h5ad'), compression="gzip")

        logger.info(f'Saved clustering result tissue_{self.filename}{suffix}.h5ad.')

    def save_anndata(self, suffix=''):
        """
        Save adata object.

        Parameters:
        - suffix (str): A suffix to add to the filename (default: '').

        """
        # save anndata file
        self.adata.write_h5ad(os.path.join(self.dir_path, f'{self.filename}{suffix}.h5ad'), compression="gzip")

        logger.info(f'Saved clustering result as a part of original anndata file {self.filename}{suffix}.h5ad.')

    def save_community_labels(self):
        """Save community labels from anndata file."""

        self.adata.obs[f'tissue_{self.method_key}'].to_csv(
            os.path.join(self.dir_path, f'{self.filename}_ccdlabels.csv'))

        logger.info(
            f'Saved community labels after clustering as a part of original anndata file to {self.filename}.csv')

    def save_mixture_stats(self):
        """Save cell mixture statistics, which contains number of cells of specific types per community."""

        self.tissue.uns['cell mixtures'].to_csv(
            os.path.join(self.dir_path, f'cell_mixture_stats_{self.params_suffix}.csv'))
