import os
from functools import reduce
from itertools import cycle

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .ms_plot_base import MSDataPlotBase
from .plot_base import PlotBase
from ..algorithm.ccd import (
    timeit,
    set_figure_params,
    plot_spatial
)

cluster_palette = ["#1f77b4", "#ff7f0e", "#279e68", "#d62728", "#aa40fc", "#8c564b",
                   "#e377c2", "#b5bd61", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
                   "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#ad494a", "#8c6d31",
                   "#b4d2b1", "#568f8b", "#1d4a60", "#cd7e59", "#ddb247", "#d15252",
                   "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#ef476f",
                   "#ffd166", "#06d6a0", "#118ab2", "#073b4c", "#fbf8cc", "#fde4cf",
                   "#ffcfd2", "#f1c0e8", "#cfbaf0", "#a3c4f3", "#90dbf4", "#8eecf5",
                   '#FFFF99', '#FF7034', '#01A638', '#7A89B8', '#FE4C40', '#6B3FA0',
                   '#0066CC', '#C62D42', '#B5B35C', '#93DFB8', '#f2f4f3', '#10002b',
                   '#22333b', '#5FA777', '#766EC8', '#1AB385', '#AFE313', '#008080',
                   '#6456B7', '#A9B2C3', '#8FD8D8', '#2D383A', '#009DC4', '#93CCEA',
                   '#FF9980', '#2887C8', '#7BA05B', '#02A4D3', '#FE6F5E', '#4F69C6',
                   '#9d4edd', '#003366', '#ECEBBD', '#a9927d', '#00CC99', '#9DE093',
                   '#7b2cbf', '#b9fbc0', '#01796F', '#00CCCC', '#652DC1', '#5a189a',
                   '#63B76C', '#76D7EA', '#9999CC', '#FF5349', '#6CDAE7', '#00755E',
                   '#FBE870', '#3c096c', '#0066FF', '#FD0E35', '#c77dff', '#F1E788',
                   '#FFAE42', '#95E0E8', '#1560BD', '#C5E17A', '#8D90A1', '#FF8833',
                   '#FFFF9F', '#3C69E7', '#C3CDE6', '#F8D568', '#FCD667', '#29AB87',
                   '#8359A3', '#5e503f', '#FBE7B2', '#FFB97B', '#33CC99', '#F2C649',
                   '#B94E48', '#0095B7', '#E77200', '#FF681F', '#e0aaff', '#FED85D',
                   '#240046', '#0a0908', '#C32148', '#98f5e1']


class PlotCCDSingle(PlotBase):

    @timeit
    def plot_celltype_table(self, **param):
        """Plot a table showing cell type abundance per cluster."""

        set_figure_params(dpi=param.get("dpi"), facecolor='white')
        sns.set(font_scale=1)

        stats = self.stereo_exp_data._ann_data.uns['cell mixtures'].copy()

        # calculate percentage of all cells of one cell type belonging to each cluster
        ct_perc_per_celltype = stats.iloc[:, :].div(np.array([sum(stats.loc[:, col]) for col in stats.columns]),
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
            if (max(ct_perc_per_celltype.loc[:, celltype]) < self.min_perc_to_show):
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
                g = sns.heatmap(
                    np.array(range(len(stats.columns) + 1))[:, np.newaxis],
                    linewidths=0.5,
                    linecolor='gray',
                    annot=np.array([''] + [column for column in stats.columns])[:, np.newaxis],
                    ax=ax,
                    cbar=False,
                    cmap=row_cmap,
                    fmt="",
                    xticklabels=False,
                    yticklabels=False,
                    square=None
                )
            else:
                table_annotation = np.array([f'cluster {stats.index[i - 1]}'] + [
                    f'{ct_perc_per_cluster.iloc[i - 1, int(x)]}%\n({ct_perc_per_celltype.iloc[i - 1, int(x)]}%)'
                    for x in range(len(stats.columns))
                ])[:, np.newaxis]
                column_cmap[0] = cluster_color[stats.index[i - 1]]
                g = sns.heatmap(  # noqa
                    np.array(range(stats.shape[1] + 1))[:, np.newaxis],
                    linewidths=0.5,
                    linecolor='gray',
                    annot=table_annotation,
                    cbar=False,
                    cmap=column_cmap,
                    ax=ax,
                    fmt='',
                    xticklabels=False,
                    yticklabels=False,
                    square=None
                )
        axes[i // 2].set_title('Cell type abundance per cluster (and per cel type set)')
        axes[i // 2].title.set_size(20)
        fig.savefig(os.path.join(self.dir_path, f'celltype_table_{self.params_suffix}.png'), bbox_inches='tight')
        if not self.hide_plots:
            plt.show()
        plt.close()


class PlotCCD(MSDataPlotBase):

    def plot_all_slices(self, clustering=False, img_name='cell_type_per_slice.png'):
        """
        Plot all slices using the specified algorithms and annotations.

        Parameters:
        - img_name (str): The name of the output image file.
        - clustering (bool, optional): Whether to plot clustering or cell type annotation. Defaults to False.

        """
        number_of_samples = len(self.pipeline_res['ccd']["algo_list"])
        number_of_rows = 2 if number_of_samples % 2 == 0 and number_of_samples > 2 else 1
        number_of_columns = (number_of_samples // 2) if number_of_samples % 2 == 0 and number_of_samples > 2 \
            else number_of_samples

        figure, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, squeeze=False, layout='constrained',
                                    figsize=(10, 6))
        h_d = {}
        unknown_label = []
        for (algo, ax) in zip(self.pipeline_res['ccd']["algo_list"], axes.flatten()):
            labels = np.unique(algo.adata.obs[f'tissue_{algo.method_key}'].values)
            if 'unknown' in labels:
                labels = labels[labels != 'unknown']
            algo.cluster_palette = {lab: cluster_palette[int(lab)] for lab in labels}
            algo.cluster_palette['unknown'] = '#CCCCCC'

            palette = algo.cluster_palette if clustering else algo.annotation_palette
            annotation = f'tissue_{self.pipeline_res["ccd"]["algo_list"][0].method_key}' if clustering else \
                self.pipeline_res['ccd']["algo_list"][0].annotation
            plot_spatial(algo.adata, annotation=annotation, palette=palette, spot_size=algo.spot_size, ax=ax)
            ax.get_legend().remove()
            ax.set_title(f'{algo.filename}', fontsize=6, loc='center', wrap=True)
            hands, labs = ax.get_legend_handles_labels()
            for h, l in zip(hands, labs):
                h._sizes = [11]
                if l == 'unknown':
                    unknown_label = np.array([[h, l]])
                    continue
                if l not in h_d.values():
                    h_d[h] = l
        try:
            handles = np.array([[h, int(l)] for (h, l) in h_d.items()])
        except Exception:
            handles = np.array([[h, l] for (h, l) in h_d.items()])

        handles = handles[handles[:, 1].argsort()]
        handles[:, 1] = handles[:, 1].astype('str')

        if len(unknown_label) > 0:
            handles = np.concatenate((handles, unknown_label), axis=0)

        legend_ncols = 1 if len(handles) <= 12 else 2
        figure.legend(handles[:, 0], handles[:, 1], bbox_to_anchor=(1.15, 0.5), loc='center', fontsize=4, frameon=False,
                      borderaxespad=0., ncol=legend_ncols, labelspacing=1, scatterpoints=10)
        return plt.figure()

    def plot_celltype_mixtures_total(self, cell_mixtures, **params):
        """
        Plot the total cell type mixtures.

        Parameters:
        - cell_mixtures (list): A list of dictionaries containing cell type mixtures.

        """

        def merge_dicts(dict1, dict2):
            return {cluster: dict1.get(cluster, 0) + dict2.get(cluster, 0) for cluster in set(dict1) | set(dict2)}

        def merge_dicts_of_dicts(dict1, dict2):
            return {celltype: merge_dicts(dict1.get(celltype, {}), dict2.get(celltype, {})) for celltype in
                    set(dict1) | set(dict2)}

        total_dict = reduce(merge_dicts_of_dicts, cell_mixtures)
        total = pd.DataFrame(total_dict).fillna(0)

        total['total_counts'] = np.array([sum(total.loc[row, :]) for row in total.index]).astype(int)

        cell_type_counts = {ct: [int(sum(total[ct]))] for ct in total.columns}
        total = pd.concat([total, pd.DataFrame(cell_type_counts, index=['total_cells'])])

        total.iloc[:-1, :-1] = total.iloc[:-1, :-1].div(total['total_counts'][:-1], axis=0).mul(100)
        total['perc_of_all_cells'] = np.around(total['total_counts'] / total['total_counts'][-1] * 100, decimals=1)
        total = total.loc[sorted(total.index.values, key=lambda x: float(x) if x != "total_cells" else float('inf'))]

        set_figure_params(dpi=params.get('dpi'), facecolor='white')
        sns.set(font_scale=1.5)

        ncols = len(total.columns)
        fig, axes = plt.subplots(ncols=ncols, figsize=(30, 20))
        fig.subplots_adjust(wspace=0)

        vmax_perc = np.max(np.ravel(total.iloc[:-1, :-2]))
        for i, ax in enumerate(axes[:-2]):
            sns.heatmap(pd.DataFrame(total.iloc[:, i]), vmin=0.0, vmax=vmax_perc, linewidths=0, linecolor=None,
                        annot=True, cbar=False, ax=ax, cmap="Greys", fmt='4.0f', xticklabels=True,
                        yticklabels=True if i == 0 else False, square=True)
        sns.heatmap(pd.DataFrame(total.iloc[:, -2]), annot=True, vmin=0, vmax=np.max(total.iloc[:-1, -2]), linewidths=0,
                    linecolor=None, cbar=False, cmap='Greens', ax=axes[-2], fmt='4.0f',
                    xticklabels=True, yticklabels=False, square=True)
        sns.heatmap(pd.DataFrame(total.iloc[:, -1]), annot=True, vmin=0, vmax=np.max(total.iloc[:-1, -1]), linewidths=0,
                    linecolor=None, cbar=False, cmap='Greens', ax=axes[-1], fmt='4.0f', xticklabels=True,
                    yticklabels=False, square=True)

        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
            ax.xaxis.tick_top()

        return plt.figure()

    def plot_cell_perc_in_community_per_slice(self, **params):
        """
        Plots the percentage of cells in each community per slice.
        """
        cells_in_comm_per_slice = {
            algo.filename: algo.get_community_labels().value_counts(normalize=True).rename(algo.filename)
            for algo in self.pipeline_res['ccd']["algo_list"]
        }
        df = pd.concat(cells_in_comm_per_slice.values(), axis=1).fillna(0).mul(100).T
        df = df[sorted(df.columns.values, key=lambda x: float(x) if x != "unknown" else float('inf'))]
        set_figure_params(dpi=params.get('dpi'), facecolor='white')
        sns.set(font_scale=1.5)
        plt.figure(figsize=(30, 20))

        ax = sns.heatmap(df, annot=True, fmt="4.0f", cmap="Greys", xticklabels=True, yticklabels=True, square=True,
                         cbar=False)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        return plt.figure()

    def plot_cell_abundance_total(self, **params):
        """
        Plots the total cell abundance for each algorithm.
        """
        fig, ax = plt.subplots(figsize=(20, 20))
        fig.subplots_adjust(wspace=0)
        set_figure_params(dpi=params.get('dpi'), facecolor='white')

        greys = cycle(['darkgray', 'gray', 'dimgray', 'lightgray'])
        colors = [next(greys) for _ in range(len(self.pipeline_res['ccd']["algo_list"]))]
        cell_percentage_dfs = []
        plot_columns = []
        for algo in self.pipeline_res['ccd']["algo_list"]:
            cell_percentage_dfs.append(pd.DataFrame(
                algo.get_adata().obs[algo.annotation].value_counts(normalize=True).mul(100).rename(algo.filename)))
            plot_columns.append(algo.filename)

        cummulative_df = pd.concat(cell_percentage_dfs, axis=1).fillna(0)
        cummulative_df.plot(y=plot_columns, kind="bar", rot=70, ax=ax, xlabel="", color=colors)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.grid(False)
        ax.set_facecolor('white')
        plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

        return plt.figure()

    def plot_cell_abundance_per_slice(self, **params):
        """
        Plots the cell abundance for each algorithm per slice.
        """
        number_of_samples = len(self.pipeline_res['ccd']["algo_list"])
        if number_of_samples <= 2:
            number_of_rows = 1
            number_of_columns = number_of_samples
        else:
            number_of_rows = 2 if number_of_samples % 2 == 0 else 1
            number_of_columns = number_of_samples // 2 if number_of_samples % 2 == 0 else number_of_samples
        fig, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, figsize=(20, 20), squeeze=False)
        axes = axes.ravel()
        fig.subplots_adjust(wspace=0)
        set_figure_params(dpi=params.get('dpi'), facecolor='white')

        cell_percentage_dfs = []
        plot_columns = []
        for algo in self.pipeline_res['ccd']["algo_list"]:
            cell_percentage_dfs.append(pd.DataFrame(
                algo.get_adata().obs[algo.annotation].value_counts(normalize=True).mul(100).rename(algo.filename)))
            plot_columns.append(algo.filename)

        cummulative_df = pd.concat(cell_percentage_dfs, axis=1).fillna(0)

        for i in range(number_of_rows * number_of_columns):
            axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            axes[i].set_facecolor('white')
            axes[i].set_title(plot_columns[i])
            cummulative_df.plot(y=plot_columns[i], kind="bar", rot=70, ax=axes[i], xlabel="", color="grey",
                                legend=False)
            axes[i].grid(False)

        for ax in axes:
            ax.grid(False)

        return plt.figure()

    def plot_cluster_abundance_total(self, **params):
        """
        Plots the total cluster abundance for each algorithm.
        """
        fig, ax = plt.subplots(figsize=(20, 20))
        fig.subplots_adjust(wspace=0)
        set_figure_params(dpi=params.get('dpi'), facecolor='white')

        greys = cycle(['darkgray', 'gray', 'dimgray', 'lightgray'])
        colors = [next(greys) for _ in range(len(self.pipeline_res['ccd']["algo_list"]))]
        cell_percentage_dfs = []
        plot_columns = []
        for algo in self.pipeline_res['ccd']["algo_list"]:
            cell_percentage_dfs.append(pd.DataFrame(
                algo.get_adata().obs[f'tissue_{algo.method_key}'].value_counts(normalize=True).mul(100).rename(
                    algo.filename)))
            plot_columns.append(algo.filename)

        cummulative_df = pd.concat(cell_percentage_dfs, axis=1).fillna(0)
        cummulative_df = cummulative_df.loc[
            sorted(cummulative_df.index.values, key=lambda x: float(x) if x != "unknown" else float('inf'))]
        cummulative_df.plot(y=plot_columns, kind="bar", rot=0, ax=ax, xlabel="", color=colors)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.grid(False)
        ax.set_facecolor('white')
        plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

        return plt.figure()

    def plot_cluster_abundance_per_slice(self, **params):
        """
        Plots the cluster abundance for each algorithm per slice.
        """
        number_of_samples = len(self.pipeline_res['ccd']["algo_list"])
        if number_of_samples <= 2:
            number_of_rows = 1
            number_of_columns = number_of_samples
        else:
            number_of_rows = 2 if number_of_samples % 2 == 0 else 1
            number_of_columns = number_of_samples // 2 if number_of_samples % 2 == 0 else number_of_samples
        fig, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, figsize=(20, 20), squeeze=False)
        axes = axes.ravel()
        fig.subplots_adjust(wspace=0)
        set_figure_params(dpi=params.get('dpi'), facecolor='white')

        cell_percentage_dfs = []
        plot_columns = []
        for algo in self.pipeline_res['ccd']["algo_list"]:
            cell_percentage_dfs.append(pd.DataFrame(
                algo.get_adata().obs[f'tissue_{algo.method_key}'].value_counts(normalize=True).mul(100).rename(
                    algo.filename)))
            plot_columns.append(algo.filename)

        cummulative_df = pd.concat(cell_percentage_dfs, axis=1).fillna(0)
        cummulative_df = cummulative_df.loc[
            sorted(cummulative_df.index.values, key=lambda x: float(x) if x != "unknown" else float('inf'))]

        for i in range(number_of_rows * number_of_columns):
            axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            axes[i].set_facecolor('white')
            axes[i].set_title(plot_columns[i])
            cummulative_df.plot(y=plot_columns[i], kind="bar", rot=0, ax=axes[i], xlabel="", color="grey", legend=False)
            axes[i].grid(False)

        for ax in axes:
            ax.grid(False)

        return plt.figure()
