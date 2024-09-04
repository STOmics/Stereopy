#!/usr/bin/env python3
# coding: utf-8
"""
@author: wenzhenbin  wenzhenbin@genomics.cn
@last modified by: wenzhenbin
@file:constant.py
@time:2023/08/01
"""
# flake8: noqa

from collections import defaultdict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stereo.constant import ANNOTATION
from stereo.constant import BATCH
from stereo.constant import BatchColType
from stereo.constant import CATEGORY
from stereo.constant import CELLTYPE_MEAN
from stereo.constant import CELLTYPE_MEAN_SCALE
from stereo.constant import CELLTYPE_STD
from stereo.constant import CONNECTIVITIES_TREE
from stereo.constant import ColorType
from stereo.constant import DirectionType
from stereo.constant import DptColType
from stereo.constant import FUZZY_C_WEIGHT
from stereo.constant import GREATER_PVALUE
from stereo.constant import GROUP
from stereo.constant import INDEX
from stereo.constant import LESS_PVALUE
from stereo.constant import LOG_FC
from stereo.constant import PAGA
from stereo.constant import PaletteType
from stereo.constant import SANKEY
from stereo.constant import SIMPLE
from stereo.constant import TMP
from stereo.constant import _LOG_PVALUE
from stereo.plots.decorator import reorganize_coordinate
from stereo.plots.ms_plot_base import MSDataPlotBase
from stereo.plots.plot_base import PlotBase


class PlotTimeSeries(PlotBase):

    def boxplot_transit_gene(self,
                             use_col: str,
                             branch: List[str],
                             genes: List[str],
                             vmax: Optional[float] = None,
                             vmin: Optional[float] = None,
                             title: Optional[str] = None
                             ):
        """
        show a boxplot of a specific gene expression in branch of use_col

        :param use_col: the col in obs representing celltype or clustering.
        :param branch: celltypes order in use_col.
        :param genes: specific gene or gene list to plot.
        :param vmax, vmin: max and min value to plot, default None means auto calculate.
        :param title: title of figure.

        :return: a boxplot fig of one or several gene expression in branch of use_col.
        """

        branch2exp = defaultdict(dict)
        stereo_exp_data = self.stereo_exp_data
        for x in branch:
            # cell_list = stereo_exp_data.cells.to_df().loc[stereo_exp_data.cells[use_col] == x, :].index
            # tmp_exp_data = stereo_exp_data.sub_by_name(cell_name=cell_list)
            cell_flag = (stereo_exp_data.cells[use_col] == x).to_numpy()
            tmp_exp_data = stereo_exp_data.exp_matrix[cell_flag]
            for gene in genes:
                # branch2exp[gene][x] = tmp_exp_data.sub_by_name(gene_name=[gene]).exp_matrix.toarray().flatten()
                if stereo_exp_data.issparse():
                    branch2exp[gene][x] = tmp_exp_data[:, stereo_exp_data.gene_names == gene].toarray().flatten()
                else:
                    branch2exp[gene][x] = tmp_exp_data[:, stereo_exp_data.gene_names == gene].flatten()

        fig = plt.figure(figsize=(4 * len(genes), 6))
        ax = fig.subplots(1, len(genes))
        if len(genes) == 1:
            for i, g in enumerate(genes):
                ax.boxplot(list(branch2exp[g].values()), labels=list(branch2exp[g].keys()))
                ax.set_title(g if title != '' else '')
                if vmax != None:  # noqa
                    if vmin == None:  # noqa
                        ax.set_ylim(0, vmax)
                    else:
                        ax.set_ylim(vmin, vmax)
            return fig
        else:
            for i, g in enumerate(genes):
                ax[i].boxplot(list(branch2exp[g].values()), labels=list(branch2exp[g].keys()))
                ax[i].set_title(g if title != '' else '')
                if vmax != None:
                    if vmin == None:
                        ax[i].set_ylim(0, vmax)
                    else:
                        ax[i].set_ylim(vmin, vmax)
            return fig

    def TVG_volcano_plot(self,
                         use_col: str,
                         branch: List[str],
                         x_label: Optional[str] = None,
                         y_label: Optional[str] = None,
                         title: Optional[str] = None):
        """
        Use fuzzy C means cluster method to cluster genes based on 1-p_value of celltypes in branch

        :param use_col: the col in obs representing celltype or clustering.
        :param branch: celltypes order in use_col.
        :param x_label: the x label.
        :param y_label: the y label.
        :param title: title of figure.

        :return: a volcano plot display time variable gene(TVG).
        """
        N_branch = len(branch)
        stereo_exp_data = self.stereo_exp_data
        df = stereo_exp_data.genes.to_df()
        df[_LOG_PVALUE] = -np.log10(np.min(df[[LESS_PVALUE, GREATER_PVALUE]], axis=1))
        max_replace_value = df.loc[np.isfinite(df[_LOG_PVALUE]), _LOG_PVALUE].max()
        df.loc[~np.isfinite(df[_LOG_PVALUE]), _LOG_PVALUE] = max_replace_value
        label2meam_exp = {}
        for x in branch:
            cell_list = stereo_exp_data.cells.to_df().loc[stereo_exp_data.cells[use_col] == x,].index
            label2meam_exp[x] = np.array(
                np.mean(stereo_exp_data.sub_by_name(cell_name=cell_list).exp_matrix, axis=0)).flatten()
        label2meam_exp = pd.DataFrame(label2meam_exp)
        label2meam_exp = label2meam_exp[branch]
        rate = 30 / (np.max(np.percentile(label2meam_exp, 99.99)))
        label2meam_exp = label2meam_exp * rate

        branch_color = [plt.cm.viridis.colors[(int(255 / (N_branch - 1)) * x)] for x in range(N_branch)]

        fig = plt.figure(figsize=(15, 15))
        ax = fig.subplots(1)

        zorder2color = {(11 + x): [] for x in range(N_branch)}
        zorder2x = {(11 + x): [] for x in range(N_branch)}
        zorder2y = {(11 + x): [] for x in range(N_branch)}
        zorder2s = {(11 + x): [] for x in range(N_branch)}
        for i, gene in enumerate(df.index):
            tmp_exp = list(label2meam_exp.loc[i])
            tmp_zorder = N_branch + 10 - np.argsort(tmp_exp)
            for j, tmp_branch in enumerate(branch):
                zorder2color[tmp_zorder[j]].append(branch_color[j])
                zorder2x[tmp_zorder[j]].append(df.loc[gene, LOG_FC])
                zorder2y[tmp_zorder[j]].append(df.loc[gene, _LOG_PVALUE])
                zorder2s[tmp_zorder[j]].append(((tmp_exp[j]) ** 2) + 1)

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, np.max([np.max(x) for x in zorder2y.values()]) + 1)

        for x in range(N_branch):
            z = 11 + x
            ax.scatter(zorder2x[z], zorder2y[z], s=zorder2s[z], zorder=z, color=zorder2color[z], alpha=0.3,
                       linewidths=0.5, edgecolors=ColorType.black.value)

        ax.set_ylabel('-log(p_value)' if y_label != '' else '', fontsize=20)
        ax.set_xlabel('mean logFoldChange' if x_label != '' else '', fontsize=20)
        ax.set_title('Volcano plot of trajectory' if title != '' else '', fontsize=25)
        for x, y in enumerate(branch):
            ax.scatter([0], [100000000], alpha=0.3, s=300, color=branch_color[x], label=y, linewidths=0.5,
                       edgecolors=ColorType.black.value)
        ax.legend(fontsize=20)

        return fig

    def bezierpath(self,
                   rs,
                   re,
                   qs,
                   qe,
                   ry,
                   qy,
                   v=True,
                   col=ColorType.green.value,
                   alpha=0.2,
                   label='',
                   lw=0,
                   zorder=0):
        """
        bezierpath patch for plot the connection of same organ in different batches
        :return: a patch object to plot
        """

        import matplotlib.patches as patches
        from matplotlib.path import Path
        smid = (qs - rs) / 2  # Start increment
        emid = (qe - re) / 2  # End increment
        hmid = (qy - ry) / 2  # Heinght increment
        if not v:
            verts = [(rs, ry),
                     (rs, ry + hmid),
                     (rs + 2 * smid, ry + hmid),
                     (rs + 2 * smid, ry + 2 * hmid),
                     (qe, qy),
                     (qe, qy - hmid),
                     (qe - 2 * emid, qy - hmid),
                     (qe - 2 * emid, qy - 2 * hmid),
                     (rs, ry),
                     ]
        else:
            verts = [(ry, rs),
                     (ry + hmid, rs),
                     (ry + hmid, rs + 2 * smid),
                     (ry + 2 * hmid, rs + 2 * smid),
                     (qy, qe),
                     (qy - hmid, qe),
                     (qy - hmid, qe - 2 * emid),
                     (qy - 2 * hmid, qe - 2 * emid),
                     (ry, rs),
                     ]
        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=col, lw=lw, alpha=alpha, label=label, edgecolor=col, zorder=zorder)
        return patch

    @reorganize_coordinate
    def paga_time_series_plot(self,
                              use_col: str,
                              batch_col: str,
                              groups: Optional[str] = None,
                              height: Optional[float] = 10,
                              width: Optional[float] = 0,
                              palette: Optional[str] = PaletteType.tab20.value,
                              link_alpha: Optional[float] = 0.5,
                              spot_size: Optional[int] = 1,
                              dpt_col: Optional[str] = DptColType.dpt_pseudotime.value):
        """
        spatial trajectory plot for paga in time_series multiple slice dataset

        :param use_col: the col in obs representing celltype or clustering.
        :param batch_col: the col in obs representing different slice of time series.
        :param groups: the particular celltype that will show, default None means show all the celltype in use_col.
        :param height: height of figure.
        :param width: width of figure.
        :param palette: color palette to paint different celltypes.
        :param link_alpha: alpha of the bezierpath, from 0 to 1.
        :param spot_size: the size of each cell scatter.
        :param dpt_col: the col in obs representing dpt pseudotime.
        :param reorganize_coordinate: if the data is merged from several slices, whether to reorganize the coordinates of the obs(cells), 
                if set it to a number, like 2, the coordinates will be reorganized to 2 columns on coordinate system as below:
                                ---------------
                                | data1 data2
                                | data3 data4
                                | data5 ...  
                                | ...   ...  
                                ---------------
                if set it to `False`, the coordinates will not be changed.
        :param horizontal_offset_additional: the additional offset between each slice on horizontal direction while reorganizing coordinates.
        :param vertical_offset_additional: the additional offset between each slice on vertical direction while reorganizing coordinates.

        :return: a fig object
        """  # noqa
        import networkx as nx
        import seaborn as sns
        from scipy import stats
        import copy
        stereo_exp_data = self.stereo_exp_data

        # 细胞类型的列表
        ct_list = list(stereo_exp_data.cells[use_col].astype(CATEGORY).cat.categories)

        # 多篇数据存储分片信息的列表
        bc_list = list(stereo_exp_data.cells[batch_col].astype(CATEGORY).cat.categories)

        # 生成每个细胞类型的颜色对应
        colors = sns.color_palette(palette, len(ct_list))
        ct2color = dict(zip(ct_list, colors))
        internal_df = copy.deepcopy(stereo_exp_data.cells.to_df())
        internal_df[0] = stereo_exp_data.position[:, 0].astype(float)
        internal_df[1] = stereo_exp_data.position[:, 1].astype(float)
        internal_df[TMP] = internal_df[use_col].astype(str) + '|' + internal_df[batch_col].astype(str)

        # 读取paga结果
        G = pd.DataFrame(stereo_exp_data.tl.result[PAGA][CONNECTIVITIES_TREE].todense())
        G.index, G.columns = ct_list, ct_list

        # 转化为对角矩阵
        for i, x in enumerate(ct_list):
            for y in ct_list[i + 1:]:
                G[x][y] = G[y][x] = G[x][y] + G[y][x]

        # 利用伪时序计算结果推断方向
        if dpt_col in internal_df:
            ct2dpt = {}

            for x, y in internal_df.groupby(use_col):
                ct2dpt[x] = y[dpt_col].to_numpy()

            dpt_ttest = defaultdict(dict)
            for x in ct_list:
                for y in ct_list:
                    dpt_ttest[x][y] = stats.ttest_ind(ct2dpt[x], ct2dpt[y])[0]
            dpt_ttest = pd.DataFrame(dpt_ttest)
            dpt_ttest[dpt_ttest > 0] = 1
            dpt_ttest[dpt_ttest < 0] = 0
            G = dpt_ttest * G

        G = nx.from_pandas_adjacency(G, nx.DiGraph)

        # 如果groups不为空，把无关的细胞变成灰色
        if groups == None:
            groups = ct_list
        elif type(groups) == str:
            groups = [groups]
        for c in ct2color:
            flag = 0
            for c1 in [x for x in G.predecessors(c)] + list(G[c]) + [c]:
                if c1 in groups:
                    flag = 1
                    break
            if flag == 0:
                ct2color[c] = (0.95, 0.95, 0.95)

        # 创建画布
        if height != None:
            width_0 = height * np.ptp(stereo_exp_data.position[:, 0]) / np.ptp(
                stereo_exp_data.position[:, 1])
            height_0 = 0
        if width > 0:
            height_0 = width * np.ptp(stereo_exp_data.position[:, 1]) / np.ptp(
                stereo_exp_data.position[:, 0])
        if height > height_0:
            width = width_0
        else:
            height = height_0
        fig = plt.figure(figsize=(width, height))
        ax = fig.subplots(1, 1)

        # 计算每个元素（细胞类型标签，箭头起点终点，贝塞尔曲线的起点和终点）的坐标
        median_center = {}
        min_vertex_left = {}
        min_vertex_right = {}
        max_vertex_left = {}
        max_vertex_right = {}
        for x in internal_df.groupby(TMP):
            tmp = x[1][[0, 1]].to_numpy()
            tmp_left = tmp[(np.percentile(tmp[:, 0], 40, interpolation='nearest') <= tmp[:, 0]) & (
                        tmp[:, 0] <= np.percentile(tmp[:, 0], 60, interpolation='nearest'))]
            tmp_right = tmp[(np.percentile(tmp[:, 0], 40, interpolation='nearest') <= tmp[:, 0]) & (
                        tmp[:, 0] <= np.percentile(tmp[:, 0], 60, interpolation='nearest'))]
            min_vertex_left[x[0]] = [np.mean(tmp_left[:, 0]), 0 - np.percentile(tmp_left[:, 1], 10)]
            min_vertex_right[x[0]] = [np.mean(tmp_right[:, 0]), 0 - np.percentile(tmp_right[:, 1], 10)]
            max_vertex_left[x[0]] = [np.mean(tmp_left[:, 0]), 0 - np.percentile(tmp_left[:, 1], 90)]
            max_vertex_right[x[0]] = [np.mean(tmp_right[:, 0]), 0 - np.percentile(tmp_right[:, 1], 90)]
            x_tmp, y_tmp = tmp[np.argmin(np.sum((tmp - np.mean(tmp, axis=0)) ** 2, axis=1))]
            median_center[x[0]] = [x_tmp, 0 - y_tmp]
            ax.text(x_tmp, 0 - y_tmp, s=x[0].split('|')[0], c=ColorType.black.value,
                    alpha=0.8)

        # 细胞散点元素
        for x in internal_df.groupby(use_col):
            tmp = x[1][[0, 1]].to_numpy()
            ax.scatter(tmp[:, 0], 0 - tmp[:, 1], label=x[0], color=ct2color[x[0]], s=spot_size)

        # 决定箭头曲率方向的阈值
        threshold_y = np.mean(np.array(list(median_center.values()))[:, 1])

        # 画箭头
        for bc in range(len(bc_list) - 1):
            for edge in G.edges():
                edge0, edge1 = edge[0] + '|' + bc_list[bc], edge[1] + '|' + bc_list[bc + 1]
                if (edge0 in median_center) and (edge1 in median_center) and (
                        (edge[0] in groups) or (edge[1] in groups)):
                    x1, y1 = median_center[edge0]
                    x2, y2 = median_center[edge1]
                    if (y1 + y2) / 2 < threshold_y:
                        ax.annotate('', (x2, y2), (x1, y1),
                                    arrowprops=dict(connectionstyle="arc3,rad=0.4", ec=ColorType.black.value,
                                                    color='#f9f8e6', arrowstyle=SIMPLE, alpha=0.5))
                    else:
                        ax.annotate('', (x2, y2), (x1, y1),
                                    arrowprops=dict(connectionstyle="arc3,rad=-0.4", ec=ColorType.black.value,
                                                    color='#f9f8e6', arrowstyle=SIMPLE, alpha=0.5))

        # 贝塞尔曲线流形显示
        if len(groups) == 1:
            for g in groups:
                for bc in range(len(bc_list) - 1):
                    edge0, edge1 = g + '|' + bc_list[bc], g + '|' + bc_list[bc + 1]
                    if (edge0 in median_center) and (edge1 in median_center):
                        p = self.bezierpath(
                            min_vertex_right[edge0][1],
                            max_vertex_right[edge0][1],
                            min_vertex_left[edge1][1],
                            max_vertex_right[edge1][1],
                            min_vertex_right[edge0][0],
                            min_vertex_left[edge1][0],
                            True,
                            col=ct2color[g],
                            alpha=link_alpha
                        )
                        ax.add_patch(p)

        # 显示时期的label
        for x in internal_df.groupby(batch_col):
            tmp = x[1][[0, 1]].to_numpy()
            ax.text(np.mean(tmp[:, 0]), 0 - np.min(tmp[:, 1]) + 1, s=x[0], c=ColorType.black.value, fontsize=20,
                    ha=DirectionType.center.value, va=DirectionType.bottom.value)

        plt.legend(ncol=int(np.sqrt(len(ct_list) / 6)) + 1)
        return fig

    def fuzz_cluster_plot(self,
                          use_col: str,
                          branch: str,
                          threshold: Optional[str] = 'p99.98',
                          summary_trend: Optional[bool] = True,
                          n_col: Optional[int] = None,
                          width: Optional[float] = None,
                          height: Optional[float] = None,
                          x_label: Optional[str] = None,
                          y_label: Optional[str] = None,
                          title: Optional[str] = None):
        """
        a line plot to show the trend of each cluster of fuzzy C means

        :param use_col: the col in obs representing celltype or clustering.
        :param branch: celltypes order in use_col.
        :param summary_trend: summary trend in use_col.
        :param threshold: the threshold of cluster score to plot.
        :param n_col: number of columns to display each cluster plot.
        :param width: width of figure.
        :param height: height of figure.
        :param x_label: the x label.
        :param y_label: the y label.
        :param title: title of figure.

        :return: a list of gene list of each cluster.
        """
        from scipy import sparse
        data = self.stereo_exp_data

        # 计算阈值
        if (CELLTYPE_MEAN not in data.genes_matrix) or (CELLTYPE_STD not in data.genes_matrix):
            label2exp = {}
            for x in branch:
                cell_list = data.cells.to_df().loc[data.cells[use_col] == x,].index
                test_exp_data = data.sub_by_name(cell_name=cell_list.to_list())
                if sparse.issparse(test_exp_data.exp_matrix):
                    label2exp[x] = test_exp_data.exp_matrix.todense()
                else:
                    label2exp[x] = np.mat(test_exp_data.exp_matrix)

            label2mean = pd.DataFrame({x: np.mean(np.array(y), axis=0) for x, y in label2exp.items()})
            label2mean.index = data.gene_names
            data.genes_matrix[CELLTYPE_MEAN] = label2mean
            tmp_exp = label2mean.sub(np.mean(label2mean, axis=1), axis=0)
            tmp_exp = tmp_exp.divide(np.std(tmp_exp, axis=1), axis=0)
            data.genes_matrix[CELLTYPE_MEAN_SCALE] = tmp_exp
        NC = data.genes_matrix[FUZZY_C_WEIGHT].shape[1]

        # 创建画布
        if n_col == None:
            n_col = int(np.ceil(np.sqrt(NC)))
        
        if n_col > NC:
            n_col = NC
        
        n_row = int(np.ceil(NC / n_col))

        if (width == None) and (height == None):
            fig = plt.figure(figsize=(n_col * 5, n_row * 5))
        elif (width == None):
            fig = plt.figure(figsize=(n_col * height / n_row, height))
        elif (height == None):
            fig = plt.figure(figsize=(width, width * n_row / n_col))
        else:
            fig = plt.figure(figsize=(width, height))

        plt.set_loglevel('WARNING')
        axs = fig.subplots(n_row, n_col)
        for x in range(NC):
            if n_row == 1 or n_col == 1:
                ax = axs[x]
            else:
                ax = axs[int(x / n_col), x % n_col]
            tmp = data.genes_matrix[FUZZY_C_WEIGHT][:, x]

            if type(threshold) == int or type(threshold) == float:
                threshold_0 = float(threshold)
            elif threshold[0] == 'p':
                threshold_0 = np.percentile(tmp, float(threshold[1:]))

            tmp = tmp >= threshold_0
            genelist = data.gene_names[tmp]
            for row in data.genes_matrix[CELLTYPE_MEAN_SCALE].loc[genelist].iterrows():
                ax.plot(row[1], label=row[0], alpha=0.5)

            if summary_trend or data.genes_matrix[CELLTYPE_MEAN_SCALE].shape[1] > 3:
                from scipy.interpolate import make_interp_spline
                tmp = data.genes_matrix[CELLTYPE_MEAN_SCALE].loc[genelist]
                y_1 = tmp.quantile(0.5)
                y_0 = tmp.quantile(0.25)
                y_2 = tmp.quantile(0.75)
                x_0 = np.arange(len(y_1))
                m_1 = make_interp_spline(x_0, y_1)
                m_0 = make_interp_spline(x_0, y_0)
                m_2 = make_interp_spline(x_0, y_2)
                xs = np.linspace(0, tmp.shape[1] - 1, 500)
                ys_1 = m_1(xs)
                ys_0 = m_0(xs)
                ys_2 = m_2(xs)
                ax.plot(xs, ys_1, c=ColorType.black.value)
                ax.fill(np.concatenate((xs, xs[::-1])), np.concatenate((ys_0, ys_2[::-1])), alpha=0.1)
            ax.set_ylabel('Log1p Exp' if y_label != '' else '')
            ax.set_xticks([x for x in range(data.genes_matrix[CELLTYPE_MEAN].shape[1])])
            if x_label != '':
                ax.set_xticklabels(list(data.genes_matrix[CELLTYPE_MEAN].columns), rotation=20, ha='right')
            ax.set_title(f'Cluster {x + 1}' if title != '' else '')

        for x in range(n_row * n_col - NC):
            y = x + NC
            ax = axs[int(y / n_col), y % n_col]
            ax.axis('off')

        return fig


class PlotTimeSeriesAnalysis(MSDataPlotBase, PlotTimeSeries):

    def time_series_tree_plot(self,
                              use_result: Optional[str] = ANNOTATION,
                              method: Optional[str] = SANKEY,
                              edges: Optional[str] = None,
                              dot_size_scale: Optional[int] = 300,
                              palette: Optional[str] = PaletteType.tab20.value,
                              ylabel_pos: Optional[str] = DirectionType.left.value,
                              width: Optional[float] = 6,
                              height: Optional[float] = 6,
                              x_label: Optional[str] = None,
                              y_label: Optional[str] = None):
        """
        a tree plot to display the cell amounts changes during time series, trajectory can be add to plot by edges.

        :param use_result: the col in obs representing celltype or clustering.
        :param method: choose from sankey and dot, choose the way to display.
        :param edges: a parameter to add arrow to illustrate development trajectory. if edges=='page', use paga result, otherwise use a list of tuple of celltype pairs as father node and child node. # flake8: noqa
        :param dot_size_scale: only used for method='dot', to adjust dot relatively size.
        :param palette: color palette to paint different celltypes.
        :param ylabel_pos: position to plot y labels.
        :param width: width of figure.
        :param height: height of figure.
        :param x_label: the x label.
        :param y_label: the y label.

        :return: a fig object
        """
        # batch list
        import seaborn as sns
        import networkx as nx

        # cell list
        ms_data = self.ms_data
        bc_list = ms_data.names
        ct_list = []
        for x in bc_list:
            ct_list.extend(ms_data[x].tl.result[use_result][GROUP].unique())
        ct_list = np.unique(ct_list)
        colors = sns.color_palette(palette, len(ct_list))
        ct2color = dict(zip(ct_list, colors))

        ret = defaultdict(dict)
        for t in bc_list:
            for x in ct_list:
                ret[x][t] = ms_data[t].tl.result[use_result].loc[ms_data[t].tl.result[use_result][GROUP] == x].shape[0]

        # order the celltype
        ret_df = pd.DataFrame(ret)
        ct2rank = {}
        ret_tmp = ret_df.copy()
        ret_tmp.index = np.arange(ret_tmp.shape[0]) + 1
        for x in ret_tmp:
            ct2rank[x] = np.mean(ret_tmp[x][ret_tmp[x] != 0].index)
        ct2rank = sorted(ct2rank.items(), key=lambda x: x[1], reverse=False)
        ret_df = ret_df[[x[0] for x in ct2rank]]
        hmax = np.max(np.max(ret_df))
        ret_df = ret_df / (2 * hmax)

        if edges == PAGA:
            # 自动获取有paga结果的mss
            scope_flag = 0
            for scope in ms_data.mss:
                if PAGA in ms_data.mss[scope]:
                    scope_flag = 1
                    break

            if scope_flag == 1:
                G = pd.DataFrame(ms_data.tl.result[scope][PAGA][CONNECTIVITIES_TREE].todense())
                G.index, G.columns = ct_list, ct_list
                G = nx.from_pandas_adjacency(G, nx.DiGraph)
                edges = G.edges()
            else:
                edges = None

        fig = plt.figure(figsize=(width, height))
        ax = fig.subplots()

        if method.lower() == SANKEY:
            pos_rec = defaultdict(list)
            i = 0
            yticks = []
            for x in ret_df:
                for j, y in enumerate(ret_df[x]):
                    ax.vlines(j, i - y, i + y, colors=ct2color[x])
                    if y != 0:
                        y1 = y
                        pos_rec[x].append((i - y1, i + y1, j))
                yticks.append(i)
                i += 1

            for x, y in pos_rec.items():
                for i in range(len(y) - 1):
                    tmp0 = y[i]
                    tmp1 = y[i + 1]
                    p = self.bezierpath(tmp0[0], tmp0[1], tmp1[0], tmp1[1], tmp0[2], tmp1[2], True, alpha=0.5,
                                        col=ct2color[x])
                    ax.add_patch(p)

            if edges != None:
                for node_a, node_b in edges:
                    pos_b = pos_rec[node_b]
                    pos_b.sort(key=lambda x: x[2])
                    tmp0 = pos_b[0]
                    x_list = [x[2] for x in pos_rec[node_a]]
                    candidate_list = [x for x in x_list if x < tmp0[2]]
                    if len(candidate_list) > 0:
                        x_a = np.max(candidate_list)
                    else:
                        x_a = np.min(x_list)
                    tmp1 = [x for x in pos_rec[node_a] if x[2] == x_a][0]
                    p = self.bezierpath(tmp0[0], tmp0[1], tmp1[0], tmp1[1], tmp0[2], tmp1[2], True, alpha=0.3,
                                        col=ColorType.grey.value)
                    ax.add_patch(p)

            xticks = np.arange(ret_df.shape[0])
        else:
            timepoint2x = {y: x + 1 for x, y in enumerate(ret_df.index)}
            timepoint2x['root'] = 0

            pos_rec = {}
            for y, ct in enumerate(ret_df.columns):
                time_list = ret_df.loc[ret_df[ct] != 0].index
                tmp_x = [timepoint2x[x] for x in time_list]
                pos_rec[ct] = [tmp_x, y]
                ax.plot([np.max(tmp_x), np.min(tmp_x)], [y, y], c=ColorType.black.value, zorder=0)
                ax.plot([0, 1, np.min(tmp_x)], [(len(ct_list) - 1) / 2, y, y], lw=0.2, c=ColorType.black.value,
                        zorder=0)
                ax.scatter(tmp_x, [y] * len(tmp_x), c=[ct2color[ct]] * len(tmp_x),
                           s=ret_df.loc[time_list, ct] * dot_size_scale)

            ax.scatter(0, (len(ct_list) - 1) / 2, c=ColorType.grey.value, s=1 * dot_size_scale)

            xticks = np.arange(ret_df.shape[0]) + 1
            yticks = np.arange(ret_df.shape[1])
            if edges != None:
                for node_a, node_b in edges:
                    x_b = np.min(pos_rec[node_b][0])
                    y_b = pos_rec[node_b][1]
                    y_a = pos_rec[node_a][1]
                    x_a_list = pos_rec[node_a][0]
                    candidate_list = [x for x in x_a_list if x < x_b]
                    if len(candidate_list) > 0:
                        x_a = np.max(candidate_list)
                    else:
                        x_a = np.min(x_a_list)
                    ax.annotate('', (x_b, y_b), (x_a, y_a),
                                arrowprops=dict(connectionstyle="arc3,rad=-0.2", ec=ColorType.red.value,
                                                color=ColorType.red.value, arrowstyle=SIMPLE, alpha=0.5))

        if x_label != '':
            ax.set_xticks(xticks)
            ax.set_xticklabels(ret_df.index)
        if y_label != '':
            ax.set_yticks(yticks)
            if ylabel_pos == DirectionType.right.value:
                ax.yaxis.tick_right()
            ax.set_yticklabels(ret_df.columns)
        return fig

    def ms_paga_time_series_plot(self,
                                 use_col: str,
                                 groups: Optional[str] = None,
                                 width: Optional[float] = 6,
                                 height: Optional[float] = 6,
                                 palette: Optional[str] = PaletteType.tab20.value,
                                 link_alpha: Optional[float] = 0.5,
                                 spot_size: Optional[int] = 1,
                                 dpt_col: Optional[str] = DptColType.dpt_pseudotime.value):
        """
        spatial trajectory plot for paga in time_series multiple slice dataset

        :param use_col: the col in obs representing celltype or clustering.
        :param groups: the particular celltype that will show, default None means show all the celltype in use_col.
        :param height: height of figure.
        :param width: width of figure.
        :param palette: color palette to paint different celltypes.
        :param link_alpha: alpha of the bezierpath, from 0 to 1.
        :param spot_size: the size of each cell scatter.
        :param dpt_col: the col in obs representing dpt pseudotime.

        :return: a fig object.
        """
        import networkx as nx
        import seaborn as sns
        from scipy import stats
        ms_data = self.ms_data
        # 多篇数据存储分片信息的列表
        bc_list = ms_data.names
        # 细胞类型的列表
        ct_list = ms_data.obs[use_col].astype(CATEGORY).cat.categories

        # 获取多篇的spatial坐标
        position = []
        for x in bc_list:
            position.append(ms_data[x].position)
        position = np.vstack(position)

        obs_df = ms_data.obs.copy()
        batch_col = BatchColType.time.value
        tmp_dic = {str(i): j for i, j in enumerate(bc_list)}
        obs_df[batch_col] = [tmp_dic[x] for x in obs_df[BATCH]]

        # 生成每个细胞类型的颜色对应
        colors = sns.color_palette(palette, len(ct_list))
        ct2color = dict(zip(ct_list, colors))
        obs_df[TMP] = obs_df[use_col].astype(str) + '|' + obs_df[batch_col].astype(str)

        # 读取paga结果
        scope_flag = 0
        for scope in ms_data.mss:
            if PAGA in ms_data.mss[scope]:
                scope_flag = 1
                break
        if scope_flag == 1:
            G = pd.DataFrame(ms_data.tl.result[scope][PAGA][CONNECTIVITIES_TREE].todense())
            G.index, G.columns = ct_list, ct_list
            # 转化为对角矩阵
            for i, x in enumerate(ct_list):
                for y in ct_list[i + 1:]:
                    G[x][y] = G[y][x] = G[x][y] + G[y][x]
            # 利用伪时序计算结果推断方向
            if dpt_col in ms_data.tl.result[scope]:
                ct2dpt = {}
                obs_df[dpt_col] = ms_data.tl.result[scope][dpt_col]

                for x, y in obs_df.groupby(use_col):
                    ct2dpt[x] = y[dpt_col].to_numpy()

                dpt_ttest = defaultdict(dict)
                for x in ct_list:
                    for y in ct_list:
                        dpt_ttest[x][y] = stats.ttest_ind(ct2dpt[x], ct2dpt[y])[0]
                dpt_ttest = pd.DataFrame(dpt_ttest)
                dpt_ttest[dpt_ttest > 0] = 1
                dpt_ttest[dpt_ttest < 0] = 0
                dpt_ttest = dpt_ttest.fillna(0)
                G = dpt_ttest * G

            G = nx.from_pandas_adjacency(G, nx.DiGraph)

        # 如果groups不为空，把无关的细胞变成灰色
        if groups is None:
            groups = ct_list
        elif type(groups) == str:
            groups = [groups]
        for c in ct2color:
            flag = 0
            for c1 in [x for x in G.predecessors(c)] + list(G[c]) + [c]:
                if c1 in groups:
                    flag = 1
            if flag == 0:
                ct2color[c] = (0.95, 0.95, 0.95)

        # 创建画布
        if height != None:
            width_0 = height * np.ptp(position[:, 0]) / np.ptp(
                position[:, 1])
            height_0 = 0
        if width > 0:
            height_0 = width * np.ptp(position[:, 1]) / np.ptp(
                position[:, 0])
        if height > height_0:
            width = width_0
        else:
            height = height_0
        fig = plt.figure(figsize=(width, height))
        ax = fig.subplots(1, 1)

        # 计算每个元素（细胞类型标签，箭头起点终点，贝塞尔曲线的起点和终点）的坐标
        median_center = {}
        min_vertex_left = {}
        min_vertex_right = {}
        max_vertex_left = {}
        max_vertex_right = {}
        obs_df[INDEX] = np.arange(obs_df.shape[0])
        for x in obs_df.groupby(TMP):
            tmp = position[x[1][INDEX], :]
            tmp_left = tmp[(np.percentile(tmp[:, 0], 40) <= tmp[:, 0]) & (tmp[:, 0] <= np.percentile(tmp[:, 0], 60))]
            tmp_right = tmp[(np.percentile(tmp[:, 0], 40) <= tmp[:, 0]) & (tmp[:, 0] <= np.percentile(tmp[:, 0], 60))]

            min_vertex_left[x[0]] = [np.mean(tmp_left[:, 0]), 0 - np.percentile(tmp_left[:, 1], 10)]
            min_vertex_right[x[0]] = [np.mean(tmp_right[:, 0]), 0 - np.percentile(tmp_right[:, 1], 10)]
            max_vertex_left[x[0]] = [np.mean(tmp_left[:, 0]), 0 - np.percentile(tmp_left[:, 1], 90)]
            max_vertex_right[x[0]] = [np.mean(tmp_right[:, 0]), 0 - np.percentile(tmp_right[:, 1], 90)]

            x_tmp, y_tmp = tmp[np.argmin(np.sum((tmp - np.mean(tmp, axis=0)) ** 2, axis=1))]
            median_center[x[0]] = [x_tmp, 0 - y_tmp]
            ax.text(x_tmp, 0 - y_tmp, s=x[0].split('|')[0], c=ColorType.black.value, alpha=0.8)

            # 细胞散点元素
        for x in obs_df.groupby(use_col):
            tmp = position[x[1][INDEX], :]
            ax.scatter(tmp[:, 0], 0 - tmp[:, 1], label=x[0], color=ct2color[x[0]], s=spot_size)

        # 决定箭头曲率方向的阈值
        threshold_y = np.mean(np.array(list(median_center.values()))[:, 1])

        # 画箭头
        for bc in range(len(bc_list) - 1):
            for edge in G.edges():
                edge0, edge1 = edge[0] + '|' + bc_list[bc], edge[1] + '|' + bc_list[bc + 1]
                if (edge0 in median_center) and (edge1 in median_center) and (
                        (edge[0] in groups) or (edge[1] in groups)):
                    x1, y1 = median_center[edge0]
                    x2, y2 = median_center[edge1]
                    if (y1 + y2) / 2 < threshold_y:
                        ax.annotate('', (x2, y2), (x1, y1),
                                    arrowprops=dict(connectionstyle="arc3,rad=0.4", ec=ColorType.black.value,
                                                    color='#f9f8e6', arrowstyle=SIMPLE, alpha=0.5))
                    else:
                        ax.annotate('', (x2, y2), (x1, y1),
                                    arrowprops=dict(connectionstyle="arc3,rad=-0.4", ec=ColorType.black.value,
                                                    color='#f9f8e6', arrowstyle=SIMPLE, alpha=0.5))

        # 贝塞尔曲线流形显示
        if len(groups) == 1:
            for g in groups:
                for bc in range(len(bc_list) - 1):
                    edge0, edge1 = g + '|' + bc_list[bc], g + '|' + bc_list[bc + 1]
                    if (edge0 in median_center) and (edge1 in median_center):
                        p = self.bezierpath(
                            min_vertex_right[edge0][1],
                            max_vertex_right[edge0][1],
                            min_vertex_left[edge1][1],
                            max_vertex_right[edge1][1],
                            min_vertex_right[edge0][0],
                            min_vertex_left[edge1][0],
                            True,
                            col=ct2color[g],
                            alpha=link_alpha
                        )
                        ax.add_patch(p)

        # 显示时期的label
        for x in obs_df.groupby(batch_col):
            tmp = position[x[1][INDEX], :]
            ax.text(np.mean(tmp[:, 0]), 0 - np.min(tmp[:, 1]) + 1, s=x[0], c=ColorType.black.value, fontsize=20,
                    ha=DirectionType.center.value, va=DirectionType.bottom.value)

        plt.legend(ncol=int(np.sqrt(len(ct_list) / 6)) + 1)
        return fig
