from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stereo.algorithm.algorithm_base import AlgorithmBase


class TimeSeriesAnalysis(AlgorithmBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def boxplot_transit_gene(self, adata, use_col, branch, genes, layer=None, vmax=None, vmin=None):
        """
        show a boxplot of a specific gene expression in branch of use_col
        :param adata: anndata object to analysis
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :param genes: specific gene or gene list to plot 
        :param layer: layer in adata.layers[layer] to get expression, default None means use adata.X
        :param vmax, vmin: max and min value to plot, default None means auto calculate
        :return: a boxplot fig of one or several gene expression in branch of use_col
        """
        branch2exp = defaultdict(dict)
        for x in branch:
            celllist = adata.obs.loc[adata.obs[use_col] == x, :].index
            for gene in genes:
                if not layer:
                    branch2exp[gene][x] = adata[celllist, gene].X.toarray().flatten()
                else:
                    branch2exp[gene][x] = adata[celllist, gene].layers[layer].toarray().flatten()

        fig = plt.figure(figsize=(4 * len(genes), 6))
        ax = fig.subplots(1, len(genes))
        if len(genes) == 1:
            for i, g in enumerate(genes):
                ax.boxplot(list(branch2exp[g].values()), labels=list(branch2exp[g].keys()))
                ax.set_title(g)
                if vmax != None:
                    if vmin == None:
                        ax.set_ylim(0, vmax)
                    else:
                        ax.set_ylim(vmin, vmax)
            # plt.show()
            return fig
        else:
            for i, g in enumerate(genes):
                ax[i].boxplot(list(branch2exp[g].values()), labels=list(branch2exp[g].keys()))
                ax[i].set_title(g)
                if vmax != None:
                    if vmin == None:
                        ax[i].set_ylim(0, vmax)
                    else:
                        ax[i].set_ylim(vmin, vmax)
            # plt.show()
            return fig

    def TVG_marker(self, adata, use_col, branch, layer=None, p_val_combination='fisher'):
        """
        Calculate time variable gene based on expression of celltypes in branch
        :param adata: anndata object to analysis
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :param layer: layer in adata.layers[layer] to get expression, default None means use adata.X
        :param p_val_combination: p_value combination method to use, choosing from ['fisher', 'mean', 'FDR']
        :return: adata contains Time Variabel Gene marker result
        """
        from scipy.stats import ttest_ind
        from scipy import sparse
        from scipy import stats
        label2exp = {}
        for x in branch:
            celllist = adata.obs.loc[adata.obs[use_col] == x,].index
            if layer in adata.layers:
                if sparse.issparse(adata.layers[layer]):
                    label2exp[x] = adata[celllist, :].layers[layer].todense()
                else:
                    label2exp[x] = adata[celllist, :].layers[layer]
            else:
                if sparse.issparse(adata.X):
                    label2exp[x] = adata[celllist, :].X.todense()
                else:
                    label2exp[x] = adata[celllist, :].X
        # result_df = {}
        logFC = []
        less_pvalue = []
        greater_pvalue = []
        scores = []
        for i in range(len(branch) - 1):
            score, pvalue = ttest_ind(label2exp[branch[i + 1]], label2exp[branch[i]], axis=0, alternative='less')
            # np.nan_to_num(score, nan=0, copy = False)
            less_pvalue.append(np.nan_to_num(pvalue, nan=1, copy=False))
            score, pvalue = ttest_ind(label2exp[branch[i + 1]], label2exp[branch[i]], axis=0, alternative='greater')
            greater_pvalue.append(np.nan_to_num(pvalue, nan=1, copy=False))
            logFC.append(np.array(np.log2(
                (np.mean(label2exp[branch[i + 1]], axis=0) + 1e-9) / (np.mean(label2exp[branch[i]], axis=0) + 1e-9)))[
                             0])
            scores.append(score)
        adata.varm['scores'] = np.array(scores).T
        adata.varm['scores'] = np.nan_to_num(adata.varm['scores'])
        adata.varm['greater_p'] = np.array(greater_pvalue).T
        adata.varm['less_p'] = np.array(less_pvalue).T
        logFC = np.array(logFC).T
        # return logFC
        adata.varm['logFC'] = logFC
        logFC = np.mean(logFC, axis=1)

        if p_val_combination == 'mean':
            less_pvalue = np.mean(np.array(less_pvalue), axis=0)
            greater_pvalue = np.mean(np.array(greater_pvalue), axis=0)
        elif p_val_combination == 'fisher':
            tmp = adata.varm['less_p'].copy()
            tmp[tmp == 0] = np.min(tmp[tmp != 0])
            tmp = np.sum(-2 * np.log(tmp), axis=1)
            less_pvalue = 1 - stats.chi2.cdf(tmp, adata.varm['less_p'].shape[1])
            tmp = adata.varm['greater_p'].copy()
            tmp[tmp == 0] = np.min(tmp[tmp != 0])
            tmp = np.sum(-2 * np.log(tmp), axis=1)
            greater_pvalue = 1 - stats.chi2.cdf(tmp, adata.varm['greater_p'].shape[1])
        elif p_val_combination == 'FDR':
            less_pvalue = 1 - np.prod(1 - adata.varm['less_p'], axis=1)
            greater_pvalue = 1 - np.prod(1 - adata.varm['greater_p'], axis=1)
        # result_df['logFC'] = less_FC
        # result_df['less_pvalue'] = less_pvalue
        # result_df['greater_pvalue'] = greater_pvalue
        # result_df = pd.DataFrame(result_df)
        adata.var['less_pvalue'] = less_pvalue
        adata.var['greater_pvalue'] = greater_pvalue
        adata.var['logFC'] = logFC

        return adata

    def TVG_volcano_plot(self, adata, use_col, branch):
        """
        Use fuzzy C means cluster method to cluster genes based on 1-p_value of celltypes in branch
        :param adata: anndata object to analysis
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :return: a volcano plot display time variable gene(TVG)
        """
        N_branch = len(branch)
        df = adata.var.copy()
        df['_log_pvalue'] = -np.log10(np.min(df[['less_pvalue', 'greater_pvalue']], axis=1))
        label2meam_exp = {}
        for x in branch:
            celllist = adata.obs.loc[adata.obs[use_col] == x,].index
            label2meam_exp[x] = np.array(np.mean(adata[celllist].X, axis=0)).flatten()
        label2meam_exp = pd.DataFrame(label2meam_exp)
        label2meam_exp = label2meam_exp[branch]
        rate = 30 / (np.max(np.percentile(label2meam_exp, 99.99)))
        label2meam_exp = label2meam_exp * rate

        branch_color = [plt.cm.viridis.colors[(int(255 / (N_branch - 1)) * x)] for x in range(N_branch)]

        fig = plt.figure(figsize=(15, 15))
        ax = fig.subplots(1)
        # ax.scatter(df['logFC'], df['_log_pvalue'])

        zorder2color = {(11 + x): [] for x in range(N_branch)}
        zorder2x = {(11 + x): [] for x in range(N_branch)}
        zorder2y = {(11 + x): [] for x in range(N_branch)}
        zorder2s = {(11 + x): [] for x in range(N_branch)}
        for i, gene in enumerate(df.index):
            tmp_exp = list(label2meam_exp.loc[i])
            # print((np.array(tmp_exp)**2*100)+1)
            tmp_zorder = N_branch + 10 - np.argsort(tmp_exp)
            for j, tmp_branch in enumerate(branch):
                # ax.scatter(df.loc[gene, 'logFC'],df.loc[gene, '_log_pvalue'], s=((tmp_exp[j])*100)+1, zorder=tmp_zorder[j], color=branch_color[j], alpha=0.5, linewidths=0.2, edgecolors='black')
                zorder2color[tmp_zorder[j]].append(branch_color[j])
                zorder2x[tmp_zorder[j]].append(df.loc[gene, 'logFC'])
                zorder2y[tmp_zorder[j]].append(df.loc[gene, '_log_pvalue'])
                zorder2s[tmp_zorder[j]].append(((tmp_exp[j]) ** 2) + 1)

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, np.max([np.max(x) for x in zorder2y.values()]) + 1)

        for x in range(N_branch):
            z = 11 + x
            ax.scatter(zorder2x[z], zorder2y[z], s=zorder2s[z], zorder=z, color=zorder2color[z], alpha=0.3,
                       linewidths=0.5, edgecolors='black')

        ax.set_ylabel('-log(p_value)', fontsize=20)
        ax.set_xlabel('mean logFoldChange', fontsize=20)
        ax.set_title('Volcano plot of trajectory', fontsize=25)
        for x, y in enumerate(branch):
            ax.scatter([0], [100000000], alpha=0.3, s=300, color=branch_color[x], label=y, linewidths=0.5,
                       edgecolors='black')
        ax.legend(fontsize=20)

        return fig

    def fuzzy_C(self, data, cluster_number, MAX=10000, m=2, Epsilon=1e-7):
        """
        fuzzy C means algorithm to cluster, helper function used in fuzzy_C_gene_pattern_cluster
        :param data: pd.DataFrame object for fuzzy C means cluster, each col represent a feature, each row represent a obsversion
        :param cluster_number: number of cluster 
        :param MAX: max value to random initialize
        :param m: degree of membership, default = 2
        :param Epsilon: max value to finish iteration
        :return: fuzzy C means cluster result
        """
        assert m > 1
        import time
        import copy
        from scipy import spatial
        U = np.random.randint(1, int(MAX), (len(data), cluster_number))
        U = U / np.sum(U, axis=1, keepdims=True)
        epoch = 0
        tik = time.time()
        while (True):
            epoch += 1
            U_old = copy.deepcopy(U)
            U1 = U ** m
            U2 = np.expand_dims(U1, axis=2)
            U2 = np.repeat(U2, data.shape[1], axis=2)
            data1 = np.expand_dims(data, axis=1)
            data1 = np.repeat(data1, cluster_number, axis=1)
            dummy_sum_num = np.sum(U2 * data1, axis=0)
            dummy_sum_dum = np.sum(U1, axis=0)
            C = (dummy_sum_num.T / dummy_sum_dum).T

            # initializing distance matrix
            distance_matrix = spatial.distance_matrix(data, C)

            # update U
            distance_matrix_1 = np.expand_dims(distance_matrix, axis=1)
            distance_matrix_1 = np.repeat(distance_matrix_1, cluster_number, axis=1)
            distance_matrix_2 = np.expand_dims(distance_matrix, axis=2)
            distance_matrix_2 = np.repeat(distance_matrix_2, cluster_number, axis=2)

            U = np.sum((distance_matrix_1 / distance_matrix_2) ** (2 / (m - 1)), axis=1)
            U = 1 / U
            if epoch % 100 == 0:
                print('epoch {} : time cosumed{:.4f}s, loss:{}'.format(epoch, time.time() - tik,
                                                                       np.max(np.abs(U - U_old))))
                tik = time.time()
            if np.max(np.abs(U - U_old)) < Epsilon:
                break
        return U

    def fuzzy_C_gene_pattern_cluster(self, adata, use_col=None, branch=None):
        """
        Use fuzzy C means cluster method to cluster genes based on 1-p_value of celltypes in branch
        :param adata: anndata object to analysis
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :return: adata contains fuzzy_C_result 
        """
        if ('greater_p' not in adata.varm) or ('less_p' not in adata.varm):
            if use_col == None:
                print('greater_p and less_p not in adata.varm, you should run get_gene_pattern first')
            else:
                self.TVG_marker(adata, use_col=use_col, branch=branch)
        sig = ((1 - adata.varm['greater_p']) >= (1 - adata.varm['less_p'])).astype(int)
        sig[sig == 0] = -1
        adata.varm['feature_p'] = np.max([(1 - adata.varm['greater_p']), (1 - adata.varm['less_p'])], axis=0) * sig
        adata.varm['fuzzy_C_weight'] = self.fuzzy_C(adata.varm['feature_p'], 6, Epsilon=1e-7)
        adata.var['fuzzy_C_result'] = np.argmax(adata.varm['fuzzy_C_weight'], axis=1)

    def bezierpath(self, rs, re, qs, qe, ry, qy, v=True, col='green', alpha=0.2, label='', lw=0, zorder=0):
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

    def paga_time_series_plot(self, adata, use_col='celltype', batch_col='sample_name',
                              groups=None, fig_height=10, fig_width=0, palette='tab20',
                              link_alpha=0.5, spot_size=1, dpt_col='dpt_pseudotime'):
        """
        spatial trajectory plot for paga in time_series multiple slice dataset
        :param adata: anndata object of multiple slice
        :param use_col: the col in obs representing celltype or clustering
        :param batch_col: the col in obs representing different slice of time series
        :param groups: the particular celltype that will show, default None means show all the celltype in use_col
        :param fig_height: height of figure
        :param fig_width: width of figure
        :param palette: color palette to paint different celltypes
        :param link_alpha: alpha of the bezierpath, from 0 to 1
        :param spot_size: the size of each cell scatter
        :param dpt_col: the col in obs representing dpt pseudotime
        :return: a fig object
        """

        import networkx as nx
        import random
        import seaborn as sns
        from matplotlib.patches import Polygon

        from collections import defaultdict
        from scipy import spatial
        import random
        from itertools import cycle
        import seaborn as sns
        from scipy import stats

        # 细胞类型的列表
        ct_list = list(adata.obs[use_col].astype('category').cat.categories)
        # 多篇数据存储分片信息的列表
        bc_list = list(adata.obs[batch_col].astype('category').cat.categories)
        # bc_list = ['E9.5_E1S1', 'E10.5_E1S1', 'E11.5_E1S1', 'E12.5_E1S1', 'E13.5_E1S1', 'E14.5_E1S1']

        # 生成每个细胞类型的颜色对应
        colors = sns.color_palette(palette, len(ct_list))
        ct2color = dict(zip(ct_list, colors))
        adata.obs['tmp'] = adata.obs[use_col].astype(str) + '|' + adata.obs[batch_col].astype(str)

        # 读取paga结果
        G = pd.DataFrame(adata.uns['paga']['connectivities_tree'].todense())
        G.index, G.columns = ct_list, ct_list
        # 转化为对角矩阵
        for i, x in enumerate(ct_list):
            for y in ct_list[i + 1:]:
                G[x][y] = G[y][x] = G[x][y] + G[y][x]
        # 利用伪时序计算结果推断方向
        if dpt_col in adata.obs:
            ct2dpt = {}

            for x, y in adata.obs.groupby(use_col):
                ct2dpt[x] = y[dpt_col].to_numpy()

            from collections import defaultdict
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
            if flag == 0:
                ct2color[c] = (0.95, 0.95, 0.95)

        # 创建画布
        if fig_height != None:
            fig_width_0 = fig_height * np.ptp(adata.obsm['spatial'][:, 0]) / np.ptp(adata.obsm['spatial'][:, 1])
            fig_height_0 = 0
        if fig_width > 0:
            fig_height_0 = fig_width * np.ptp(adata.obsm['spatial'][:, 1]) / np.ptp(adata.obsm['spatial'][:, 0])
        if fig_height > fig_height_0:
            fig_width = fig_width_0
        else:
            fig_height = fig_height_0
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.subplots(1, 1)

        # 计算每个元素（细胞类型标签，箭头起点终点，贝塞尔曲线的起点和终点）的坐标
        median_center = {}
        min_vertex_left = {}
        min_vertex_right = {}
        max_vertex_left = {}
        max_vertex_right = {}
        for x in adata.obs.groupby('tmp'):
            tmp = adata[x[1].index, :].obsm['spatial']
            # ax.scatter(tmp[:, 0], 0-tmp[:, 1], label=x[0].split('|')[0], color = ct2color[x[0].split('|')[0]], s =s )
            ### --*-- to infer the 4 point to plot Polygon
            tmp_left = tmp[(np.percentile(tmp[:, 0], 40) <= tmp[:, 0]) & (tmp[:, 0] <= np.percentile(tmp[:, 0], 60))]
            tmp_right = tmp[(np.percentile(tmp[:, 0], 40) <= tmp[:, 0]) & (tmp[:, 0] <= np.percentile(tmp[:, 0], 60))]
            # tmp_left = tmp[(tmp[:,0]<=np.percentile(tmp[:,0],20))]
            # tmp_right = tmp[(np.percentile(tmp[:,0],8)<=tmp[:,0])]
            min_vertex_left[x[0]] = [np.mean(tmp_left[:, 0]), 0 - np.percentile(tmp_left[:, 1], 10)]
            min_vertex_right[x[0]] = [np.mean(tmp_right[:, 0]), 0 - np.percentile(tmp_right[:, 1], 10)]
            max_vertex_left[x[0]] = [np.mean(tmp_left[:, 0]), 0 - np.percentile(tmp_left[:, 1], 90)]
            max_vertex_right[x[0]] = [np.mean(tmp_right[:, 0]), 0 - np.percentile(tmp_right[:, 1], 90)]
            ### --*--
            # max_vertex[x[0]]= [np.mean(tmp[:,0]) ,0-np.percentile(tmp[:,1], 90)]
            # max_vertex[x[0]]= [tmp[np.argmin(tmp[:,1])][0] ,0-tmp[np.argmin(tmp[:,1])][1]]
            x_tmp, y_tmp = tmp[np.argmin(np.sum((tmp - np.mean(tmp, axis=0)) ** 2, axis=1))]
            median_center[x[0]] = [x_tmp, 0 - y_tmp]
            ax.text(x_tmp, 0 - y_tmp, s=x[0].split('|')[0], c='black',
                    alpha=0.8)  # , bbox=dict(facecolor='red', alpha=0.1))
            # break

        # 细胞散点元素
        for x in adata.obs.groupby(use_col):
            tmp = adata[x[1].index, :].obsm['spatial']
            ax.scatter(tmp[:, 0], 0 - tmp[:, 1], label=x[0], color=ct2color[x[0]], s=spot_size)
        ## all arrow

        # 决定箭头曲率方向的阈值
        threshold_y = np.mean(np.array(list(median_center.values()))[:, 1])

        # 画箭头
        for bc in range(len(bc_list) - 1):
            for edge in G.edges():
                edge0, edge1 = edge[0] + '|' + bc_list[bc], edge[1] + '|' + bc_list[bc + 1]
                # if (edge0 in median_center) and (edge1 in median_center):
                if (edge0 in median_center) and (edge1 in median_center) and (
                        (edge[0] in groups) or (edge[1] in groups)):
                    x1, y1 = median_center[edge0]
                    x2, y2 = median_center[edge1]
                    # ax.arrow(x1,y1,x2-x1,y2-y1, width=70, fc ='#f9f8e6', ec ='black',capstyle = 'round', arrowprops=dict(connectionstyle="arc3,rad=0.4"))
                    # ax.annotate('', (x2,y2), (x1,y1), arrowprops=dict(connectionstyle="arc3,rad=0.4", ec = 'black',color='#f9f8e6',width = 5))
                    if (y1 + y2) / 2 < threshold_y:
                        ax.annotate('', (x2, y2), (x1, y1),
                                    arrowprops=dict(connectionstyle="arc3,rad=0.4", ec='black', color='#f9f8e6',
                                                    arrowstyle='simple', alpha=0.5))
                    else:
                        ax.annotate('', (x2, y2), (x1, y1),
                                    arrowprops=dict(connectionstyle="arc3,rad=-0.4", ec='black', color='#f9f8e6',
                                                    arrowstyle='simple', alpha=0.5))

        # 贝塞尔曲线流形显示
        if len(groups) == 1:
            for g in groups:
                for bc in range(len(bc_list) - 1):
                    edge0, edge1 = g + '|' + bc_list[bc], g + '|' + bc_list[bc + 1]
                    if (edge0 in median_center) and (edge1 in median_center):
                        # tmp_vertex = [max_vertex_right[edge0],min_vertex_right[edge0], min_vertex_left[edge1], max_vertex_left[edge1]]
                        # p = Polygon(tmp_vertex, color=ct2color[g], ec=None, alpha = 0.1)
                        p = self.bezierpath(min_vertex_right[edge0][1], max_vertex_right[edge0][1], \
                                            min_vertex_left[edge1][1], max_vertex_right[edge1][1], \
                                            min_vertex_right[edge0][0], min_vertex_left[edge1][0], True,
                                            col=ct2color[g], alpha=link_alpha)
                        ax.add_patch(p)

        # 显示时期的label
        for x in adata.obs.groupby(batch_col):
            tmp = adata[x[1].index, :].obsm['spatial']
            ax.text(np.mean(tmp[:, 0]), 0 - np.min(tmp[:, 1]) + 1, s=x[0], c='black', fontsize=20, ha='center',
                    va='bottom')  # , bbox=dict(facecolor='red', alpha=0.1))

        plt.legend(ncol=int(np.sqrt(len(ct_list) / 6)) + 1)
        # plt.savefig('./plot/mouse_embyro_trajectory.png', dpi=300, bbox_inches='tight')
        # plt.show()
        return fig
