from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.plots.plot_base import PlotBase


class TimeSeriesAnalysis(AlgorithmBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def main(self, run_method="tvg_marker", use_col='timepoint', branch=None, p_val_combination='FDR'):
        if run_method == 'tvg_marker':
            self.TVG_marker(
                self.stereo_exp_data,
                use_col=use_col,
                branch=branch,
                p_val_combination=p_val_combination
            )
        else:
            self.fuzzy_C_gene_pattern_cluster(self.stereo_exp_data)

    def TVG_marker(self, stereo_exp_data, use_col, branch, p_val_combination='fisher'):
        """
        Calculate time variable gene based on expression of celltypes in branch
        :param stereo_exp_data: stereo_exp_data object to analysis
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :param p_val_combination: p_value combination method to use, choosing from ['fisher', 'mean', 'FDR']
        :return: stereo_exp_data contains Time Variabel Gene marker result
        """
        from scipy.stats import ttest_ind
        from scipy import sparse
        from scipy import stats
        label2exp = {}
        for x in branch:
            cell_list = stereo_exp_data.cells.to_df().loc[stereo_exp_data.cells[use_col] == x,].index
            test_exp_data = stereo_exp_data.sub_by_name(cell_name=cell_list.to_list())
            if sparse.issparse(test_exp_data.exp_matrix):
                label2exp[x] = test_exp_data.exp_matrix.todense()
            else:
                label2exp[x] = np.mat(test_exp_data.exp_matrix)
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
        stereo_exp_data.genes_matrix['scores'] = np.array(scores).T
        stereo_exp_data.genes_matrix['scores'] = np.nan_to_num(stereo_exp_data.genes_matrix['scores'])
        stereo_exp_data.genes_matrix['greater_p'] = np.array(greater_pvalue).T
        stereo_exp_data.genes_matrix['less_p'] = np.array(less_pvalue).T
        logFC = np.array(logFC).T
        # return logFC
        stereo_exp_data.genes_matrix['logFC'] = logFC
        logFC = np.mean(logFC, axis=1)

        if p_val_combination == 'mean':
            less_pvalue = np.mean(np.array(less_pvalue), axis=0)
            greater_pvalue = np.mean(np.array(greater_pvalue), axis=0)
        elif p_val_combination == 'fisher':
            tmp = stereo_exp_data.genes_matrix['less_p'].copy()
            tmp[tmp == 0] = np.min(tmp[tmp != 0])
            tmp = np.sum(-2 * np.log(tmp), axis=1)
            less_pvalue = 1 - stats.chi2.cdf(tmp, stereo_exp_data.genes_matrix['less_p'].shape[1])
            tmp = stereo_exp_data.genes_matrix['greater_p'].copy()
            tmp[tmp == 0] = np.min(tmp[tmp != 0])
            tmp = np.sum(-2 * np.log(tmp), axis=1)
            greater_pvalue = 1 - stats.chi2.cdf(tmp, stereo_exp_data.genes_matrix['greater_p'].shape[1])
        elif p_val_combination == 'FDR':
            less_pvalue = 1 - np.prod(1 - stereo_exp_data.genes_matrix['less_p'], axis=1)
            greater_pvalue = 1 - np.prod(1 - stereo_exp_data.genes_matrix['greater_p'], axis=1)
        # result_df['logFC'] = less_FC
        # result_df['less_pvalue'] = less_pvalue
        # result_df['greater_pvalue'] = greater_pvalue
        # result_df = pd.DataFrame(result_df)
        stereo_exp_data.genes['less_pvalue'] = less_pvalue
        stereo_exp_data.genes['greater_pvalue'] = greater_pvalue
        stereo_exp_data.genes['logFC'] = logFC
        return stereo_exp_data

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
        while True:
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

    def fuzzy_C_gene_pattern_cluster(self, stereo_exp_data, cluster_number, Epsilon=1e7, use_col=None, branch=None):
        """
        Use fuzzy C means cluster method to cluster genes based on 1-p_value of celltypes in branch
        :param stereo_exp_data: stereo_exp_data object to analysis
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :return: stereo_exp_data contains fuzzy_C_result
        """
        if ('greater_p' not in stereo_exp_data.genes_matrix) or ('less_p' not in stereo_exp_data.genes_matrix):
            if use_col == None:
                print('greater_p and less_p not in stereo_exp_data.genes_matrix, you should run get_gene_pattern first')
            else:
                self.TVG_marker(stereo_exp_data, use_col=use_col, branch=branch)
        sig = ((1 - stereo_exp_data.genes_matrix['greater_p']) >= (1 - stereo_exp_data.genes_matrix['less_p'])).astype(
            int)
        sig[sig == 0] = -1
        stereo_exp_data.genes_matrix['feature_p'] = np.max(
            [(1 - stereo_exp_data.genes_matrix['greater_p']), (1 - stereo_exp_data.genes_matrix['less_p'])],
            axis=0) * sig
        stereo_exp_data.genes_matrix['fuzzy_C_weight'] = self.fuzzy_C(stereo_exp_data.genes_matrix['feature_p'], cluster_number,
                                                                      Epsilon=Epsilon)
        stereo_exp_data.genes['fuzzy_C_result'] = np.argmax(stereo_exp_data.genes_matrix['fuzzy_C_weight'], axis=1)


class PlotTimeSeriesAnalysis(PlotBase):

    def boxplot_transit_gene(self, stereo_exp_data, use_col, branch, genes, vmax=None, vmin=None):
        """
        show a boxplot of a specific gene expression in branch of use_col
        :param stereo_exp_data: stereo_exp_data object to analysis
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :param genes: specific gene or gene list to plot
        :param vmax, vmin: max and min value to plot, default None means auto calculate
        :return: a boxplot fig of one or several gene expression in branch of use_col
        """

        branch2exp = defaultdict(dict)

        for x in branch:
            celllist = stereo_exp_data.cells.loc[stereo_exp_data.cells[use_col] == x, :].index
            tmp_exp_data = stereo_exp_data.sub_by_name(cell_name=celllist)
            for gene in genes:
                branch2exp[gene][x] = tmp_exp_data.sub_by_name(gene_name=[gene]).exp_matrix.toarray().flatten()

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

    def fuzz_cluster_plot(self, data, use_col, branch, threshold = 'p99.98',summary_trend = True, n_col = None, fig_width = None, fig_height = None ):
        """
        a line plot to show the trend of each cluster of fuzzy C means 
        :param data: stereoExpData object with fuzzy C result to plot
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :param threshold: the threshold of cluster score to plot
        :param n_col: number of columns to display each cluster plot
        :return: a list of gene list of each cluster
        """
        from scipy import sparse
        #计算阈值
    
        if ('celltype_mean' not in data.genes_matrix) or ('celltype_std' not in  data.genes_matrix) :
            label2exp = {}
            for x in branch:
                cell_list = data.cells.to_df().loc[data.cells[use_col] == x,].index
                test_exp_data = data.sub_by_name(cell_name=cell_list.to_list())
                if sparse.issparse(test_exp_data.exp_matrix):
                    label2exp[x] = test_exp_data.exp_matrix.todense()
                else:
                    label2exp[x] = np.mat(test_exp_data.exp_matrix)
                    
            label2mean = pd.DataFrame({x:np.mean(np.array(y), axis=0) for x,y in label2exp.items()})
            #label2std = pd.DataFrame({x:np.std(np.array(y), axis=0) for x,y in label2exp.items()})
            
            label2mean.index = data.gene_names
            #label2std.index = data.gene_names
            data.genes_matrix['celltype_mean'] = label2mean
            #data.genes_matrix['celltype_std'] = label2std
            #tmp_exp = adata.varm['celltype_mean']
            tmp_exp = label2mean.sub(np.mean(label2mean, axis=1), axis=0)
            tmp_exp = tmp_exp.divide(np.std(tmp_exp, axis=1), axis=0)
            data.genes_matrix['celltype_mean_scale'] = tmp_exp
        NC = data.genes_matrix['fuzzy_C_weight'].shape[1]
        
        # 创建画布
        if n_col == None:
            n_col = int(np.ceil(np.sqrt(NC)))
        if (fig_width == None) and (fig_height == None):
            fig = plt.figure(figsize=(n_col*5,int(np.ceil(NC/n_col))*5 ))
        elif (fig_width == None):
            fig = plt.figure(figsize=(n_col*fig_height/np.ceil(NC/n_col), fig_height))
        elif (fig_height == None):
            fig = plt.figure(figsize=(fig_width, fig_width*np.ceil(NC/n_col)/n_col ))
        else:
            fig = plt.figure(figsize=(fig_width, fig_height))
            
        axs = fig.subplots(int(np.ceil(NC/n_col)), n_col)
        ret = []
        for x in range(NC):
            if NC == n_col:
                ax = axs[x%n_col]
            else:
                ax = axs[int(x/n_col), x%n_col]
            tmp = data.genes_matrix['fuzzy_C_weight'][:,x]
            
            if type(threshold) == int or type(threshold)==float:
                threshold_0 = float(threshold)
            elif threshold[0] == 'p':
                threshold_0 = np.percentile(tmp, float(threshold[1:]))
            
            tmp = tmp >= threshold_0
            genelist = data.gene_names[tmp]
            #for row in adata.varm['celltype_mean'].loc[genelist].iterrows():
            for row in data.genes_matrix['celltype_mean_scale'].loc[genelist].iterrows():
                #ax.errorbar(x=np.arange(len(row[1])), y=np.log1p(row[1]), yerr=np.log1p(row[1]), label=row[0], alpha = 0.5)
                ax.plot(row[1], label=row[0], alpha = 0.5)
            
            if summary_trend:
                tmp = data.genes_matrix['celltype_mean_scale'].loc[genelist]
                y_1 = tmp.quantile(0.5)
                y_0 = tmp.quantile(0.25)
                y_2 = tmp.quantile(0.75)
                from scipy.interpolate import make_interp_spline
                x_0 = np.arange(len(y_1))
                m_1 = make_interp_spline(x_0, y_1)
                m_0 = make_interp_spline(x_0, y_0)
                m_2 = make_interp_spline(x_0, y_2)
                xs = np.linspace(0, tmp.shape[1]-1, 500)
                ys_1 = m_1(xs)
                ys_0 = m_0(xs)
                ys_2 = m_2(xs)
                ax.plot(xs, ys_1, c='black')
                ax.fill(np.concatenate((xs,xs[::-1])), np.concatenate((ys_0, ys_2[::-1])), alpha=0.1)
            #ax.set_ylim(0,0.05)
            ax.set_ylabel('Log1p Exp')
            ax.set_xticks([x for x in range(data.genes_matrix['celltype_mean'].shape[1])])
            ax.set_xticklabels(list(data.genes_matrix['celltype_mean'].columns), rotation = 20, ha='right')
            ax.set_title(f'Cluster {x+1}')
            #ax.legend()
            ret.append(genelist)
        #return ret
        return fig
        
    def TVG_volcano_plot(self, stereo_exp_data, use_col, branch):
        """
        Use fuzzy C means cluster method to cluster genes based on 1-p_value of celltypes in branch
        :param stereo_exp_data: stereo_exp_data object to analysis
        :param use_col: the col in obs representing celltype or clustering
        :param branch: celltypes order in use_col
        :return: a volcano plot display time variable gene(TVG)
        """
        N_branch = len(branch)
        df = stereo_exp_data.genes.to_df()
        df['_log_pvalue'] = -np.log10(np.min(df[['less_pvalue', 'greater_pvalue']], axis=1))
        label2meam_exp = {}
        for x in branch:
            celllist = stereo_exp_data.cells.loc[stereo_exp_data.cells[use_col] == x,].index
            label2meam_exp[x] = np.array(
                np.mean(stereo_exp_data.sub_by_name(cell_name=celllist).exp_matrix, axis=0)).flatten()
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

    def paga_time_series_plot(self, stereo_exp_data, use_col='celltype', batch_col='sample_name',
                              groups=None, fig_height=10, fig_width=0, palette='tab20',
                              link_alpha=0.5, spot_size=1, dpt_col='dpt_pseudotime'):
        """
        spatial trajectory plot for paga in time_series multiple slice dataset
        :param stereo_exp_data: stereo_exp_data object of multiple slice
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
        import seaborn as sns
        from scipy import stats

        # 细胞类型的列表
        ct_list = list(stereo_exp_data.cells[use_col].astype('category').cat.categories)
        # 多篇数据存储分片信息的列表
        bc_list = list(stereo_exp_data.cells[batch_col].astype('category').cat.categories)
        # bc_list = ['E9.5_E1S1', 'E10.5_E1S1', 'E11.5_E1S1', 'E12.5_E1S1', 'E13.5_E1S1', 'E14.5_E1S1']

        # 生成每个细胞类型的颜色对应
        colors = sns.color_palette(palette, len(ct_list))
        ct2color = dict(zip(ct_list, colors))
        stereo_exp_data.cells['tmp'] = stereo_exp_data.cells[use_col].astype(str) + '|' + stereo_exp_data.cells[
            batch_col].astype(str)

        # 读取paga结果
        G = pd.DataFrame(stereo_exp_data.tl.result['paga']['connectivities_tree'].todense())
        G.index, G.columns = ct_list, ct_list
        # 转化为对角矩阵
        for i, x in enumerate(ct_list):
            for y in ct_list[i + 1:]:
                G[x][y] = G[y][x] = G[x][y] + G[y][x]
        # 利用伪时序计算结果推断方向
        if dpt_col in stereo_exp_data.cells:
            ct2dpt = {}

            for x, y in stereo_exp_data.cells.groupby(use_col):
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
            fig_width_0 = fig_height * np.ptp(stereo_exp_data.position[:, 0]) / np.ptp(
                stereo_exp_data.position[:, 1])
            fig_height_0 = 0
        if fig_width > 0:
            fig_height_0 = fig_width * np.ptp(stereo_exp_data.position[:, 1]) / np.ptp(
                stereo_exp_data.position[:, 0])
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
        for x in stereo_exp_data.cells.to_df().groupby('tmp'):
            tmp = stereo_exp_data.sub_by_name(cell_name=x[1].index).position
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
        for x in stereo_exp_data.cells.to_df().groupby(use_col):
            tmp = stereo_exp_data.sub_by_name(cell_name=x[1].index).position
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
                        p = self.bezierpath(min_vertex_right[edge0][1], max_vertex_right[edge0][1],
                                            min_vertex_left[edge1][1], max_vertex_right[edge1][1],
                                            min_vertex_right[edge0][0], min_vertex_left[edge1][0], True,
                                            col=ct2color[g], alpha=link_alpha)
                        ax.add_patch(p)

        # 显示时期的label
        for x in stereo_exp_data.cells.to_df().groupby(batch_col):
            tmp = stereo_exp_data.sub_by_name(cell_name=x[1].index).position
            ax.text(np.mean(tmp[:, 0]), 0 - np.min(tmp[:, 1]) + 1, s=x[0], c='black', fontsize=20, ha='center',
                    va='bottom')  # , bbox=dict(facecolor='red', alpha=0.1))

        plt.legend(ncol=int(np.sqrt(len(ct_list) / 6)) + 1)
        # plt.savefig('./plot/mouse_embyro_trajectory.png', dpi=300, bbox_inches='tight')
        # plt.show()
        return fig

    def time_series_tree_plot(self, ms_data, use_result = 'annotation', method = 'sankey', edges = None, dot_size_scale = 300, palette='tab20', ylabel_pos='left', fig_height=6, fig_width=6,):
        """
        a tree plot to display the cell amounts changes during time series, trajectory can be add to plot by edges.
        :param ms_data: ms_stereo_exp_data object of multiple slice
        :param use_result: the col in obs representing celltype or clustering
        :param method: choose from sankey and dot, choose the way to display
        :param edges: a parameter to add arrow to illustrate development trajectory. if edges=='page', use paga result, otherwise use a list of tuple of celltype pairs as father node and child node
        :param dot_size_scale: only used for method='dot', to adjust dot relatively size
        :param fig_height: height of figure
        :param fig_width: width of figure
        :param palette: color palette to paint different celltypes
        :param ylabel_pos: position to plot y labels
        :return: a fig object
        """
        # batch list
        import seaborn as sns
        import networkx as nx
        bc_list = ms_data.names
        # cell list
        ct_list = []
        for x in bc_list:
            ct_list.extend(ms_data[x].tl.result[use_result]['group'].unique())
        ct_list = np.unique(ct_list)
        
        #bc_list = ['E9.5_E1S1', 'E10.5_E1S1', 'E11.5_E1S1', 'E12.5_E1S1', 'E13.5_E1S1', 'E14.5_E1S1']
        colors = sns.color_palette('tab20',len(ct_list))
        ct2color = dict(zip(ct_list, colors))
        
        ret = defaultdict(dict)
        for t in bc_list:
            for x in ct_list:
                ret[x][t] = ms_data[t].tl.result[use_result].loc[ms_data[t].tl.result[use_result]['group']==x].shape[0]
        
        # order the celltype
        ret_df = pd.DataFrame(ret)
        ct2rank = {}
        
        ret_tmp = ret_df.copy()
        ret_tmp.index = np.arange(ret_tmp.shape[0])+1
        for x in ret_tmp:
            ct2rank[x] = np.mean(ret_tmp[x][ret_tmp[x] != 0].index)
            
        ct2rank=sorted(ct2rank.items(),key=lambda x:x[1],reverse=False)
        ret_df = ret_df[[x[0] for x in ct2rank]]
        
        hmax = np.max(np.max(ret_df))
        ret_df = ret_df/(2*hmax)
        
        if edges == 'paga':
            # 自动获取有paga结果的mss
            scope_flag = 0
            for scope in ms_data.mss:
                if 'paga' in ms_data.mss[scope]:
                    scope_flag = 1
                    break
            #print(scope_flag, scope)
            if scope_flag == 1:
                
                G = pd.DataFrame(ms_data.tl.result[scope]['paga']['connectivities_tree'].todense())
                G.index, G.columns = ct_list, ct_list
                #print(G)
                # 转化为对角矩阵
                #for i, x in enumerate(ct_list):
                #    for y in ct_list[i + 1:]:
                #        G[x][y] = G[y][x] = G[x][y] + G[y][x]
                G = nx.from_pandas_adjacency(G, nx.DiGraph)
                edges = G.edges()
            else:
                edges = None
                
        fig = plt.figure(figsize=(fig_width,fig_height))
        ax = fig.subplots()
        
        # 
        if method.lower() == 'sankey':
            pos_rec = defaultdict(list)
            i = 0
            yticks = []
            for x in ret_df:
                for j, y in enumerate(ret_df[x]):
                    ax.vlines(j, i-y, i+y, colors=ct2color[x] )
                    #p = self.bezierpath(j, True, col=ct2color[g], alpha=link_alpha)
                    #ax.add_patch(p)
                    if y != 0:
                        y1 = y
                        pos_rec[x].append((i-y1, i+y1, j))
                yticks.append(i)
                i+=1
            
            for x, y in pos_rec.items():
                for i in range(len(y)-1):
                    tmp0 = y[i]
                    tmp1 = y[i+1]
                    p = self.bezierpath(tmp0[0], tmp0[1], tmp1[0], tmp1[1], tmp0[2], tmp1[2], True,  alpha=0.5, col=ct2color[x])
                    ax.add_patch(p)
                    
            if edges != None:
                for node_a, node_b in edges:
                    pos_b = pos_rec[node_b]
                    pos_b.sort(key=lambda x:x[2])
                    tmp0 = pos_b[0]
                    x_list = [x[2] for x in pos_rec[node_a]]
                    candidate_list = [x for x in x_list if x < tmp0[2]]
                    if len(candidate_list) >0 :
                        x_a = np.max(candidate_list)
                    else:
                        x_a = np.min(x_list)
                    tmp1 = [x for x in pos_rec[node_a] if x[2] == x_a][0]
                    p = self.bezierpath(tmp0[0], tmp0[1], tmp1[0], tmp1[1], tmp0[2], tmp1[2], True,  alpha=0.3, col='grey')
                    ax.add_patch(p)
            
            xticks = np.arange(ret_df.shape[0])
            
        else:
            timepoint2x = {y:x+1 for x,y in enumerate(ret_df.index) }
            timepoint2x['root'] = 0
            ct2y = {y:x for x,y in enumerate(ret_df.columns)}
            xticks = np.arange(np.max(list(timepoint2x.values())))
            root_pos = [0, len(ct_list)/2]
            
            pos_rec = {}
            for y, ct in enumerate(ret_df.columns):
                time_list = ret_df.loc[ret_df[ct]!=0].index
                tmp_x = [timepoint2x[x] for x in time_list]
                pos_rec[ct] = [tmp_x, y] 
                ax.plot([np.max(tmp_x),np.min(tmp_x)], [y,y], c='black', zorder=0)
                ax.plot([0, 1, np.min(tmp_x)], [(len(ct_list)-1)/2, y, y], lw=0.2, c='black',zorder=0)
                ax.scatter(tmp_x, [y]*len(tmp_x), c=[ct2color[ct]]*len(tmp_x), s=ret_df.loc[time_list, ct]*dot_size_scale)
            
            ax.scatter(0, (len(ct_list)-1)/2, c='grey', s=1*dot_size_scale)
            
            xticks = np.arange(ret_df.shape[0])+1
            yticks = np.arange(ret_df.shape[1])
            if edges != None:
                for node_a, node_b in edges:
                    x_b = np.min(pos_rec[node_b][0])
                    y_b = pos_rec[node_b][1]
                    y_a = pos_rec[node_a][1]
                    x_a_list = pos_rec[node_a][0]
                    candidate_list = [x for x in x_a_list if x <x_b]
                    if len(candidate_list) > 0:
                        x_a = np.max(candidate_list)
                    else:
                        x_a = np.min(x_a_list)
                    ax.annotate('', (x_b,y_b), (x_a,y_a), arrowprops=dict(connectionstyle="arc3,rad=-0.2",ec='red', color='red',arrowstyle = 'simple', alpha=0.5))
            
        ax.set_xticks(xticks)
        ax.set_xticklabels(ret_df.index)
        ax.set_yticks(yticks)
        if ylabel_pos == 'right':
            ax.yaxis.tick_right()
        ax.set_yticklabels(ret_df.columns)
        return fig
    

    
    def ms_paga_time_series_plot(self, ms_data, use_col='celltype',
                              groups=None, fig_height=10, fig_width=0, palette='tab20',
                              link_alpha=0.5, spot_size=1, dpt_col='dpt_pseudotime'):
        """
        spatial trajectory plot for paga in time_series multiple slice dataset
        :param ms_data: ms_data object of multiple slice
        :param use_col: the col in obs representing celltype or clustering
        #param batch_col: the col in obs representing different slice of time series
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
        import seaborn as sns
        from scipy import stats
        
        
        # 多篇数据存储分片信息的列表
        bc_list = ms_data.names
        # 细胞类型的列表
        ct_list = ms_data.obs[use_col].astype('category').cat.categories
        
        # 获取多篇的spatial坐标
        position = []
        for x in bc_list:
            position.append( ms_data[x].position )
        position = np.vstack(position)
        
        obs_df = ms_data.obs.copy()
        batch_col = 'time'
        tmp_dic = {str(i):j for i,j in enumerate(bc_list)}
        obs_df[batch_col] =  [tmp_dic[x] for x in obs_df['batch']]
        
        # 生成每个细胞类型的颜色对应
        colors = sns.color_palette(palette, len(ct_list))
        ct2color = dict(zip(ct_list, colors))
        obs_df['tmp'] = obs_df[use_col].astype(str) + '|' + obs_df[batch_col].astype(str)

        # 读取paga结果
        scope_flag = 0
        for scope in ms_data.mss:
            if 'paga' in ms_data.mss[scope]:
                scope_flag = 1
                break
        #print(scope_flag, scope)
        if scope_flag == 1:
            G = pd.DataFrame(ms_data.tl.result[scope]['paga']['connectivities_tree'].todense())
            G.index, G.columns = ct_list, ct_list
            # 转化为对角矩阵
            for i, x in enumerate(ct_list):
                for y in ct_list[i + 1:]:
                    G[x][y] = G[y][x] = G[x][y] + G[y][x]
            # 利用伪时序计算结果推断方向
            if dpt_col in ms_data.tl.result[scope]:
                ct2dpt = {}
                #df_tmp = ms_data.obs.copy()
                obs_df[dpt_col] = ms_data.tl.result[scope][dpt_col]
    
                for x, y in obs_df.groupby(use_col):
                    ct2dpt[x] = y[dpt_col].to_numpy()
    
                from collections import defaultdict
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
            fig_width_0 = fig_height * np.ptp(position[:, 0]) / np.ptp(
                position[:, 1])
            fig_height_0 = 0
        if fig_width > 0:
            fig_height_0 = fig_width * np.ptp(position[:, 1]) / np.ptp(
                position[:, 0])
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
        obs_df['index'] = np.arange(obs_df.shape[0])
        for x in obs_df.groupby('tmp'):
            tmp = position[x[1]['index'], :]
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
        for x in obs_df.groupby(use_col):
            tmp = position[x[1]['index'], :]
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
                        p = self.bezierpath(min_vertex_right[edge0][1], max_vertex_right[edge0][1],
                                            min_vertex_left[edge1][1], max_vertex_right[edge1][1],
                                            min_vertex_right[edge0][0], min_vertex_left[edge1][0], True,
                                            col=ct2color[g], alpha=link_alpha)
                        ax.add_patch(p)

        # 显示时期的label
        for x in obs_df.groupby(batch_col):
            tmp = position[x[1]['index'], :]
            ax.text(np.mean(tmp[:, 0]), 0 - np.min(tmp[:, 1]) + 1, s=x[0], c='black', fontsize=20, ha='center',
                    va='bottom')  # , bbox=dict(facecolor='red', alpha=0.1))

        plt.legend(ncol=int(np.sqrt(len(ct_list) / 6)) + 1)
        # plt.savefig('./plot/mouse_embyro_trajectory.png', dpi=300, bbox_inches='tight')
        # plt.show()
        return fig