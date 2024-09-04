# python core modules
from typing import Union

# third party modules
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pyscenic.rss import regulon_specificity_scores

# modules in self project
from stereo.log_manager import logger
from stereo.plots.decorator import plot_scale
from stereo.plots.decorator import reorganize_coordinate
from stereo.plots.plot_base import PlotBase
from stereo.plots.scatter import base_scatter


class PlotRegulatoryNetwork(PlotBase):
    """
    Plot Gene Regulatory Networks related plots
    """

    # dotplot method for StereoExpData
    @staticmethod
    def _cal_percent_exp_df(
            exp_matrix: pd.DataFrame,
            cluster_meta: pd.DataFrame,
            regulon_genes: str,
            celltype: list,
            groupby: str = 'group',
            cell_label: str = 'bins',
            cutoff: float = 0
    ):
        """
        Expression percent
        cell numbers
        :param exp_matrix:
        :param cluster_meta:
        :param regulon_genes:
        :param celltype:
        :param cutoff:
        :return:
        """
        # which cells are in cluster X
        cells = cluster_meta[cluster_meta[groupby] == celltype][cell_label]
        ncells = set(exp_matrix.index).intersection(set(cells))
        # get expression data for cells
        # input genes in regulon Y
        # get expression data for regulon Y genes in cluster X cells
        g_ct_exp = exp_matrix.loc[list(ncells), regulon_genes]
        # count regulon totol expression value
        g_ct_exp['total'] = g_ct_exp.sum(axis=1)
        # count the number of genes which expressed in cluster X cells
        regulon_cell_num = g_ct_exp['total'][g_ct_exp['total'] > cutoff].count()
        total_cell_num = g_ct_exp.shape[0]
        if total_cell_num == 0:
            return 0
        else:
            reg_ct_percent = regulon_cell_num / total_cell_num
            reg_ct_avg_exp = np.mean(g_ct_exp['total'])
            return round(reg_ct_percent, 2), round(reg_ct_avg_exp, 2)

    def grn_dotplot(self,
                    cluster_res_key: str,
                    regulon_names: Union[str, list] = None,
                    celltypes: Union[str, list] = None,
                    groupby: str = 'group',
                    cell_label: str = 'bins',
                    network_res_key: str = 'regulatory_network_inference',
                    palette: str = 'Reds',
                    width: int = None,
                    height: int = None,
                    **kwargs):
        """
        Intuitive way of visualizing how feature expression changes across different
        identity classes (clusters). The size of the dot encodes the percentage of
        cells within a class, while the color encodes the AverageExpression level
        across all cells within a class (red is high).

        :param cluster_res_key: the key which specifies the clustering result in data.tl.result.
        :param regulon_names: the regulon which would be shown on plot, defaults to None.
            If set it to None, it will be set to all regulon.
            1) string: only one cluster.
            2) list: an array contains the regulon which would be shown.
        :param celltypes: the celltypes in cluster pairs which would be shown on plot, defaults to None.
            If set it to None, it will be set to all clusters.
            1) string: only one cluster.
            2) list: an array contains the clusters which would be shown.
        :param groupby: cell type label.
        :param cell_label: cell bin label.
        :param network_res_key: the key which specifies inference regulatory network result
             in data.tl.result, defaults to 'regulatory_network_inference'
        :param palette: Color theme, defaults to 'Reds'
        :param kwargs: features Input vector of features, or named list of feature vectors
        :param width: the figure width in pixels.
        :param height: the figure height in pixels.

        :return: matplotlib.figure
        """
        if network_res_key not in self.pipeline_res:
            logger.info(f"The result specified by {network_res_key} is not exists.")

        expr_matrix = self.stereo_exp_data.tl.raw.to_df()
        dot_data = {'cell type': [], 'regulons': [], 'percentage': [], 'avg exp': []}

        regulon_dict = self.pipeline_res[network_res_key]['regulons']

        if cluster_res_key in self.stereo_exp_data.cells._obs.columns:
            meta = pd.DataFrame({
                'bins': self.stereo_exp_data.cells.cell_name,
                'group': self.stereo_exp_data.cells._obs[cluster_res_key].tolist()
            })
        else:
            meta = self.pipeline_res[cluster_res_key]

        if celltypes is None:
            meta_new = meta.drop_duplicates(subset='group', inplace=False)
            celltypes = sorted(meta_new['group'])
        elif isinstance(celltypes, str) and celltypes.upper() == 'ALL':
            meta_new = meta.drop_duplicates(subset='group', inplace=False)
            celltypes = sorted(meta_new['group'])
        elif isinstance(celltypes, str) and celltypes.upper() != 'ALL':
            celltypes = [celltypes]

        if regulon_names is None:
            regulon_names = regulon_dict.keys()
        elif isinstance(regulon_names, str) and regulon_names.upper() == 'ALL':
            regulon_names = regulon_dict.keys()
        elif isinstance(regulon_names, str) and regulon_names.upper() != 'ALL':
            regulon_names = [regulon_names]

        for reg in regulon_names:
            if '(+)' not in reg:
                reg = reg + '(+)'
            target_genes = regulon_dict[f'{reg}']
            for ct in celltypes:
                reg_ct_percent, reg_ct_avg_exp = PlotRegulatoryNetwork._cal_percent_exp_df(exp_matrix=expr_matrix,
                                                                                           cluster_meta=meta,
                                                                                           regulon_genes=target_genes,
                                                                                           celltype=ct,
                                                                                           groupby=groupby,
                                                                                           cell_label=cell_label)

                dot_data['regulons'].append(reg)
                dot_data['cell type'].append(ct)
                dot_data['percentage'].append(reg_ct_percent)
                dot_data['avg exp'].append(reg_ct_avg_exp)

        dot_df = pd.DataFrame(dot_data)

        # plotting
        ncols = len(dot_df['regulons'].unique())
        nrows = len(dot_df['cell type'].unique())

        if width is None or height is None:
            width, height = int(5 + max(3, ncols * 0.8)), int(3 + max(5, nrows * 0.5))
        else:
            width = width / 100 if width >= 100 else int(5 + max(3, ncols * 0.8))
            height = height / 100 if height >= 100 else int(3 + max(5, nrows * 0.5))

        fig, ax = plt.subplots(figsize=(width, height))
        sns.scatterplot(data=dot_df, size='percentage', hue='avg exp', x='regulons', y='cell type', sizes=(100, 300),
                        marker='o', palette=palette, legend='auto', ax=ax, **kwargs)
        ax.legend(fontsize=12, frameon=False, ncol=1, loc=(1.02, 0))
        ax.tick_params(axis='x', labelsize=12, labelrotation=90)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xlabel('Regulon')
        ax.set_ylabel('Cell type')
        return fig

    def auc_heatmap(
            self,
            network_res_key: str = 'regulatory_network_inference',
            width: int = 8,
            height: int = 8,
            pivot_kws: dict = None,
            method: str = 'average',
            metric: str = 'euclidean',
            z_score: int = None,
            standard_scale: int = None,
            cbar_kws: dict = None,
            row_cluster: bool = True,
            col_cluster: bool = True,
            row_linkage: np.ndarray = None,
            col_linkage: np.ndarray = None,
            row_colors: Union[list, pd.DataFrame, pd.Series] = None,
            col_colors: Union[list, pd.DataFrame, pd.Series] = None,
            mask: Union[list, pd.DataFrame] = None,
            dendrogram_ratio: float = (.1, .2),
            colors_ratio: float = 0.03,
            cbar_pos: tuple = (-.05, .2, .03, .4),
            tree_kws: dict = None
    ):
        """
        Plot heatmap for auc value for regulons

        Parameters
        ----------
        network_res_key: str, optional
            the key which specifies inference regulatory network result in data.tl.result, defaults to 'regulatory_network_inference' # noqa
        height: int, optional
            height of drawing
        width: int, optional
            width of drawing
        pivot_kws : dict, optional
            If `data` is a tidy dataframe, can provide keyword arguments for
            pivot to create a rectangular dataframe.
        method : str, optional
            Linkage method to use for calculating clusters. See
            :func:`scipy.cluster.hierarchy.linkage` documentation for more
            information.
        metric : str, optional
            Distance metric to use for the data. See
            :func:`scipy.spatial.distance.pdist` documentation for more options.
            To use different metrics (or methods) for rows and columns, you may
            construct each linkage matrix yourself and provide them as
            `{row,col}_linkage`.
        z_score : int or None, optional
            Either 0 (rows) or 1 (columns). Whether or not to calculate z-scores
            for the rows or the columns. Z scores are: z = (x - mean)/std, so
            values in each row (column) will get the mean of the row (column)
            subtracted, then divided by the standard deviation of the row (column).
            This ensures that each row (column) has mean of 0 and variance of 1.
        standard_scale : int or None, optional
            Either 0 (rows) or 1 (columns). Whether or not to standardize that
            dimension, meaning for each row or column, subtract the minimum and
            divide each by its maximum.
        cbar_kws : dict, optional
            Keyword arguments to pass to `cbar_kws` in :func:`heatmap`, e.g. to
            add a label to the colorbar.
        {row,col}_cluster : bool, optional
            If ``True``, cluster the {rows, columns}.
        {row,col}_linkage : :class:`numpy.ndarray`, optional
            Precomputed linkage matrix for the rows or columns. See
            :func:`scipy.cluster.hierarchy.linkage` for specific formats.
        {row,col}_colors : list-like or pandas DataFrame/Series, optional
            List of colors to label for either the rows or columns. Useful to evaluate
            whether samples within a group are clustered together. Can use nested lists or
            DataFrame for multiple color levels of labeling. If given as a
            :class:`pandas.DataFrame` or :class:`pandas.Series`, labels for the colors are
            extracted from the DataFrames column names or from the name of the Series.
            DataFrame/Series colors are also matched to the data by their index, ensuring
            colors are drawn in the correct order.
        mask : bool array or DataFrame, optional
            If passed, data will not be shown in cells where `mask` is True.
            Cells with missing values are automatically masked. Only used for
            visualizing, not for calculating.
        {dendrogram,colors}_ratio : float, or pair of floats, optional
            Proportion of the figure size devoted to the two marginal elements. If
            a pair is given, they correspond to (row, col) ratios.
        cbar_pos : tuple of (left, bottom, width, height), optional
            Position of the colorbar axes in the figure. Setting to ``None`` will
            disable the colorbar.
        tree_kws : dict, optional
            Parameters for the :class:`matplotlib.collections.LineCollection`
            that is used to plot the lines of the dendrogram tree.

        :return: matplotlib.figure
        """
        logger.info('Generating auc heatmap plot')

        if network_res_key not in self.pipeline_res:
            logger.info(f"The result specified by {network_res_key} is not exists.")

        g = sns.clustermap(
            self.pipeline_res[network_res_key]['auc_matrix'],
            pivot_kws=pivot_kws,
            method=method,
            metric=metric,
            z_score=z_score,
            standard_scale=standard_scale,
            cbar_kws=cbar_kws,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            col_linkage=col_linkage,
            row_linkage=row_linkage,
            row_colors=row_colors,
            col_colors=col_colors,
            mask=mask,
            colors_ratio=colors_ratio,
            tree_kws=tree_kws,
            figsize=(width, height),
            dendrogram_ratio=dendrogram_ratio,
            cbar_pos=cbar_pos,
        )

        return g.figure

    @plot_scale
    @reorganize_coordinate
    def spatial_scatter_by_regulon(
            self,
            network_res_key: str = 'regulatory_network_inference',
            reg_name: str = None,
            dot_size: int = None,
            palette: str = 'CET_L4',
            width: int = None,
            height: int = None,
            vmin: float = None,
            vmax: float = None,
            **kwargs):
        """
        Plot genes of one regulon on a 2D map

        :param network_res_key: the key which specifies inference regulatory network result
             in data.tl.result, defaults to 'regulatory_network_inference'
        :param reg_name: specify the regulon you want to draw, defaults to None, if none, will select randomly.
        :param dot_size: marker size, defaults to None
        :param palette: Color theme, defaults to 'CET_L4'
        :param height: height of drawing
        :param width: width of drawing
        :param vmin: The value representing the lower limit of the color scale. Values smaller than vmin are plotted with the same color as vmin.
        :param vmax: The value representing the lower limit of the color scale. Values smaller than vmin are plotted with the same color as vmin.
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

        :return: matplotlib.figure
        """  # noqa
        logger.info('Please adjust the dot_size to prevent dots from covering each other')

        if network_res_key not in self.pipeline_res:
            logger.info(f"The result specified by {network_res_key} is not exists.")

        if reg_name is None:
            regulon_dict = self.pipeline_res[network_res_key]['regulons']
            reg_name = list(regulon_dict.keys())[0]
        elif '(+)' not in reg_name:
            reg_name = reg_name + '(+)'
        cell_coor = self.stereo_exp_data.position
        # prepare plotting data
        auc_zscore = cal_zscore(self.pipeline_res[network_res_key]['auc_matrix'][reg_name])
        # sort data points by zscore (low to high), because first dot will be covered by latter dots
        df = pd.DataFrame({'x': cell_coor[:, 0], 'y': cell_coor[:, 1], 'auc_zscore': auc_zscore})
        df.sort_values(by=['auc_zscore'], inplace=True)
        # plot cell/bin dot, x y coor
        if 'color_bar_reverse' in kwargs:
            color_bar_reverse = kwargs['color_bar_reverse']
            del kwargs['color_bar_reverse']
        else:
            color_bar_reverse = False

        fig = base_scatter(
            x=df['x'],
            y=df['y'],
            hue=df['auc_zscore'],
            title=reg_name,
            x_label='spatial1',
            y_label='spatial2',
            dot_size=dot_size,
            palette=palette,
            color_bar=True,
            color_bar_reverse=color_bar_reverse,
            width=width,
            height=height,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )

        return fig

    @staticmethod
    def plot_2d_reg_h5ad(data: anndata.AnnData, pos_label, auc_mtx, reg_name: str, **kwargs):
        """
        Plot genes of one regulon on a 2D map
        :param pos_label:
        :param data:
        :param auc_mtx:
        :param reg_name:
        :return:

        Example:
            plot_2d_reg_h5ad(data, 'spatial', auc_mtx, 'Zfp354c')
        """
        if '(+)' not in reg_name:
            reg_name = reg_name + '(+)'
        cell_coor = data.obsm[pos_label]
        auc_zscore = cal_zscore(auc_mtx)
        # prepare plotting data
        sub_zscore = auc_zscore[reg_name]
        # sort data points by zscore (low to high), because first dot will be covered by latter dots
        zorder = np.argsort(sub_zscore[reg_name].values)
        # plot cell/bin dot, x y coor
        sc = plt.scatter(cell_coor[:, 0][zorder], cell_coor[:, 1][zorder], c=sub_zscore[reg_name][zorder], marker='.',
                         edgecolors='none', cmap='plasma', lw=0, **kwargs)
        plt.box(False)
        plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        plt.savefig(f'{reg_name.split("(")[0]}.png')
        plt.close()

    def auc_heatmap_by_group(
            self,
            network_res_key: str = 'regulatory_network_inference',
            cluster_res_key: str = None,
            top_n_feature: int = 5,
            width: int = 18,
            height: int = 28,
            pivot_kws: dict = None,
            method: str = 'average',
            metric: str = 'euclidean',
            z_score: int = None,
            standard_scale: int = None,
            cbar_kws: dict = None,
            row_cluster: bool = True,
            col_cluster: bool = True,
            row_linkage: np.ndarray = None,
            col_linkage: np.ndarray = None,
            col_colors: Union[list, pd.DataFrame, pd.Series] = None,
            mask: Union[list, pd.DataFrame] = None,
            dendrogram_ratio: float = (.1, .2),
            colors_ratio: float = 0.03,
            cbar_pos: tuple = (-.05, .2, .03, .4),
            tree_kws: dict = None
    ):
        """
        Plot heatmap for Regulon specificity scores (RSS) value

        Parameters
        ----------
        network_res_key: str, optional
            the key which specifies inference regulatory network result in data.tl.result, defaults to 'regulatory_network_inference' # noqa
        cluster_res_key:  str, optional
            the key which specifies the clustering result in data.tl.result, defaults to None
        top_n_feature: int, optional
            get the top n feature
        height: int, optional
            height of drawing
        width: int, optional
            width of drawing
        pivot_kws : dict, optional
            If `data` is a tidy dataframe, can provide keyword arguments for
            pivot to create a rectangular dataframe.
        method : str, optional
            Linkage method to use for calculating clusters. See
            :func:`scipy.cluster.hierarchy.linkage` documentation for more
            information.
        metric : str, optional
            Distance metric to use for the data. See
            :func:`scipy.spatial.distance.pdist` documentation for more options.
            To use different metrics (or methods) for rows and columns, you may
            construct each linkage matrix yourself and provide them as
            `{row,col}_linkage`.
        z_score : int or None, optional
            Either 0 (rows) or 1 (columns). Whether or not to calculate z-scores
            for the rows or the columns. Z scores are: z = (x - mean)/std, so
            values in each row (column) will get the mean of the row (column)
            subtracted, then divided by the standard deviation of the row (column).
            This ensures that each row (column) has mean of 0 and variance of 1.
        standard_scale : int or None, optional
            Either 0 (rows) or 1 (columns). Whether or not to standardize that
            dimension, meaning for each row or column, subtract the minimum and
            divide each by its maximum.
        cbar_kws : dict, optional
            Keyword arguments to pass to `cbar_kws` in :func:`heatmap`, e.g. to
            add a label to the colorbar.
        {row,col}_cluster : bool, optional
            If ``True``, cluster the {rows, columns}.
        {row,col}_linkage : :class:`numpy.ndarray`, optional
            Precomputed linkage matrix for the rows or columns. See
            :func:`scipy.cluster.hierarchy.linkage` for specific formats.
        {row,col}_colors : list-like or pandas DataFrame/Series, optional
            List of colors to label for either the rows or columns. Useful to evaluate
            whether samples within a group are clustered together. Can use nested lists or
            DataFrame for multiple color levels of labeling. If given as a
            :class:`pandas.DataFrame` or :class:`pandas.Series`, labels for the colors are
            extracted from the DataFrames column names or from the name of the Series.
            DataFrame/Series colors are also matched to the data by their index, ensuring
            colors are drawn in the correct order.
        mask : bool array or DataFrame, optional
            If passed, data will not be shown in cells where `mask` is True.
            Cells with missing values are automatically masked. Only used for
            visualizing, not for calculating.
        {dendrogram,colors}_ratio : float, or pair of floats, optional
            Proportion of the figure size devoted to the two marginal elements. If
            a pair is given, they correspond to (row, col) ratios.
        cbar_pos : tuple of (left, bottom, width, height), optional
            Position of the colorbar axes in the figure. Setting to ``None`` will
            disable the colorbar.
        tree_kws : dict, optional
            Parameters for the :class:`matplotlib.collections.LineCollection`
            that is used to plot the lines of the dendrogram tree.

        :return: matplotlib.figure
        """
        if network_res_key not in self.pipeline_res:
            logger.info(f"The result specified by {network_res_key} is not exists.")
        elif cluster_res_key not in self.pipeline_res:
            logger.info(f"The result specified by {cluster_res_key} is not exists.")

        auc_mtx = self.pipeline_res[network_res_key]['auc_matrix']

        if cluster_res_key in self.stereo_exp_data.cells._obs.columns:
            meta = pd.DataFrame({
                'bins': self.stereo_exp_data.cells.cell_name,
                'group': self.stereo_exp_data.cells._obs[cluster_res_key].tolist()
            })
        else:
            meta = self.pipeline_res[cluster_res_key].copy(deep=True)

        # Regulon specificity scores (RSS) across predicted cell types
        rss_cellType = regulon_specificity_scores(auc_mtx, meta['group'])
        # rss_cellType.to_csv('regulon_specificity_scores.txt')
        # Select the top 5 regulon_list from each cell type
        cats = sorted(list(set(meta['group'])))
        topreg = []
        for i, c in enumerate(cats):
            topreg.extend(
                list(rss_cellType.T[c].sort_values(ascending=False)[:top_n_feature].index)
            )
        topreg = list(set(topreg))

        # plot z-score
        auc_zscore = cal_zscore(auc_mtx)
        sns.set(font_scale=1.2)
        # set group color
        lut = dict(zip(meta['group'].unique(), ncolors(len(meta['group'].unique()))))
        row_colors = meta['group'].map(lut)
        meta['group'] = row_colors

        g = sns.clustermap(
            auc_zscore[topreg],
            row_colors=meta.set_index(['bins']),
            figsize=(width, height),
            pivot_kws=pivot_kws,
            method=method,
            metric=metric,
            z_score=z_score,
            standard_scale=standard_scale,
            cbar_kws=cbar_kws,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            col_linkage=col_linkage,
            row_linkage=row_linkage,
            col_colors=col_colors,
            mask=mask,
            colors_ratio=colors_ratio,
            tree_kws=tree_kws,
            dendrogram_ratio=dendrogram_ratio,
            cbar_pos=cbar_pos,
        )

        return g.figure

    def spatial_scatter_by_regulon_3D(
            self,
            network_res_key: str = 'regulatory_network_inference',
            reg_name: str = None,
            # fn: str = None,
            view_vertical: int = 0,
            view_horizontal: int = 0,
            show_axis: bool = False,
            width: int = None,
            height: int = None,
            **kwargs):
        """Plot genes of one regulon on a 3D map

        :param network_res_key: the key which specifies inference regulatory network result in data.tl.result, defaults to 'regulatory_network_inference' # noqa
        :param reg_name: specify the regulon you want to draw, defaults to None, if none, will select randomly.
        :param fn: specify the file name of the output figure, defaults to None, if none, will use regulon name.
        :param view_vertical: vertical angle to view to the 3D object
        :param view_horizontal: horizontal angle to view the 3D object

        Example:
            data.plt.plot_3d_reg('regulatory_network_inference', 'Zfp354c', view_vertical=30, view_horizontal=-30)
        """

        if reg_name is None:
            regulon_dict = self.pipeline_res[network_res_key]['regulons']
            reg_name = list(regulon_dict.keys())[0]
        elif '(+)' not in reg_name:
            reg_name = reg_name + '(+)'

        # if fn is None:
        #     fn = f'{reg_name.strip("(+)")}.pdf'

        # prepare plotting data
        arr2 = self.stereo_exp_data.position_z
        position_3D = np.concatenate((self.stereo_exp_data.position, arr2), axis=1)

        cell_coor = position_3D
        assert cell_coor.shape[1] == 3  # TODO: ensure position is 3D

        auc_mtx = self.pipeline_res[network_res_key]['auc_matrix']

        auc_zscore = cal_zscore(auc_mtx)
        sub_zscore = auc_zscore[reg_name]

        # plot
        fig = plt.figure()
        if width is not None:
            fig.set_figwidth(width)
        if height is not None:
            fig.set_figheight(height)
        # ax = Axes3D(fig)
        ax: Axes3D = fig.add_subplot(projection='3d')
        sc = ax.scatter(cell_coor[:, 0],
                        cell_coor[:, 1],
                        cell_coor[:, 2],
                        c=sub_zscore,
                        marker='.',
                        edgecolors='none',
                        cmap='plasma',
                        lw=0, **kwargs)
        # set view angle
        ax.view_init(view_vertical, view_horizontal)
        # scale axis
        # xlen = cell_coor[:, 0].max() - cell_coor[:, 0].min()
        # ylen = cell_coor[:, 1].max() - cell_coor[:, 1].min()
        # zlen = cell_coor[:, 2].max() - cell_coor[:, 2].min()
        # yscale = ylen / xlen
        # zscale = zlen / xlen
        # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, yscale, zscale, 1]))

        if not show_axis:
            plt.box(False)
            plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        # plt.savefig(fn, format='pdf')
        return fig


def get_n_hls_colors(num):
    import random
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10  # noqa
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def ncolors(num):
    import colorsys
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append(rgb_to_hex(r, g, b))

    return rgb_colors


def cal_zscore(auc_mtx: pd.DataFrame) -> pd.DataFrame:
    """
    calculate z-score for each gene among cells
    :param auc_mtx:
    :return:
    """
    func = lambda x: (x - x.mean()) / x.std(ddof=0)  # noqa
    auc_zscore = auc_mtx.transform(func, axis=0)
    return auc_zscore
