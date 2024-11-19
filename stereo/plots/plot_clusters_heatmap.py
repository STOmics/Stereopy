from typing import Optional, Sequence, Literal

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from stereo.log_manager import logger
from stereo.plots.plot_base import PlotBase
from stereo.utils.pipeline_utils import cell_cluster_to_gene_exp_cluster
from ._plot_basic.heatmap_plt import heatmap


class ClustersGenesHeatmap(PlotBase):
    __category_width = 0.37
    __category_height = 0.35
    __legend_width = 2
    __dendrogram_height = 0.6
    __title_font_size = 8

    def clusters_genes_heatmap(
            self,
            cluster_res_key: str,
            dendrogram_res_key: Optional[str] = None,
            topn: Optional[int] = 5,
            gene_names: Optional[Sequence[str]] = None,
            expression_kind: Literal['mean', 'sum'] = 'mean',
            groups: Optional[Sequence[str]] = None,
            width: int = None,
            height: int = None,
            colormap: str = 'Greens',
            standard_scale: str = 'gene'
    ):
        """

        Heatmap representing mean expression of genes on each cell cluster.

        :param cluster_res_key: the key to get cluster result.
        :param dendrogram_res_key: the key to get dendrogram result, defaults to None to avoid show dendrogram on plot.
        :param topn: select `topn` expressed genes in each cluster, defaults to 5, ignored if `gene_names` is not None,
                    the number of genes shown in plot may be more than `topn` because the `topn` genes in each cluster are not the same.
        :param gene_names: a list of genes to show, defaults to None to show all genes.
        :param expression_kind: the kind of expression to show, 'mean' or 'sum', defaults to 'mean'.
        :param groups: a list of cell clusters to show, defaults to None to show all cell clusters.
        :param width: the figure width in pixels, defaults to None.
        :param height: the figure height in pixels, defaults to None.
        :param colormap: colormap used on plot, defaults to 'Greens'
        :param standard_scale: Whether or not to standardize that dimension between 0 and 1,
                                meaning for each gene or cluster,
                                subtract the minimum and divide each by its maximum, defaults to 'gene'

        """
        if cluster_res_key not in self.pipeline_res:
            raise KeyError(f"Can not find the cluster result in data.tl.result by key {cluster_res_key}")

        drg_res = None
        if dendrogram_res_key is not None:
            if dendrogram_res_key not in self.pipeline_res:
                raise KeyError(f"Can not find the dendrogram result in data.tl.result by key {dendrogram_res_key}")
            else:
                drg_res = self.pipeline_res[dendrogram_res_key]
                if cluster_res_key != drg_res['cluster_res_key'][0]:
                    raise KeyError(
                        f'The cluster result used in dendrogram may not be the same as that specified '
                        f'by key {cluster_res_key}')

        if gene_names is not None:
            topn = None
        
        if topn is None:
            if gene_names is None:
                gene_names = self.stereo_exp_data.gene_names
            else:
                if isinstance(gene_names, str):
                    gene_names = np.array([gene_names], dtype='U')
                elif not isinstance(gene_names, np.ndarray):
                    gene_names = np.array(gene_names, dtype='U')

            if len(gene_names) == 0:
                return None

        if groups is None or drg_res is not None:
            cluster_res: pd.DataFrame = self.pipeline_res[cluster_res_key]
            cluster_res['group'] = cluster_res['group'].astype('category')
            group_codes = cluster_res['group'].cat.categories.to_numpy()
            groups = None
            if drg_res is not None:
                categories_idx_ordered = drg_res['categories_idx_ordered']
                group_codes = group_codes[categories_idx_ordered]
        else:
            group_codes = groups
            if isinstance(group_codes, str):
                group_codes = np.array([group_codes], dtype='U')
            elif not isinstance(group_codes, np.ndarray):
                group_codes = np.array(group_codes, dtype='U')

        if topn is None:
            genes_expression: pd.DataFrame = cell_cluster_to_gene_exp_cluster(
                self.stereo_exp_data,
                cluster_res_key,
                groups=groups,
                genes=gene_names,
                kind='mean'
            )
        else:
            genes_expression = cell_cluster_to_gene_exp_cluster(
                self.stereo_exp_data,
                cluster_res_key,
                groups=groups,
                kind=expression_kind
            )
            gene_names = []
            for c in genes_expression.columns:
                gene_names.extend(genes_expression[c].sort_values(ascending=False).index[:topn].tolist())
            gene_names = np.unique(gene_names)
            genes_expression = genes_expression.loc[gene_names]
        
        genes_expression = genes_expression[group_codes]

        if standard_scale == 'cluster':
            genes_expression -= genes_expression.min(0)
            genes_expression = (genes_expression / genes_expression.max(0)).fillna(0)
        elif standard_scale == 'gene':
            genes_expression = genes_expression.sub(genes_expression.min(1), axis=0)
            genes_expression = genes_expression.div(genes_expression.max(1), axis=0).fillna(0)
        elif standard_scale is None:
            pass
        else:
            logger.warning('Unknown type for standard_scale, ignored')

        if width is None:
            if len(group_codes) < 10:
                main_area_width = self.__category_width * 10
            else:
                main_area_width = self.__category_width * len(group_codes)
        else:
            if width < self.__legend_width:
                width = self.__category_width * 10 + self.__legend_width
            main_area_width = width - self.__legend_width
        if height is None:
            if len(gene_names) < 10:
                main_area_height = self.__category_height * 10
            else:
                main_area_height = self.__category_height * len(gene_names)
        else:
            if height < self.__dendrogram_height:
                height = self.__category_height * 10 + self.__dendrogram_height
            main_area_height = height - self.__dendrogram_height
        width_ratios = [main_area_width, self.__legend_width]
        height_ratios = [self.__dendrogram_height + main_area_height]
        width = sum(width_ratios)
        height = sum(height_ratios)
        fig = plt.figure(figsize=(width, height))

        axs = gridspec.GridSpec(
            nrows=1,
            ncols=2,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=(0.15 / main_area_width),
            hspace=0
        )

        axs_main = gridspec.GridSpecFromSubplotSpec(
            nrows=2,
            ncols=1,
            width_ratios=[main_area_width],
            height_ratios=[self.__dendrogram_height, main_area_height],
            wspace=0,
            hspace=0,
            subplot_spec=axs[0, 0]
        )

        axs_on_right = gridspec.GridSpecFromSubplotSpec(
            nrows=2,
            ncols=1,
            height_ratios=[0.95, 0.05],
            subplot_spec=axs[0, 1],
            hspace=0.1
        )

        ax_heatmap = fig.add_subplot(axs_main[1, 0])
        ax_colorbar = fig.add_subplot(axs_on_right[1, 0])
        heatmap(
            genes_expression,
            ax=ax_heatmap,
            plot_colorbar=True,
            colorbar_ax=ax_colorbar,
            cmap=colormap,
            colorbar_orientation='horizontal',
            colorbar_ticklocation='bottom',
            colorbar_title=f'{expression_kind.capitalize()} expression in group',
            show_xaxis=True,
            show_yaxis=True,
        )

        if drg_res is not None:
            from .plot_dendrogram import PlotDendrogram
            ax_drg = fig.add_subplot(axs_main[0, 0], sharex=ax_heatmap)
            plt_drg = PlotDendrogram(self.stereo_exp_data, self.pipeline_res)
            plt_drg.dendrogram(
                orientation='top',
                ax=ax_drg,
                res_key=dendrogram_res_key,
                ticks=list(range(len(group_codes)))
            )
            ax_drg.set_axis_off()

        return fig
