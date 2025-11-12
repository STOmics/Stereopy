from typing import Literal

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .plot_base import PlotBase


class PlotGenesInPseudotime(PlotBase):
    __DEFAULT_FIGURE_WIDTH = 17
    __DEFAULT_AXES_HEIGHT = 3
    __FONT_SIZE = 10

    def plot_genes_in_pseudotime(
            self,
            marker_genes_res_key: str = 'marker_genes',
            group: str = None,
            pseudotime_key: str = 'dpt_pseudotime',
            sort_by: Literal[
                'scores',
                'pvalues',
                'pvalues_adj',
                'log2fc',
                'pct',
                'pct_rest'
            ] = 'scores',
            topn: int = 5,
            cmap: str = 'plasma',
            width: int = None,
            height: int = None,
            size: int = 30,
            marker: str = '.'
    ):
        """
        Distribution of expression count of marker genes along with pseudotime

        :param marker_genes_res_key: Specifies the key to get result of `find_marker_genes` , defaults to 'marker_genes'
        :param group: Sepcifies the cell type to plot, defaults to None
        :param sort_by: Get top N marker genes in this order, defaults to 'scores'
        :param topn: Get top N marker genes, defaults to 5
        :param cmap: Color map, definded in `matplotlib <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_, defaults to 'plasma'
        :param width: The figure width in pixels, defaults to None
        :param height: The figure height in pixels, defaults to None
        :param size: The size of markers in plot, defaults to 30
        :param marker: The style of markers in plot, defaults to '.'

        """  # noqa
        if group is None:
            raise ValueError('Must specify the group to plot.')

        if marker_genes_res_key not in self.pipeline_res:
            raise KeyError(f"Can not find 'find_marker_genes' result by key '{marker_genes_res_key}'.")

        # if 'dpt_pseudotime' not in self.pipeline_res:
        #     raise KeyError("Can not find pseudotime infomation.")

        if pseudotime_key not in self.stereo_exp_data.cells and pseudotime_key not in self.pipeline_res:
            raise KeyError("Can not find pseudotime infomation.")

        marker_genes_res: dict = self.pipeline_res[marker_genes_res_key]

        marker_genes_res_used = None
        for key in marker_genes_res.keys():
            if 'vs' not in key:
                continue
            group_in_key = key.split('.vs.')[0]
            if group == group_in_key:
                marker_genes_res_used: pd.DataFrame = marker_genes_res[key]
                break

        if marker_genes_res_used is None:
            raise ValueError(f"Can not find the 'find_marker_genes' result related to group '{group}'.")

        if sort_by not in marker_genes_res_used.columns:
            raise ValueError(f"The key '{sort_by}' used to sort Can not be found in 'find_marker_genes' result.")

        marker_genes_res_used.sort_values(by=sort_by, ascending=False, inplace=True)
        topn = min(topn, marker_genes_res_used.shape[0])
        genes = marker_genes_res_used[0:topn]['genes'].to_numpy()

        # ptime = self.pipeline_res['dpt_pseudotime']
        if pseudotime_key in self.stereo_exp_data.cells:
            ptime = self.stereo_exp_data.cells[pseudotime_key].to_numpy()
        else:
            ptime = self.pipeline_res[pseudotime_key]

        if width is None:
            width = self.__DEFAULT_FIGURE_WIDTH

        if height is None:
            height = self.__DEFAULT_AXES_HEIGHT * len(genes)

        fig, axes = plt.subplots(nrows=len(genes), ncols=1, sharex=True, figsize=(width, height))

        norm = Normalize(vmin=ptime.min(), vmax=ptime.max(), clip=False)

        x_label = 'Pseudotime'
        y_label = 'Expression'
        hue_label = 'Pseudotime'

        for i, gene in enumerate(genes):
            flag = self.stereo_exp_data.gene_names == gene
            gene_exp = self.stereo_exp_data.exp_matrix[:, flag]
            if self.stereo_exp_data.issparse():
                gene_exp = gene_exp.toarray()
            gene_exp = gene_exp.reshape(-1)
            plot_data = pd.DataFrame({
                x_label: ptime,
                y_label: gene_exp,
                'hue': ptime,
                'size': size
            })
            ax: Axes = axes[i]
            ax.set_title(gene, fontdict=dict(fontsize=self.__FONT_SIZE), loc='center')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_visible(False)
            ax.set_xlabel(x_label, fontdict=dict(fontsize=self.__FONT_SIZE), loc='center')
            ax.set_ylabel(y_label, fontdict=dict(fontsize=self.__FONT_SIZE), loc='center')
            sns.scatterplot(
                data=plot_data,
                x=x_label,
                y=y_label,
                hue='hue',
                hue_norm=norm,
                palette=cmap,
                size='size',
                sizes=(size, size),
                marker=marker,
                ax=ax,
                legend=False
            )
        axes[-1].spines['bottom'].set_visible(True)
        axes[-1].xaxis.set_visible(True)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes, shrink=0.2, orientation='vertical')
        cb.ax.set_title(hue_label, fontdict=dict(fontsize=self.__FONT_SIZE), loc='left')
        return fig
