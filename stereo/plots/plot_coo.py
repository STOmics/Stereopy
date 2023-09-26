import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from natsort import natsorted

from stereo.plots.plot_base import PlotBase


class PlotCoOccurrence(PlotBase):

    def _close_axis(self, axes):
        if isinstance(axes, Axes):
            axes.set_axis_off()
        else:
            for ax in axes:
                self._close_axis(ax)

    def co_occurrence_plot(
            self,
            groups=[],
            width=None,
            height=None,
            res_key='co_occurrence'
    ):
        '''
        Visualize the co-occurence by line plot; each subplot represent a celltype, each line in subplot represent the
        co-occurence value of the pairwise celltype as the distance range grow.

        :param groups: Choose a few cluster to plot, plot all clusters by default.
        :param width: the figure width.
        :param height: the figure height.
        :param res_key: The key to store co-occurence result in data.tl.result

        '''
        if len(groups) == 0:
            groups = natsorted(self.pipeline_res[res_key].keys())
        else:
            groups = natsorted(groups)
        nrow = int(np.sqrt(len(groups)))
        ncol = np.ceil(len(groups) / nrow).astype(int)
        if width is None:
            width = 5 * ncol
        if height is None:
            height = 5 * nrow
        fig = plt.figure(figsize=(width, height))
        axs = fig.subplots(nrow, ncol)
        self._close_axis(axs)
        for i, g in enumerate(groups):
            interest = self.pipeline_res[res_key][g]
            if nrow == 1:
                if ncol == 1:
                    ax = axs
                else:
                    ax = axs[i]
            else:
                ax = axs[int(i / ncol)][(i % ncol)]
            ax.plot(interest, label=interest.columns)
            ax.set_title(g)
            ax.set_axis_on()
            ax.legend(fontsize=7, ncol=max(1, nrow - 1), loc='upper right')
        return fig

    def co_occurrence_heatmap(
            self,
            cluster_res_key,
            dist_min=0,
            dist_max=10000,
            width=None,
            height=None,
            res_key='co_occurrence'
    ):
        '''
        Visualize the co-occurence by heatmap; each subplot represent a certain distance, each heatmap in subplot represent the 
        co-occurence value of the pairwise celltype.

        :param cluster_res_key: The key of the cluster or annotation result of cells stored in data.tl.result which ought to be equal 
                        to cells in length.
        :param dist_min: Minimum distance interested, threshold between (dist_min,dist_max) will be plot
        :param dist_max: Maximum distance interested, threshold between (dist_min,dist_max) will be plot
        :param width: the figure width.
        :param height: the figure height.
        :param res_key: The key to store co-occurence result in data.tl.result

        '''  # noqa
        from seaborn import heatmap
        for tmp in self.pipeline_res[res_key].values():
            break
        groups = [x for x in tmp.index if (x < dist_max) & (x > dist_min)]
        nrow = int(np.sqrt(len(groups)))
        ncol = np.ceil(len(groups) / nrow).astype(int)
        if width is None:
            width = 9 * ncol
        if height is None:
            height = 8 * nrow
        fig = plt.figure(figsize=(width, height))
        axs = fig.subplots(nrow, ncol)
        self._close_axis(axs)
        clust_unique = list(self.stereo_exp_data.cells[cluster_res_key].astype('category').cat.categories)
        for i, g in enumerate(groups):
            interest = pd.DataFrame({x: self.pipeline_res[res_key][x].T[g] for x in self.pipeline_res[res_key]})
            if nrow == 1:
                if ncol == 1:
                    ax = axs
                else:
                    ax = axs[i]
            else:
                ax = axs[int(i / ncol)][(i % ncol)]
            if set(interest.index) == set(clust_unique) and set(interest.columns) == set(clust_unique):
                interest = interest.loc[clust_unique, clust_unique]
            heatmap(interest, ax=ax, center=0)
            ax.set_title('{:.4g}'.format(g))
            ax.set_axis_on()
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', va='top')
        return fig
