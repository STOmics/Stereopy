from natsort import natsorted
from typing import Union, Optional, Literal
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import seaborn as sns

from stereo.core.stereo_exp_data import StereoExpData, AnnBasedStereoExpData
from stereo.stereo_config import stereo_conf
from .ms_plot_base import MSDataPlotBase


class PlotMsSpatialScatter(MSDataPlotBase):

    def __init__(self, ms_data, pipeline_res=None):
        super().__init__(ms_data, pipeline_res)
        self.__default_ncols = 3

    def ms_spatial_scatter(
        self,
        color_by: Literal["total_counts", "n_genes_by_counts", "gene", "cluster"] = "total_counts",
        color_key: Optional[str] = None,
        ncols: Optional[int] = 3,
        dot_size: Optional[int] = 15,
        palette: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        x_label: Optional[Union[list, str]] = 'spatial1',
        y_label: Optional[Union[list, str]] = 'spatial2',
        vmin: float = None,
        vmax: float = None,
        marker: str = 'o',
        invert_y: bool = True,
        use_raw: bool = True
    ):
        """
        Plot spatial scatter plot for multiple slices.
        Each slice will be shown in a single plot.

        :param color_by: spcify the way of coloring, default to 'total_counts'.
                        if set to 'gene', you need to specify a gene name by `color_key`.
                        if set to 'cluster', you need to specify the key to get cluster result by `color_key`.
        :param color_key: the key to get the data to color the plot, it is ignored when the `color_by` is set to 'total_counts' or 'n_genes_by_counts'.
        :param ncols: the slices in plot will be shown as `ncols` columns, defaults to 3.
                        `nrows` will be calculated automatically based on ncols and the number of slices.
        :param dot_size: size of markers in plot, defaults to 15.
        :param palette: palette used to color, defaults to None.
                        by default, it will be set to a palette customized by stereo.
                        more palettes refer to `matplotlib.colors <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.
        :param width: the plot width in pixels, defaults to None.
                        by default, it will be set to 6 times of `ncols`.
        :param height: the plot height in pixels, defaults to None.
                        by default, it will be set to 6 times of `nrows`.
        :param x_label: the label of x-axis, defaults to 'spatial1'.
        :param y_label: the label of y-axis, defaults to 'spatial2'.
        :param vmin: The value representing the lower limit of the color scale. Values smaller than vmin are plotted with the same color as vmin.
        :param vmax: The value representing the higher limit of the color scale. Values greater than vmax are plotted with the same color as vmax.
                        vmin and vmax will be ignored when `color_by` is 'cluster'.
                        by default, vmin and vmax are set to the min and max of the color data.
        :param marker: the marker style, defaults to 'o'.
                        valid values refer to `matplotlib.markers <https://matplotlib.org/stable/api/markers_api.html>`_.
        :param invert_y: whether to invert the y-axis, defaults to True.
        :param use_raw: wheter to use raw data when `color_by` is 'gene', defaults to True.
                        set to False automatically when raw data is not available.
        :return: the figure object.
        """
        nrows, ncols = self._get_row_col(ncols)
        if width is None:
            width = ncols * 6
        if height is None:
            height = nrows * 6

        fig: Figure
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height), sharex=False, sharey=False)
        data_type = 'category' if color_by == 'cluster' else 'numeric'
        hue_list = []
        for i in range(self.ms_data.num_slice):
            data: Union[StereoExpData, AnnBasedStereoExpData] = self.ms_data.data_list[i]
            if color_by == 'cluster':
                if color_key not in data.cells:
                    hue = np.array([], dtype='U')
                else:
                    hue = np.array(data.cells[color_key], dtype='U')
            elif color_by == 'gene':
                if use_raw and data.raw is not None:
                    exp_matrix = data.raw.exp_matrix
                else:
                    exp_matrix = data.exp_matrix
                if data.issparse():
                    hue = exp_matrix[:, data.genes.gene_name == color_key].toarray().flatten()
                else:
                    hue = exp_matrix[:, data.genes.gene_name == color_key].flatten()
            else:
                if color_by not in data.cells:
                    hue = np.array([], dtype=int)
                else:
                    hue = data.cells[color_by]
            hue_list.append(hue)
        hue_all = np.concatenate(hue_list)
        if data_type == 'numeric':
            if palette is None:
                palette = ListedColormap(stereo_conf.linear_colors('stereo'))
            if hue_all.size > 0:
                norm = plt.Normalize(
                    vmin=np.min(hue_all) if vmin is None else vmin,
                    vmax=np.max(hue_all) if vmax is None else vmax
                )
                mappable = cm.ScalarMappable(norm=norm, cmap=palette)
                mappable.set_array([])
                cbar = fig.colorbar(mappable, ax=axes, shrink=0.5, orientation='vertical')
                cbar.ax.set_title(color_key if color_by == 'gene' else color_by, loc='left', y=1.02)
            else:
                norm = None
        else:
            hue_unique = natsorted(np.unique(hue_all))
            if palette is None:
                palette = stereo_conf.get_colors('stereo_30', n=len(hue_unique))
            else:
                palette = sns.color_palette(palette, len(hue_unique))
            palette: OrderedDict = OrderedDict(zip(hue_unique, palette))
            if len(hue_unique) > 0:
                lines = []
                for key, value in palette.items():
                    line = mlines.Line2D([], [], color=value, marker=marker, linestyle='None', markersize=5, label=key)
                    lines.append(line)
                legend_cols, left = divmod(len(lines), 15)
                if left > 0:
                    legend_cols += 1
                fig.legend(handles=lines, loc='upper left', bbox_to_anchor=(0.95, 0.7),
                        ncol=legend_cols, title=color_key, title_fontsize=12, borderaxespad=0, frameon=False)
            
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                if nrows > 1 and ncols > 1:
                    ax: Axes =  axes[i, j]
                elif nrows == 1 and ncols > 1:
                    ax: Axes = axes[j]
                elif nrows > 1 and ncols == 1:
                    ax: Axes = axes[i]
                else:
                    ax: Axes = axes
                if invert_y:
                    ax.invert_yaxis()
                if idx >= self.ms_data.num_slice:
                    ax.axis('off')
                    continue
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                data: Union[StereoExpData, AnnBasedStereoExpData] = self.ms_data.data_list[idx]
                x = data.position[:, 0]
                y = data.position[:, 1]
                hue = hue_list[idx]
                ax.set_title(f'sample {self.ms_data.names[idx]}')
                if len(hue) == 0:
                    continue
                plot_data = pd.DataFrame({'x': x, 'y': y, 'hue': hue, 'size': dot_size})
                if data_type == 'numeric':
                    sns.scatterplot(
                        data=plot_data,
                        x='x', y='y', hue='hue', size='size', sizes=(dot_size, dot_size), palette=palette, 
                        hue_norm=norm, marker=marker, linewidth=0, legend=False, ax=ax)
                elif data_type == 'category':
                    sns.scatterplot(
                        data=plot_data,
                        x='x', y='y', hue='hue', size='size', sizes=(dot_size, dot_size), palette=palette, hue_order=hue_unique, 
                        marker=marker, linewidth=0, legend=False, ax=ax)

        return fig

    def _get_row_col(self, ncols: int = None):
        if ncols is None:
            ncols = self.__default_ncols
        ncols = min(ncols, self.ms_data.num_slice)
        nrows, left = divmod(self.ms_data.num_slice, ncols)
        if left > 0:
            nrows += 1
        return nrows, ncols