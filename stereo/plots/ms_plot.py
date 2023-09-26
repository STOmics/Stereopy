from typing import (
    Optional,
    Union,
    List
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from stereo.plots.ms_plot_base import MSDataPlotBase
from stereo.plots.scatter import base_scatter
from .decorator import download
from .decorator import plot_scale


class MSPlot(MSDataPlotBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __create_axes(
            self,
            plot_count=None,
            ncols=2,
            width=None,
            height=None
    ):
        if plot_count is None:
            plot_count = len(self.ms_data.data_list)
        ncols = min(ncols, plot_count)
        nrows = np.ceil(plot_count / ncols).astype(int)
        if width is None:
            width = ncols * 10
        if height is None:
            height = nrows * 8
        fig = plt.figure(figsize=(width, height))

        left = 0.2 / ncols
        bottom = 0.13 / nrows
        grid_specs = gridspec.GridSpec(
            nrows=nrows,
            ncols=ncols,
            left=left,
            right=1 - (ncols - 1) * left - 0.01 / ncols,
            bottom=bottom,
            top=1 - (nrows - 1) * bottom - 0.1 / nrows,
        )
        axes = [fig.add_subplot(grid_specs[i]) for i in range(plot_count)]
        return fig, axes

    @download
    @plot_scale
    def ms_spatial_aligned_scatter(
            self,
            ncols: int = 2,
            dot_size: int = None,
            palette: str = 'stereo_30',
            width: int = None,
            height: int = None,
            slices: Optional[List[Union[int, str]]] = None,
            type: str = "pairwise",
            merged: bool = False,
            use_raw: bool = False,
            **kwargs
    ):
        if type == 'center':
            if not hasattr(self.ms_data, 'center_slice'):
                raise Exception(
                    "There is no center slice, be sure that you have ran the spatial alignment whit `center` method.")

        if dot_size is None:
            max_cell_count = 0
            for data in self.ms_data:
                if data.shape[0] > max_cell_count:
                    max_cell_count = data.shape[0]
            dot_size = 220000 / max_cell_count

        if slices is None:
            slices = list(range(len(self.ms_data.data_list)))
        slices_count = len(slices)

        if merged:
            fig, axes = self.__create_axes(plot_count=1, ncols=1, width=width, height=height)
            pos, hue = [], []
            if type == 'center':
                center_slice = self.ms_data.center_slice
                pos.append(center_slice.raw_position if use_raw else center_slice.position)
                hue.append(np.repeat('center', repeats=center_slice.shape[0]))
            for i in slices:
                slice = self.ms_data[i]
                if use_raw:
                    pos.append(slice.raw_position)
                else:
                    pos.append(slice.position)
                slice_name = self.ms_data.names[i] if isinstance(i, int) else i
                hue.append(np.repeat(slice_name, repeats=slice.shape[0]))
            pos = np.concatenate(pos, axis=0)
            hue = np.concatenate(hue, axis=0)
            base_scatter(
                x=pos[:, 0],
                y=pos[:, 1],
                hue=hue,
                palette=palette,
                dot_size=dot_size,
                marker='.',
                ax=axes[0],
                title="all slices",
                width=width,
                height=height,
                legend_ncol=1,
                **kwargs
            )
            return fig

        if type == 'pairwise':
            fig, axes = self.__create_axes(plot_count=slices_count - 1, ncols=ncols, width=width, height=height)
            for i in range(slices_count - 1):
                slice_a_idx, slice_b_idx = slices[i], slices[i + 1]
                slice_a, slice_b = self.ms_data[slice_a_idx], self.ms_data[slice_b_idx]
                slice_a_name = self.ms_data.names[slice_a_idx] if isinstance(slice_a_idx, int) else slice_a_idx
                slice_b_name = self.ms_data.names[slice_b_idx] if isinstance(slice_b_idx, int) else slice_b_idx
                if use_raw:
                    pos = np.concatenate([slice_a.raw_position, slice_b.raw_position], axis=0)
                else:
                    pos = np.concatenate([slice_a.position, slice_b.position], axis=0)
                hue = np.concatenate([np.repeat(slice_a_name, repeats=slice_a.shape[0]),
                                      np.repeat(slice_b_name, repeats=slice_b.shape[0])], axis=0)
                base_scatter(
                    x=pos[:, 0],
                    y=pos[:, 1],
                    hue=hue,
                    palette=palette,
                    dot_size=dot_size,
                    marker='.',
                    ax=axes[i],
                    title=f"slice {slice_a_name} and {slice_b_name}",
                    width=width,
                    height=height,
                    legend_ncol=1,
                    **kwargs
                )
        elif type == 'center':
            fig, axes = self.__create_axes(plot_count=slices_count, ncols=ncols, width=width, height=height)
            center_slice = self.ms_data.center_slice
            for i, slice_idx in enumerate(slices):
                slice = self.ms_data[slice_idx]
                slice_name = self.ms_data.names[slice_idx] if isinstance(slice_idx, int) else slice_idx
                if use_raw:
                    pos = np.concatenate([center_slice.raw_position, slice.raw_position], axis=0)
                else:
                    pos = np.concatenate([center_slice.position, slice.position], axis=0)
                hue = np.concatenate([np.repeat('center', repeats=center_slice.shape[0]),
                                      np.repeat(slice_name, repeats=slice.shape[0])], axis=0)
                base_scatter(
                    x=pos[:, 0],
                    y=pos[:, 1],
                    hue=hue,
                    palette=palette,
                    dot_size=dot_size,
                    marker='.',
                    ax=axes[i],
                    title=f"center slice and slice {slice_name}",
                    width=width,
                    height=height,
                    legend_ncol=1,
                    **kwargs
                )
        return fig

    def ms_cluster_scatter(self, res_key='leiden', ncols=3, **kwargs):
        ncols = min(ncols, len(self.ms_data.data_list))
        nrows = np.ceil(len(self.ms_data.data_list) / ncols).astype(int)
        fig = plt.figure(figsize=(ncols * 10, nrows * 8))

        left = 0.2 / ncols
        bottom = 0.13 / nrows
        axs = gridspec.GridSpec(
            nrows=nrows,
            ncols=ncols,
            left=left,
            right=1 - (ncols - 1) * left - 0.01 / ncols,
            bottom=bottom,
            top=1 - (nrows - 1) * bottom - 0.1 / nrows,
        )

        g = set()
        for i, data in enumerate(self.ms_data.data_list):
            show_legend = False
            hue_type = None
            if i >= len(self.ms_data.data_list) - 1:
                show_legend = True
                hue_type = g
            ax = fig.add_subplot(axs[i])
            data.plt.cluster_scatter(ax=ax, res_key=res_key, title=self.ms_data.names[i], show_legend=show_legend,
                                     hue_order=hue_type, **kwargs)

            res = data.tl.result[res_key]
            hue = np.array(res['group'])
            g |= set(hue)
        return fig
