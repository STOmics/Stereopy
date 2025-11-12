from typing import Literal

import matplotlib.pylab as plt
import numpy as np
from matplotlib.axes import Axes

from stereo.log_manager import logger
from .plot_base import PlotBase


class PlotDendrogram(PlotBase):

    def dendrogram(
            self,
            orientation: Literal['top', 'bottom', 'left', 'right'] = 'top',
            remove_labels: bool = False,
            ticks=None,
            title=None,
            width=None,
            height=None,
            ax: Axes = None,
            res_key: str = 'dendrogram',
    ):
        """
        Plots a dendrogram using the precomputed dendrogram
        information stored in `data.tl.result[res_key]`
        """

        def translate_pos(pos_list, new_ticks, old_ticks):
            if not isinstance(old_ticks, list):
                # assume that the list is a numpy array
                old_ticks = old_ticks.tolist()
            new_xs = []
            for x_val in pos_list:
                if x_val in old_ticks:
                    new_x_val = new_ticks[old_ticks.index(x_val)]
                else:
                    # find smaller and bigger indices
                    idx_next = np.searchsorted(old_ticks, x_val, side="left")
                    idx_prev = idx_next - 1
                    old_min = old_ticks[idx_prev]
                    old_max = old_ticks[idx_next]
                    new_min = new_ticks[idx_prev]
                    new_max = new_ticks[idx_next]
                    new_x_val = ((x_val - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
                new_xs.append(new_x_val)
            return new_xs

        if ax is None:
            fig, dendro_ax = plt.subplots()
        else:
            dendro_ax = ax
            fig = ax.get_figure()

        if width is not None:
            fig.set_figwidth(width)
        if height is not None:
            fig.set_figheight(height)

        dendro_info = self.pipeline_res[res_key]['dendrogram_info']
        leaves = dendro_info["ivl"]
        icoord = np.array(dendro_info['icoord'])
        dcoord = np.array(dendro_info['dcoord'])

        orig_ticks = np.arange(5, len(leaves) * 10 + 5, 10).astype(float)
        # check that ticks has the same length as orig_ticks
        if ticks is not None and len(orig_ticks) != len(ticks):
            logger.warning(
                "ticks argument does not have the same size as orig_ticks. "
                "The argument will be ignored"
            )
            ticks = None

        for xs, ys in zip(icoord, dcoord):
            if ticks is not None:
                xs = translate_pos(xs, ticks, orig_ticks)
            if orientation in ['right', 'left']:
                xs, ys = ys, xs
            dendro_ax.plot(xs, ys, color='#555555')

        dendro_ax.tick_params(bottom=False, top=False, left=False, right=False)
        ticks = ticks if ticks is not None else orig_ticks
        if orientation in ['right', 'left']:
            dendro_ax.set_yticks(ticks)
            dendro_ax.set_yticklabels(leaves, fontsize='small', rotation=0)
            dendro_ax.tick_params(labelbottom=False, labeltop=False)
            if orientation == 'left':
                xmin, xmax = dendro_ax.get_xlim()
                dendro_ax.set_xlim(xmax, xmin)
                dendro_ax.tick_params(labelleft=False, labelright=True)
        else:
            dendro_ax.set_xticks(ticks)
            dendro_ax.set_xticklabels(leaves, fontsize='small', rotation=90)
            dendro_ax.tick_params(labelleft=False, labelright=False)
            if orientation == 'bottom':
                ymin, ymax = dendro_ax.get_ylim()
                dendro_ax.set_ylim(ymax, ymin)
                dendro_ax.tick_params(labeltop=True, labelbottom=False)

        if remove_labels:
            dendro_ax.tick_params(
                labelbottom=False, labeltop=False, labelleft=False, labelright=False
            )

        dendro_ax.grid(False)

        dendro_ax.spines['right'].set_visible(False)
        dendro_ax.spines['top'].set_visible(False)
        dendro_ax.spines['left'].set_visible(False)
        dendro_ax.spines['bottom'].set_visible(False)

        if title is not None:
            dendro_ax.set_title(title)

        return fig
