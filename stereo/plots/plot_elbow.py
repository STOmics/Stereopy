from typing import Optional

import matplotlib.pylab as plt
import numpy as np

from .plot_base import PlotBase


class PlotElbow(PlotBase):

    def elbow(
            self,
            pca_res_key: str = 'pca',
            n_pcs: int = None,
            width: Optional[int] = None,
            height: Optional[int] = None,
            title: str = 'Elbow Plot',
            x_label: str = 'Principal Component',
            y_label: str = 'Variance Explained',
            line_width: int = 2,
            color: str = 'blue',
            marker: str = 'o',
            marker_color: Optional[str] = None,
            marker_size: int = 4,
            cum: bool = False
    ):
        """
        Plot elbow for pca.

        :param pca_res_key: the key sepcifies the pca result, defaults to 'pca'.
        :param n_pcs: number of PCs to be displayed, defaults to None to show all PCs.
        :param width: the figure width in pixels, defaults to None.
        :param height: the figure height in pixels, defaults to None.
                        if width or height is None, they will be set to default value by matplotlib,
                        width defaults to 6.4 inches and height defaults to 4.8 inches.  
        :param title: the tilte of the plot, defaults to 'Elbow Plot'.
        :param x_label: the label of the x-axis, defaults to 'Principal Component'.
        :param y_label: the label of the y-axis, defaults to 'Variance Explained'.
        :param line_width: the width of the line in plot, defaults to 2.
        :param color: the color of the line in plot, defaults to 'blue'.
        :param marker: the marker style, defaults to 'o'.
        :param marker_color: the marker color, defaults to the same as line's color.
        :param marker_size: the marker size, defaults to 4.
        :param cum: setting to True means each marker represents a cumulation of current marker and all previous, default to False.

        """  # noqa
        if pca_res_key not in self.pipeline_res:
            raise KeyError(f"Can not find the pca result in data.tl.result by key {pca_res_key}")

        res_key = f'{pca_res_key}_variance_ratio'
        variance_ratio = self.pipeline_res[res_key]
        if n_pcs is not None:
            variance_ratio = variance_ratio[0:n_pcs]

        if cum:
            variance_ratio = np.cumsum(variance_ratio)

        pcs = np.arange(variance_ratio.size) + 1

        fig, ax = plt.subplots()
        if width is not None:
            fig.set_figwidth(width)
        if height is not None:
            fig.set_figheight(height)

        if marker_color is None:
            marker_color = color

        ax.plot(
            pcs,
            variance_ratio,
            marker=marker,
            linewidth=line_width,
            color=color,
            markerfacecolor=marker_color,
            markeredgecolor=marker_color,
            markersize=marker_size
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return fig
