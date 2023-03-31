import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from stereo.plots.ms_plot_base import MSDataPlotBase


class MSPlot(MSDataPlotBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            data.plt.cluster_scatter(ax=ax, res_key=res_key, title=self.ms_data.names[i], show_legend=show_legend, hue_order=hue_type, **kwargs)

            res = data.tl.result[res_key]
            hue = np.array(res['group'])
            g |= set(hue)
        return fig
