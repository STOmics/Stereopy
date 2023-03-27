import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from stereo.plots.ms_plot_base import MSDataPlotBase


class MSPlot(MSDataPlotBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ms_cluster_scatter(self, ncols=3, **kwargs):
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

        for i, data in enumerate(self.ms_data.data_list):
            ax = fig.add_subplot(axs[i])
            data.plt.cluster_scatter(ax=ax, res_key='{}_integrated', title=self.ms_data.names[i], **kwargs)
        return fig
