import os
import time

import numpy as np
import matplotlib.pyplot as plt

from stereo.plots.plot_vec.vec import Vec
from stereo.plots.plot_base import PlotBase


class PlotVec(PlotBase):

    # TODO：加入输入的边界条件
    # TODO: 把画图的代码单独取出, 其余的结构性代码作为main，放在plot_vec中

    def plot_vec(
            self,
            x_raw,
            y_raw,
            ty_raw,
            ptime,
            type='vec',
            count_thresh=0,
            tick_step=4000,
            line_len_co=1,
            vec_alpha=1,
            line_width=0.0025,
            density=2,
            background=None,
            background_alpha=0.5,
            num_pix=50,
            filter_type='gauss',
            sigma_val=0.4,
            radius_val=1,
            scatter_s=1,
            seed_val=1,
            num_legend_per_col=12,
            dpi_val=1000
    ):
        assert len(x_raw.shape) == 1 and x_raw.shape[0] == y_raw.shape[0] == ptime.shape[0], \
            "input has wrong array shape"

        # 预处理，获得矩阵数据需要的输入
        vec = Vec()
        x_raw, y_raw, ty_raw, ptime = vec.filter_minority(ty_raw, count_thresh, x_raw, y_raw, ty_raw, ptime)

        vec.preprocess(x_raw, y_raw, num_pix)
        # print('preprocessed')

        # 生成画图用的矩阵数据
        plt_avg_ptime = vec.gen_arr_for_mean(ptime)
        plt_avg_ptime_fil = vec.filter(plt_avg_ptime, filter_type, sigma_val, radius_val)

        plt_common_ty = vec.gen_arr_for_common(ty_raw)
        # print('most common type in each pixel calculated.')

        u, v = vec.cal_param(plt_avg_ptime_fil)
        # print('u, v calculated.')

        mask_nan = np.isnan(u) | np.isnan(v) | (u == 0) | (v == 0)
        u[mask_nan] = np.nan
        v[mask_nan] = np.nan

        return vec.plot_line(x_raw, y_raw, ty_raw, plt_common_ty, u, v,
                             type, background, background_alpha, scatter_s,
                             seed_val, num_legend_per_col,
                             line_len_co, vec_alpha, line_width, density,
                             tick_step, dpi_val)

    def plot_time_scatter(self, group='leiden', dpi_val=2000):
        data = self.stereo_exp_data
        x_raw = data.position[:, 0]
        y_raw = data.position[:, 1]
        # x_raw = adata.obsm['spatial_stereoseq']['X'].to_numpy()
        # y_raw = adata.obsm['spatial_stereoseq']['Y'].to_numpy()

        data.cells[group] = data.cells[group].astype('category')
        ptime = data.tl.result['dpt_pseudotime']

        figure = plt.figure(dpi=dpi_val)
        plt.scatter(x_raw, y_raw, s=4, c=ptime, linewidths=0, cmap='rainbow')
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.savefig(os.path.join(fig_dir, fig_name), dpi=2000)
        # plt.close()
        return figure
