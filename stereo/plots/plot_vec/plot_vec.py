import numpy as np

from stereo.plots.decorator import plot_scale
from stereo.plots.plot_base import PlotBase
from stereo.plots.plot_vec.vec import Vec


class PlotVec(PlotBase):

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
        """
        Plot vectors or streams of pseudo-time.
        
        :param x_raw: Array of x coordinates, taken from the first column in adata spatial axis array
        :param y_raw: Array of y coordinates, taken from the second column in adata spatial axis array
        :param ty_raw: 1d NdArray, involving cell types of all cells, or bin sets, can take the format of either string, int, or float
        :param ptime: Array of pseudo-time, suggested being calculated by StereoPy dpt process
        :param type: 'vec' or 'vector' plots vector plots, 'stream' or 'streamplot' plots stream plots
        :param count_thresh: threshold of counts when filtering spots to plot
        :param tick_step: step between tick labels
        :param line_len_co: length coeeficient of vectors
        :param vec_alpha: transparency of vectors, 0-1
        :param line_width: width of vectors
        :param density: density of streams in stream plot
        :param background: 'field' plots fields-like background, with pixel color representing cell types, while 'scatter', 'cell', 'bin', or 'spot' plots each spot as a scatter
        :param background_alpha: transparency of background
        :param num_pix: number of pixel on shorter axis (x or y) when plotting background as fields
        :param filter_type: type of kernel when smoothing vectors, if type is 'vec' or 'vector'. Pass 'gauss' to use Gaussian kernel, pass 'mean' to use Mean kernel.
        :param sigma_val: sigma of kernel if passing 'gauss' to filter_type
        :param radius_val: half of width of kernel array, if passing 'mean' to filter_type
        :param scatter_s: size of scatter, if passing 'scatter', 'cell', 'bin', or 'spot' to background
        :param seed_val: seed value to assign colors for different cell types when plotting background
        :param num_legend_per_col: number of lines per column in legend
        :param dpi_val: dpi value of figure
        """
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
        u, v = vec.cal_param(plt_avg_ptime_fil)
        mask_nan = np.isnan(u) | np.isnan(v) | (u == 0) | (v == 0)
        u[mask_nan] = np.nan
        v[mask_nan] = np.nan

        return vec.plot_line(x_raw, y_raw, ty_raw, plt_common_ty, u, v,
                             type, background, background_alpha, scatter_s,
                             seed_val, num_legend_per_col,
                             line_len_co, vec_alpha, line_width, density,
                             tick_step, dpi_val)

    @plot_scale
    def plot_time_scatter(
            self,
            group='leiden',
            vmin: float = None,
            vmax: float = None,
            palette: str = 'stereo',
            **kwargs
    ):
        """
        Spatial distribution of pseudotime.

        :param group: The key to get clustering result, now it will be ignored, defaults to 'leiden'.
        :param vmin: Define the data range that the colormap covers, defaults to None.
        :param vmax: Define the data range that the colormap covers, defaults to None.
        :param palette: colormap, defaults to 'stereo'.
        
        """
        data = self.stereo_exp_data

        data.cells[group] = data.cells[group].astype('category')
        ptime = data.tl.result['dpt_pseudotime']

        from ..scatter import multi_scatter

        fig = multi_scatter(
            x=data.position[:, 0],
            y=data.position[:, 1],
            hue=[ptime],
            x_label=['spatial1'],
            y_label=['spatial2'],
            title=['dpt_pseudotime'],
            color_bar=True,
            width=None,
            height=None,
            palette=palette,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
        return fig
