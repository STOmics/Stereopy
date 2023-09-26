"""
输入x_raw, y_raw, ptime
"""

import collections
from collections import Counter

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import convolve as conv
from scipy.ndimage import gaussian_filter as gauss_fil


class Vec():
    def __init__(self):
        self.offset_x = None
        self.offset_y = None
        self.scale = None
        self.yx_scaled = None
        self.uniq_yx_scaled = None
        self.uni_ind = None
        self.new_arr_sh = None

    def filter_minority(self, ty_raw, count_thresh, *args):
        """
        filter out spots with occurrences less than threshold value
        :param ty_raw:
        :param count_thresh:
        :param *args:
        :return: args with elements left remained
        """

        ty_count_dic = dict(Counter(ty_raw))

        # generate a list of all types to be remained
        ty_arr = np.array(list(ty_count_dic.keys()))
        val_arr = np.array(list(ty_count_dic.values()))
        keep_ty = ty_arr[np.where(val_arr > count_thresh)]

        # generate a mask for args that stores indices of remained elements
        mask_keep = np.isin(ty_raw, keep_ty)

        arg_keep_li = []
        for arg in args:
            arg_keep_li.append(arg[mask_keep])
        return arg_keep_li

    def preprocess(self, x_raw, y_raw, num_pix):
        """
        process raw coordinate to get coordinates and array shape, for calculating new array in the future\

        :param x_raw: (n,)
        :param y_raw: (n,)
        :param  num_pix: the length of the shorter axis of the matrix. shorter than span of x_raw and span of y_raw.
        """
        # zero out both axis
        self.offset_x = -1 * x_raw.min()
        self.offset_y = -1 * y_raw.min()

        x_raw = x_raw + self.offset_x
        y_raw = y_raw + self.offset_y

        # calculate scale
        shorter_len = np.array([y_raw.max() + 1, x_raw.max() + 1]).min()
        self.scale = num_pix / shorter_len

        # process coordinates into new array framework
        y_scaled = np.ceil(y_raw * self.scale).astype(np.int32)  # (n,)
        x_scaled = np.ceil(x_raw * self.scale).astype(np.int32)  # (n,), val_seq: (n,)

        # get coordinate output
        yx_scaled = np.concatenate([np.expand_dims(y_scaled, axis=1), np.expand_dims(x_scaled, axis=1)],
                                   axis=1)  # (n,2)

        # get unique version of coordinate output
        uniq_yx_scaled, uni_ind = np.unique(yx_scaled, return_index=True, axis=0)

        # get new array shape
        new_arr_sh = (y_scaled.max() + 1, x_scaled.max() + 1)

        self.yx_scaled = yx_scaled
        self.uniq_yx_scaled = uniq_yx_scaled
        self.uni_ind = uni_ind
        self.new_arr_sh = new_arr_sh
        return

    def gen_arr_for_mean(self, val_seq_for_mean):
        """
        generate a matrix from [x_raw, y_raw, val_seq] while keeping its geometric shape, with exact shape as
        scale * ((y_raw.max - y_raw.min), (x_raw.max - x_raw.min)).

        Value of element in the matrix corresponds to the average of raw value in each pixel

        :param val_seq_for_mean: array for calculating mean. np.ndarray, (n,)

        :return: s_arr: the new array, with nan values taking indices with no spots
        """

        # initiate the new matrix filled with nan values
        s_arr = np.empty(shape=(self.new_arr_sh))  # e.g. 2.1 * 2.6 + 1 = 6.46 -> 7, 支持0~6的索引
        s_arr[:] = np.nan

        # find mean of each group of (y_scaled, x_scaled)
        # 原方案：已跑通，时间n^2, 发育的脑数据约十分钟
        # val_seq_agg_mean = [val_seq_for_mean[np.array([i for i in range(self.yx_scaled.shape[0]) if (self.yx_scaled[i] == uniq_yx).all()])].mean() # noqa
        #                     for uniq_yx in self.uniq_yx_scaled]
        # assign values
        # s_arr[self.yx_scaled[:, 0][self.uni_ind], self.yx_scaled[:, 1][
        #     self.uni_ind]] = val_seq_agg_mean  # e.g 最大值：2.1 * 2.6 = 5.46 -> 6

        # increase speed:
        # 方案1： 先sort再split：sort有困难
        # np.sort(self.yx_scaled, )
        # np.unique(self.yx_scaled, return_index=True, axis=0)[1]

        # 方案2： 用numpy_indexed做groupby：方法会自动排序，和val_seq_for_mean顺序对应不上
        # val_seq_grouped = npi.group_by((self.yx_scaled[:, 0], self.yx_scaled[:, 1])).split(val_seq_for_mean)  # list of arrays # noqa
        # [arr.meanval_seq_grouped

        # 方案3：改用pandas加速 TODO: 和力昂沟通是否避免用pandas
        df = pd.DataFrame({'y': self.yx_scaled[:, 0], 'x': self.yx_scaled[:, 1], 'val': val_seq_for_mean})
        df = df.groupby(by=['y', 'x'], sort=False).agg({'y': 'mean', 'x': 'mean', 'val': 'mean'}).reset_index(drop=True)
        s_arr[df['y'].to_numpy(dtype='int'), df['x'].to_numpy(dtype='int')] = df['val']
        return s_arr

    def gen_arr_for_common(self, val_seq_for_common):
        """
        generate a matrix from [x_raw, y_raw, val_seq] while keeping its geometric shape, with exact shape as
        scale * ((y_raw.max - y_raw.min), (x_raw.max - x_raw.min)).

        Value of element in the matrix corresponds to the most common value in each pixel

        :param val_seq_for_common: array for calculating the most common. np.ndarray, (n,)

        :return: s_arr: the new array, with nan values taking indices with no spots
        """

        # initiate the new matrix filled with nan values
        s_arr = np.empty(shape=(self.new_arr_sh),
                         dtype=object)  # e.g. 2.1 * 2.6 + 1 = 6.46 -> 7, 支持0~6的索引  # element is None by default

        # find mean of each group of (y_scaled, x_scaled)
        # 旧方案：已经测通，时间复杂度n^2,占用约10min
        # val_seq_agg_common = [collections.Counter([val_seq_for_common[i] for i in range(self.yx_scaled.shape[0])
        #                                            if (self.yx_scaled[i] == uniq_yx).all()]).most_common()[0][0]
        #                       for uniq_yx in self.uniq_yx_scaled]
        # # assign values
        # s_arr[self.yx_scaled[:, 0][self.uni_ind], self.yx_scaled[:, 1][self.uni_ind]] = val_seq_agg_common  # e.g 最大值：2.1 * 2.6 = 5.46 -> 6 # noqa

        # 新方案：改用pandas加速 TODO: 和力昂沟通是否避免用pandas
        df = pd.DataFrame(columns=['y', 'x', 'val'])
        # df['val'] = df['val'].astype(str)
        df['y'] = self.yx_scaled[:, 0]
        df['x'] = self.yx_scaled[:, 1]

        df['val'] = val_seq_for_common

        df = df.groupby(by=['y', 'x'], sort=False).agg(
            {'y': 'mean', 'x': 'mean', 'val': lambda x: pd.Series.mode(x)[0]}).reset_index(drop=True)

        s_arr[df['y'].to_numpy(dtype='int'), df['x'].to_numpy(dtype='int')] = df['val']
        return s_arr

    def filter(self, arr, type_val, sigma_val, radius_val):
        """
        # use this website to find an ideal combination of sigma and radius: http://demofox.org/gauss.html
        :param arr: array to be filtered, with nan at indices without valid values
        :param type:
        :return:
        """

        mask = np.isnan(arr)

        arr[mask] = 0  # change nan val to zero, to avoid scipy-defined nan value processing
        if type_val == 'gauss':
            arr_fil = gauss_fil(arr, sigma=sigma_val, mode='mirror')  # 0.5, 2
        elif type_val == 'mean':
            d = 2 * radius_val + 1
            arr_fil = conv(arr, weights=np.ones((d, d) / d * d))
        else:
            arr_fil = arr.copy()
        arr_fil[mask] = np.nan
        return arr_fil

    # func2: 计算画图参数
    def cal_param(self, s_arr):
        # generate two nan indices for filling-up uses
        arr_append_col = np.empty((s_arr.shape[0], 1))
        arr_append_col[:] = np.nan
        arr_append_row = np.empty((1, s_arr.shape[1]))
        arr_append_row[:] = np.nan

        # generate u, v
        s_arr_r = np.append(arr_append_col, s_arr[:, :-1], axis=1)  # a_arr shifted right
        s_arr_l = np.append(s_arr[:, :-1], arr_append_col, axis=1)
        u = s_arr_r - s_arr_l
        mask_nan_u = np.isnan(s_arr_r) | np.isnan(s_arr_l)
        u[mask_nan_u] = np.nan

        s_arr_b = np.append(arr_append_row, s_arr[:-1, :], axis=0)  # a_arr shifted down
        s_arr_t = np.append(s_arr[:-1, :], arr_append_row, axis=0)
        v = s_arr_b - s_arr_t
        mask_nan_v = np.isnan(s_arr_b) | np.isnan(s_arr_t)
        u[mask_nan_v] = np.nan

        return u, v

    def _apply_trans(self, x_arr, y_arr):
        """
        apply the transformation applied onto raw data of vector, onto other data
        :param x_arr: np.NdArray: (n,)
        :param y_arr: np.NdArray: (n,)
        :return:
        """
        x_arr_re = x_arr.copy()
        y_arr_re = y_arr.copy()

        x_arr_re = x_arr_re + self.offset_x
        y_arr_re = y_arr_re + self.offset_y

        x_arr_re = x_arr_re * self.scale
        y_arr_re = y_arr_re * self.scale  # (n,)
        return x_arr_re, y_arr_re

    def plot_line(self, x_raw, y_raw, ty_raw, plt_common_ty, u, v,
                  type, background, background_alpha, scatter_s, seed_val, num_legend_per_col,
                  line_len_co, vec_alpha, line_width, density,
                  tick_step, dpi_val):

        # prepare colormap
        # 试过不好，有些颜色过于接近：cmap_val = plt.cm.get_cmap('jet', uni_ele.shape[0])
        # 试过不好，有些颜色过于接近：cmap_val = mpl.colors.ListedColormap(colorcet.glasbey_bw)
        np.random.seed(seed_val)
        cmap_val = mpl.colors.ListedColormap(np.random.rand(256, 3))

        # prepare dictionary that maps types to int
        # ty_raw unique values are as rich as, or richer than plt_common_ty # None表示空pixel
        undup = collections.Counter(ty_raw).keys()
        ty_val_dict = dict(zip(undup, np.arange(len(undup))))  # 生成从str映射到int的字典
        ty_val_dict[None] = np.nan  # 字典中，None对应nan

        scatter_corpus = ['scatter', 'cell', 'bin', 'spot']
        imshow_corpus = ['field']

        figure = plt.figure(dpi=dpi_val)
        if background in scatter_corpus + imshow_corpus:
            if background in scatter_corpus:
                # scatter
                x_plt, y_plt = self._apply_trans(x_raw, y_raw)
                plt_ty = np.vectorize(ty_val_dict.get)(ty_raw)
                im = plt.scatter(x_plt, y_plt, c=plt_ty, cmap=cmap_val, s=scatter_s, linewidths=0,
                                 alpha=background_alpha)

                # generate type list, value list and color list for the legend
                ty_li_uni = np.unique(ty_raw)
                val_li_uni = np.vectorize(ty_val_dict.get)(ty_li_uni)  # may include nan values
                colors_li_uni = [im.cmap(im.norm(val)) for val in val_li_uni]  # 从图上取颜色

            else:
                # plot the image
                plt_common_ty_val = np.vectorize(ty_val_dict.get)(plt_common_ty)
                im = plt.imshow(plt_common_ty_val, cmap=cmap_val, alpha=background_alpha)
                plt.gca().invert_yaxis()

                # generate type list, value list and color list for the legend
                ty_li_uni = np.unique(plt_common_ty[plt_common_ty != np.array(None)])
                val_li_uni = np.vectorize(ty_val_dict.get)(ty_li_uni)  # may include nan values
                colors_li_uni = [im.cmap(im.norm(val)) for val in val_li_uni]  # 从图上取颜色

            patches = [mpatches.Patch(color=colors_li_uni[i], label=ty_li_uni[i]) for i in range(len(val_li_uni)) if
                       not np.isnan(val_li_uni[i])]  # nan excluded
            ncols_val = len(patches) // (num_legend_per_col + 1) + 1
            plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncols_val,
                       framealpha=0)  # loc和bbox_to_anchor组合，loc表示legend的锚点，bbox_to_anchor表示锚点相对图的位置

        if type in ['vec', 'vector']:
            plt.quiver(line_len_co * (-u), line_len_co * (-v), alpha=vec_alpha,
                       width=line_width)  # 从小time指向大time # streamplot

        elif type in ['stream', 'streamplot']:
            x = np.arange(0, u.shape[1])
            y = np.arange(0, u.shape[0])
            X, Y = np.meshgrid(x, y)  # col, row
            plt.streamplot(X, Y, line_len_co * (-u), line_len_co * (-v), color='k', linewidth=line_width,
                           density=density)  # start_points=start_p,

        x_tick_la = np.arange(
            np.floor(x_raw.min() / tick_step) * tick_step,
            np.ceil(x_raw.max() / tick_step) * tick_step,
            step=tick_step
        ).astype(np.int32)

        y_tick_la = np.arange(
            np.floor(y_raw.min() / tick_step) * tick_step,
            np.ceil(y_raw.max() / tick_step) * tick_step,
            step=tick_step
        ).astype(np.int32)

        x_tick_po, y_tick_po = self._apply_trans(x_tick_la, y_tick_la)
        plt.xticks(ticks=x_tick_po, labels=x_tick_la)
        plt.yticks(ticks=y_tick_po, labels=y_tick_la)

        plt.gca().set_aspect('equal', adjustable='box')
        # in order to fit `StereoPy` original point
        plt.gca().invert_yaxis()

        return figure

# func3：画图
