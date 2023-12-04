"""
输入x_raw, y_raw, ptime
"""

import numpy as np
import pandas as pd
from scipy.ndimage import convolve as conv
from scipy.ndimage import gaussian_filter as gauss_fil


class Vec():
    def __init__(self):
        self.offset_x = None
        self.offset_y = None
        self.offset_z = None
        self.scale = None
        self.yx_scaled = None
        self.uniq_yx_scaled = None
        self.uni_ind = None
        self.new_arr_sh = None

    # def filter_minority(self, ty_raw, count_thresh, *args):
    #     """
    #     filter out spots with occurrences less than threshold value
    #     :param ty_raw:
    #     :param count_thresh:
    #     :param *args:
    #     :return: args with elements left remained
    #     """
    #
    #     ty_count_dic = dict(Counter(ty_raw))
    #
    #     # generate a list of all types to be remained
    #     ty_arr = np.array(list(ty_count_dic.keys()))
    #     val_arr = np.array(list(ty_count_dic.values()))
    #     keep_ty = ty_arr[np.where(val_arr > count_thresh)]
    #
    #     # generate a mask for args that stores indices of remained elements
    #     mask_keep = np.isin(ty_raw, keep_ty)
    #
    #     arg_keep_li = []
    #     for arg in args:
    #         arg_keep_li.append(arg[mask_keep])
    #     return arg_keep_li

    def preprocess(self, x_raw, y_raw, z_raw, num_pix):
        """
        process raw coordinate to get coordinates and array shape, for voxelizing result in the future

        :param x_raw: (n,)
        :param y_raw: (n,)
        :param z_raw: (n,)
        :param  num_pix: the length of the shorter axis of the matrix. shorter than span of x_raw and span of y_raw.

        :return self.offset_x
        :return self.offset_y
        :return self.offset_z
        :return self.scale

        :return self.xyz_scaled
        :return self.uniq_xyz_scaled
        :return self.uni_ind
        :return self.new_arr_sh
        """
        # zero out axes
        self.offset_x = -1 * x_raw.min()
        self.offset_y = -1 * y_raw.min()
        self.offset_z = -1 * z_raw.min()

        x_sh = x_raw + self.offset_x
        y_sh = y_raw + self.offset_y
        z_sh = z_raw + self.offset_z

        # calculate scale
        shortest_edge_max_coor = np.array([x_sh.max(), y_sh.max(), z_sh.max()]).min()  # 最短边的最大坐标
        self.scale = (num_pix - 1) / shortest_edge_max_coor  # scale, 使最短边的最大坐标变换后，成为num_pix-1

        # process coordinates into new array framework
        x_scaled = np.ceil(x_sh * self.scale).astype(np.int32)  # (n,), val_seq: (n,)
        y_scaled = np.ceil(y_sh * self.scale).astype(np.int32)  # (n,)
        z_scaled = np.ceil(z_sh * self.scale).astype(np.int32)  # (n,)

        # get coordinate output
        xyz_scaled = np.concatenate([np.expand_dims(x_scaled, axis=1),
                                     np.expand_dims(y_scaled, axis=1),
                                     np.expand_dims(z_scaled, axis=1)],
                                    axis=1)  # (n,2)

        # get unique version of coordinate output
        uniq_xyz_scaled, uni_ind = np.unique(xyz_scaled, return_index=True, axis=0)

        # get new array shape
        new_arr_sh = (x_scaled.max() + 1,
                      y_scaled.max() + 1,
                      z_scaled.max() + 1)

        self.xyz_scaled = xyz_scaled
        self.uniq_xyz_scaled = uniq_xyz_scaled
        self.uni_ind = uni_ind
        self.new_arr_sh = new_arr_sh
        return

    def gen_arr_for_mean(self, val_seq_for_mean):
        """
        generate a matrix from val_seq with exact shape as self.new_arr_sh

        Value of element in the matrix corresponds to the average of raw value in each pixel

        :param val_seq_for_mean: array for calculating mean. np.ndarray, (n,)

        :return: s_arr: the new array, with nan values taking indices with no spots
        """
        # initiate the new matrix filled with nan values
        s_arr = np.empty(shape=(self.new_arr_sh))  # e.g. 2.1 * 2.6 + 1 = 6.46 -> 7, 支持0~6的索引
        s_arr[:] = np.nan

        # find mean of each group of (y_scaled, x_scaled)
        # 原方案：已跑通，时间n^2, 发育的脑数据约十分钟
        # val_seq_agg_mean = [val_seq_for_mean[np.array([i for i in range(self.yx_scaled.shape[0]) if (self.yx_scaled[i] == uniq_yx).all()])].mean()  # noqa
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
        df = pd.DataFrame({'x': self.xyz_scaled[:, 0], 'y': self.xyz_scaled[:, 1], 'z': self.xyz_scaled[:, 2],
                           'val': val_seq_for_mean})
        df = df.groupby(by=['x', 'y', 'z'], sort=False).agg(
            {'x': 'mean', 'y': 'mean', 'z': 'mean', 'val': 'mean'}).reset_index(drop=True)
        s_arr[df['x'].to_numpy(dtype='int'), df['y'].to_numpy(dtype='int'), df['z'].to_numpy(dtype='int')] = df['val']
        return s_arr

    # def gen_arr_for_common(self, val_seq_for_common):
    #     """
    #     generate a matrix from [x_raw, y_raw, val_seq] while keeping its geometric shape, with exact shape as
    #     scale * ((y_raw.max - y_raw.min), (x_raw.max - x_raw.min)).
    #
    #     Value of element in the matrix corresponds to the most common value in each pixel
    #
    #     :param val_seq_for_common: array for calculating the most common. np.ndarray, (n,)
    #
    #     :return: s_arr: the new array, with nan values taking indices with no spots
    #     """
    #
    #     # initiate the new matrix filled with nan values
    #     s_arr = np.empty(shape=(self.new_arr_sh), dtype=object)  # e.g. 2.1 * 2.6 + 1 = 6.46 -> 7, 支持0~6的索引  # element is None by default  # noqa
    #
    #     # find mean of each group of (y_scaled, x_scaled)
    #     # 旧方案：已经测通，时间复杂度n^2,占用约10min
    #     # val_seq_agg_common = [collections.Counter([val_seq_for_common[i] for i in range(self.yx_scaled.shape[0])
    #     #                                            if (self.yx_scaled[i] == uniq_yx).all()]).most_common()[0][0]
    #     #                       for uniq_yx in self.uniq_yx_scaled]
    #     # # assign values
    #     # s_arr[self.yx_scaled[:, 0][self.uni_ind], self.yx_scaled[:, 1][self.uni_ind]] = val_seq_agg_common  # e.g 最大值：2.1 * 2.6 = 5.46 -> 6 # noqa
    #
    #     # 新方案：改用pandas加速 TODO: 和力昂沟通是否避免用pandas
    #     df = pd.DataFrame(columns=['y', 'x', 'val'])
    #     # df['val'] = df['val'].astype(str)
    #     df['y'] = self.yx_scaled[:, 0]
    #     df['x'] = self.yx_scaled[:, 1]
    #
    #     df['val'] = val_seq_for_common
    #
    #     df = df.groupby(by=['y', 'x'], sort=False).agg({'y': 'mean', 'x': 'mean', 'val': lambda x: pd.Series.mode(x)[0]}).reset_index(drop=True) # noqa
    #
    #     s_arr[df['y'].to_numpy(dtype='int'), df['x'].to_numpy(dtype='int')] = df['val']
    #     return s_arr

    def filter(self, arr, sigma_val, radius_val, type_val='gauss'):
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
            arr_fil = conv(arr, weights=np.ones((d, d)))
        else:
            arr_fil = arr.copy()
        arr_fil[mask] = np.nan
        return arr_fil

    # func2: 计算画图参数
    def cal_param(self, s_arr):
        # generate two nan indices for filling-up uses
        arr_append_x = np.empty((1, s_arr.shape[1], s_arr.shape[2]))
        arr_append_x[:] = np.nan

        arr_append_y = np.empty((s_arr.shape[0], 1, s_arr.shape[2]))
        arr_append_y[:] = np.nan

        arr_append_z = np.empty((s_arr.shape[0], s_arr.shape[1], 1))
        arr_append_z[:] = np.nan

        # generate ux, uy, uz
        s_arr_x_sh = np.append(arr_append_x, s_arr[:-1, :, :],
                               axis=0)  # shifted  (1, sh[1], sh[2]), (sh[0]-1, sh[1], sh[2])
        s_arr_x_unsh = np.append(s_arr[:-1, :, :], arr_append_x, axis=0)  # unshifted 0
        ux = s_arr_x_sh - s_arr_x_unsh
        mask_nan_ux = np.isnan(s_arr_x_sh) | np.isnan(s_arr_x_unsh)
        ux[mask_nan_ux] = np.nan

        s_arr_y_sh = np.append(arr_append_y, s_arr[:, :-1, :], axis=1)
        s_arr_y_unsh = np.append(s_arr[:, :-1, :], arr_append_y, axis=1)
        uy = s_arr_y_sh - s_arr_y_unsh
        mask_nan_uy = np.isnan(s_arr_y_sh) | np.isnan(s_arr_y_unsh)
        uy[mask_nan_uy] = np.nan

        s_arr_z_sh = np.append(arr_append_z, s_arr[:, :, :-1], axis=2)
        s_arr_z_unsh = np.append(s_arr[:, :, :-1], arr_append_z, axis=2)
        uz = s_arr_z_sh - s_arr_z_unsh
        mask_nan_uz = np.isnan(s_arr_z_sh) | np.isnan(s_arr_z_unsh)
        uz[mask_nan_uz] = np.nan
        return ux, uy, uz
