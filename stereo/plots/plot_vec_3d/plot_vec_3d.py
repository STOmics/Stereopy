import numpy as np

from .vec import Vec
from ..plot_base import PlotBase


class PlotVec3D(PlotBase):

    def plot_vec_3d(
            self,
            x_raw,
            y_raw,
            z_raw,
            ptime,
            num_cell=20,
            fil_type='gauss',
            sigma_val=1,
            radius_val=3,
    ):
        """
        Plot vector field of cell trajectories embodied in 3D space.

        :param x_raw: Array of x coordinates, taken from the first column in adata spatial axis array
        :param y_raw: Array of y coordinates, taken from the second column in adata spatial axis array
        :param z_raw: Array of z coordinates, taken from the third column in adata spatial axis array, or other
                      equivalent methods
        :param ptime: Array of pseudo-time, suggested being calculated by StereoPy dpt process
        :param num_cell: Number of cells along the shortest coordinates among x, y and z. Number of cells among other
                         coordinates will be calculated proportionally to their respective axis length
        :param fil_type: Filter type when smoothing the ptime voxelized result. Allowed values are:
                        'gaussian': using Gaussian filter
                        'mean': using Mean filter
        :param sigma_val: Sigma of filter when smoothing the voxelized ptime result using Gaussian filter, by passing
                         'gaussian' to fil_type. Use this website to help understand kernel results when assigning different
                          sigma values: http://demofox.org/gauss.html. Value neglected when passing 'mean' to fil_type
        :param radius_val: Size of 'extended radius' of kernel, when smoothing the voxelized ptime result using Mean
                           filter, by passing 'mean' to fil_type. The diameter of kernel d equals to (2 * radius_val + 1)
                           Value neglected when passing 'gaussian' to fil_type

        :return:
        """  # noqa

        # todo: 1. assert

        # 预处理，获得矩阵数据需要的输入
        vec = Vec()

        vec.preprocess(x_raw, y_raw, z_raw, num_pix=num_cell)
        print('\npreprocessed')

        # 生成画图用的矩阵数据
        plt_avg_ptime = vec.gen_arr_for_mean(ptime)
        print('\nplt_avg_ptime', plt_avg_ptime.shape)

        plt_avg_ptime_fil = vec.filter(plt_avg_ptime, sigma_val=sigma_val, radius_val=radius_val, type_val=fil_type)
        print('\nptime voxelized and smoothed')
        print('plt_avg_ptime_fil', plt_avg_ptime_fil.shape)

        ux, uy, uz = vec.cal_param(plt_avg_ptime_fil)
        print('\nux, uy, uz calculated.')
        print('ux', ux.shape)
        print('uy', uy.shape)
        print('uz', uz.shape)

        x = np.linspace(0, ux.shape[0], ux.shape[0])
        y = np.linspace(0, ux.shape[1], ux.shape[1])
        z = np.linspace(0, ux.shape[2], ux.shape[2])
        xm, ym, zm = np.meshgrid(y, x, z)

        # todo：继续在这里写画图代码
        return ux, uy, uz, xm, ym, zm
