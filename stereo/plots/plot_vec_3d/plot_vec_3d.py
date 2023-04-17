from vec import Vec
import numpy as np
import time
import os
import gc


class PlotVec3D():
    @staticmethod
    def main(x_raw,
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

        :return: todo：加入画图部分代码后添加
        """

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


def test():
    import scanpy
    import anndata

    # 1.1 读入，预处理数据
    dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/data/fruitfly_embryo/bin_recons_spot_level/concat'
    spatial_col = 'spatial_elas'
    ty_col = 'annotation'
    adata = anndata.read(os.path.join(dir, '3d.h5ad'))
    print(set(adata.obs[ty_col]))

    # 1.2 计算dpt
    adata.uns['iroot'] = np.flatnonzero(adata.obs[ty_col] == 'amnioserosa')[10]  # 羊膜浆膜类细胞
    scanpy.tl.dpt(adata, n_branchings=0)

    ptime = adata.obs['dpt_pseudotime'].to_numpy()
    x_raw = adata.obsm[spatial_col][:, 0]
    y_raw = adata.obsm[spatial_col][:, 1]
    z_raw = adata.obsm[spatial_col][:, 2]
    print('test data ready.')

    # 2 画图：目前只包括数据准备
    plot_vec = PlotVec3D()
    ux, uy, uz, xm, ym, zm = plot_vec.main(x_raw,
                                           y_raw,
                                           z_raw,
                                           ptime,
                                           num_cell=10,
                                           fil_type='gauss',
                                           sigma_val=1,
                                           radius_val=3
                                           )

    # 中间结果可视化1：ux, uy, uz截面图
    # import math
    # import matplotlib.pyplot as plt
    #
    # def plot_csection(arr):
    #     plt.figure()
    #     plt.imshow(arr)
    #     plt.show()
    #
    # plt_arr_x = ux[math.floor((ux.shape[0]-1)/2), :, :]
    # plt_arr_y = uy[:, math.floor((uy.shape[1] - 1) / 2), :]
    # plt_arr_z = uz[:, :, math.floor((uz.shape[2] - 1) / 2)]
    # plot_csection(plt_arr_x)

    # 中间结果可视化2：matplotlib画quiver，注意：1. 目前z轴显示比例和x、y轴不同，实际上向量的z分量更大些； 2. 未展示mesh效果
    # # todo: notice me: xyz not of same scale
    # print(xm.shape, ym.shape, zm.shape)
    # import matplotlib.pyplot as plt
    # ax = plt.figure().add_subplot(projection='3d')
    # # ax.set_aspect('equal')  # error： not supported by matplotlib
    # ax.set_box_aspect([1, 1, 1])  # neglected by matplotlib
    # ax.quiver(xm, ym, zm, ux, uy, uz, length=0.5)


if __name__ == '__main__':
    test()