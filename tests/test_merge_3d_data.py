import pytest
import unittest

from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.utils._download import _download
from stereo.utils.data_helper import merge
from settings import DEMO_3D_SLICE_0_15_URLS_LIST, TEST_DATA_PATH


class TestMerge3DData(unittest.TestCase):

    def setUp(self) -> None:
        self._demo_3d_file_list = []
        for demo_url in DEMO_3D_SLICE_0_15_URLS_LIST:
            self._demo_3d_file_list.append(_download(demo_url, dir_str=TEST_DATA_PATH))

    def _preprocess(self, data):
        data.tl.normalize_total(target_sum=1e4)
        data.tl.log1p()
        data.tl.highly_variable_genes(
            min_mean=0.0125, max_mean=3, min_disp=0.5, res_key='highly_variable_genes', n_top_genes=None
        )
        data.tl.scale(zero_center=False)
        data.tl.pca(use_highly_genes=True, hvg_res_key='highly_variable_genes', n_pcs=20, res_key='pca',
                    svd_solver='arpack')
        data.tl.neighbors(pca_res_key='pca', n_pcs=30, res_key='neighbors', n_jobs=2)

    @pytest.mark.heavy
    def test_merge_3d_data(self):
        slices = []
        for slice_path in self._demo_3d_file_list:
            slices.append(AnnBasedStereoExpData(slice_path))
        merged_data = merge(*slices, space_between='0.7um', reorganize_coordinate=False)

        self._preprocess(merged_data)
        ty_col = 'annotation'
        merged_data.cells[ty_col] = merged_data.cells[ty_col].astype('category')
        merged_data.tl.paga(groups=ty_col)

        x_raw = merged_data.position[:, 0]
        y_raw = merged_data.position[:, 1]
        z_raw = merged_data.position_z.reshape(1, -1)[0]

        ty = merged_data.cells[ty_col].to_numpy()
        con = merged_data.tl.result['paga']['connectivities'].todense()  # arr (n_clus, n_clus)
        con_tree = merged_data.tl.result['paga']['connectivities_tree'].todense()

        choose_ty = list(set(ty))
        choose_ty = ['CNS', 'epidermis', 'salivary gland', 'carcass']
        type_traj = 'curve'

        ty_repre_xyz = {}  # type -- np.array(3,)
        mesh_each_ty = []  # each element: mesh of each type, element with the same order as choose_ty

        import numpy as np
        from stereo.algorithm.gen_mesh import ThreeDimGroup

        for ty_name in choose_ty:
            tdg = ThreeDimGroup(list(x_raw), list(y_raw), list(z_raw), list(ty), ty_name=ty_name, eps_val=2,
                                min_samples=3, thresh_num=10)
            try:
                mesh, mesh_li = tdg.create_mesh_of_type(method='march', mc_scale_factor=1.5)
                xyz_repre = tdg.find_repre_point(mesh_li, x_ran_sin=2.25)
            except:  # mesh calculation failed
                mesh = None
                xyz_repre = np.array(
                    [x_raw[ty == ty_name].mean(), y_raw[ty == ty_name].mean(), z_raw[ty == ty_name].mean()])
            mesh_each_ty.append(mesh)
            ty_repre_xyz[ty_name] = xyz_repre

        merged_data.plt.plot_cluster_traj_3d(con_tree, x_raw, y_raw, z_raw, ty, choose_ty, ty_repre_xyz,
                                             type_traj=type_traj)

    @pytest.mark.heavy
    def test_merge_3d_data_vec(self):
        # 1.1 读入，预处理数据
        slices = []
        for slice_path in self._demo_3d_file_list:
            slices.append(AnnBasedStereoExpData(slice_path))
        merged_data = merge(*slices, space_between='1um', reorganize_coordinate=False)

        import os

        # merged_data = AnnBasedStereoExpData("/mnt/d/projects/stereopy_dev/demo_data/3d.h5ad")

        ty_col = 'annotation'
        x_raw = merged_data.position[:, 0]
        y_raw = merged_data.position[:, 1]
        z_raw = merged_data.position_z.reshape(1, -1)[0]

        xli = x_raw.tolist()
        yli = y_raw.tolist()
        zli = z_raw.tolist()
        tyli = merged_data.cells[ty_col].tolist()

        # print(set(tyli))
        # pl = pv.Plotter()

        # 2. 计算mesh
        # from stereo.algorithm.gen_mesh import gen_mesh
        # merged_data = gen_mesh(merged_data, xli, yli, zli, tyli, method='delaunay', tol=1.5, eps_val=2, min_samples=5,
        #                        thresh_num=10, key_name='delaunay_3d')
        # merged_data = gen_mesh(merged_data, xli, yli, zli, tyli, method='march', mc_scale_factor=1.5, eps_val=2,
        #                        min_samples=5, thresh_num=10, key_name='march_cubes')
        merged_data = merged_data.tl.gen_mesh(cluster_res_key=ty_col, method='delaunay', tol=1.5, eps_val=2, min_samples=5,
                               thresh_num=10, key_name='delaunay_3d')
        merged_data = merged_data.tl.gen_mesh(cluster_res_key=ty_col, method='march', mc_scale_factor=1.5, eps_val=2,
                               min_samples=5, thresh_num=10, key_name='march_cubes')

        print('test data ready.')

        self._preprocess(merged_data)

        # 1.2 计算dpt
        import numpy as np

        merged_data.tl.result['iroot'] = np.flatnonzero(merged_data.cells[ty_col] == 'amnioserosa')[10]  # 羊膜浆膜类细胞
        merged_data.tl.dpt(n_branchings=0)

        ptime = merged_data.tl.result['dpt_pseudotime']

        # 2 画图：目前只包括数据准备

        ux, uy, uz, xm, ym, zm = merged_data.plt.plot_vec_3d(x_raw, y_raw, z_raw, ptime, num_cell=10, fil_type='gauss',
                                                             sigma_val=1, radius_val=3)

        # 中间结果可视化1：ux, uy, uz截面图
        import math

        def plot_csection(arr):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(arr)
            plt.show()

        plt_arr_x = ux[math.floor((ux.shape[0] - 1) / 2), :, :]
        plt_arr_y = uy[:, math.floor((uy.shape[1] - 1) / 2), :]
        plt_arr_z = uz[:, :, math.floor((uz.shape[2] - 1) / 2)]
        plot_csection(plt_arr_x)

        # 中间结果可视化2：matplotlib画quiver，注意：1. 目前z轴显示比例和x、y轴不同，实际上向量的z分量更大些； 2. 未展示mesh效果
        # # todo: notice me: xyz not of same scale
        print(xm.shape, ym.shape, zm.shape)
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        # ax.set_aspect('equal')  # error： not supported by matplotlib
        ax.set_box_aspect([1, 1, 1])  # neglected by matplotlib
        ax.quiver(xm, ym, zm, ux, uy, uz, length=0.5)
