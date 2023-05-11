import os
import unittest

from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.utils.data_helper import merge


class TestMerge3DData(unittest.TestCase):

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

    def test_merge_3d_data(self):
        slices = []
        for slice_path in os.listdir("/mnt/d/projects/stereopy_dev/demo_data/3d/"):
            slices.append(AnnBasedStereoExpData("/mnt/d/projects/stereopy_dev/demo_data/3d/" + slice_path))
        merged_data = merge(*slices, space_between='10nm')

        print(merged_data)
        self._preprocess(merged_data)
        ty_col = 'annotation'
        merged_data.cells[ty_col] = merged_data.cells[ty_col].astype('category')
        merged_data.tl.paga(groups=ty_col)

        x_raw = merged_data.position[:, 0]
        y_raw = merged_data.position[:, 1]
        z_raw = merged_data.position_z

        ty = merged_data.cells[ty_col].to_numpy()
        con = merged_data.tl.result['paga']['connectivities'].todense()  # arr (n_clus, n_clus)
        con_tree = merged_data.tl.result['paga']['connectivities_tree'].todense()

        choose_ty = list(set(ty))
        choose_ty = ['CNS', 'epidermis', 'salivary gland', 'carcass']
        type_traj = 'curve'

        ty_repre_xyz = {}  # type -- np.array(3,)
        mesh_each_ty = []  # each element: mesh of each type, element with the same order as choose_ty

        import numpy as np
        from stereo.algorithm.gen_mesh_3d import ThreeDimGroup

        for ty_name in choose_ty:
            tdg = ThreeDimGroup(list(x_raw), list(y_raw), list(z_raw), list(ty), ty_name=ty_name, eps_val=2, min_samples=3, thresh_num=10)
            try:
                mesh, mesh_li = tdg.create_mesh_of_type(method='march', mc_scale_factor=1.5)
                xyz_repre = tdg.find_repre_point(mesh_li, x_ran_sin=2.25)
            except:  # mesh calculation failed
                mesh = None
                xyz_repre = np.array([x_raw[ty == ty_name].mean(), y_raw[ty == ty_name].mean(), z_raw[ty == ty_name].mean()])
            mesh_each_ty.append(mesh)
            ty_repre_xyz[ty_name] = xyz_repre

        from stereo.plots.plot_cluster_traj_3d.plot_cluster_traj_3d import PlotClusterTraj3D
        merged_data.plt.plot_cluster_traj_3d(con_tree, x_raw, y_raw, z_raw, ty, choose_ty, ty_repre_xyz, type_traj=type_traj)

