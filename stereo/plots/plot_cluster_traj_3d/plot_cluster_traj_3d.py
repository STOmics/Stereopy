import os
import matplotlib.pyplot as plt

from plot_cluster_traj_3d.traj import Traj


# FIXME:  继承PlotBase?
class PlotClusterTraj3D():
    @staticmethod
    def main(con, x_raw, y_raw, z_raw, ty,
             choose_ty, ty_repre_xyz,
             count_thresh=0,
             type_traj='curve',
             lower_thresh_not_equal=0.5,
             n_per_inter=100,
             ):

        """
        Visualize PAGA result in 3D.

        :param con: connectivities matrix or connectivities_tree matrix output by PAGA.
        :param x_raw: 1d NdArray, involving raw x coordinates of all cells, or bin sets
        :param y_raw: 1d NdArray, involving raw y coordinates of all cells, or bin sets
        :param z_raw: 1d NdArray, involving raw z coordinates of all cells, or bin sets
        :param ty: 1d NdArray, involving cell types of all cells, or bin sets, can take the format of either string,
        int, or float
        :param choose_ty: list of selected cell types to be visualized
        :param ty_repre_xyz: dictionary, each key has the same value as a cell type, while each value is its unique
        representing point coordinate in 1d NdArray, with shape of (3,). All cell types that a user wishes to plot
        should be included in this dictionary
        :param count_thresh: Threshold value that filters out all cell types with number of cells or bin sets lower than it
        :param type_traj: Type of visualization, either in curve (parameter value: 'curve'), or in straight lines
        (parameter value: 'straight')
        :param lower_thresh_not_equal: Threshold value that neglects all element in parameter: con with value lower than it
        :param n_per_inter: Number of interpolated points between two connected cell types, if parameter: type_traj is 0.5

        :return: TODO: 主要由前端开发确定
        """

        # TODO: 对输入进行断言

        traj = Traj(con, x_raw, y_raw, z_raw, ty, choose_ty)
        traj.gen_ty_all_no_dup_same_ord()

        mask_keep, keep_ty = traj.filter_minority(count_thresh)
        traj.revise_con_based_on_selection(keep_ty)

        if not choose_ty is None:
            traj.revise_con_based_on_selection(choose_ty)

        traj.gen_repre_x_y_z_by_ty(ty_repre_xyz)

        traj.get_con_pairs(lower_thresh_not_equal)

        if type_traj == 'curve':
            traj.compute_com_traj_li()
            print(traj.com_tra_li)
            print([[traj.ty_all_no_dup_same_ord[i] for i in li] for li in traj.com_tra_li])

            x_unknown_li_all_tra, y_unknown_li_all_tra, z_unknown_li_all_tra = traj.cal_position_param_curve(n_per_inter)
            com_tra_wei_li = traj.compute_weight_on_com_tra_li()
            print(com_tra_wei_li)

            return x_unknown_li_all_tra, y_unknown_li_all_tra, z_unknown_li_all_tra, traj.com_tra_li, com_tra_wei_li

        else:
            traj.compute_com_traj_li()
            print(traj.com_tra_li)
            print([[traj.ty_all_no_dup_same_ord[i] for i in li] for li in traj.com_tra_li])
            x_li, y_li, z_li = traj.cal_position_param_straight()
            wei_li = traj.compute_weight_on_pairs()
            return x_li, y_li, z_li, traj.con_pair, wei_li


def test():
    import scanpy
    import anndata
    import numpy as np
    from gen_mesh_3d import ThreeDimGroup

    def _plot_line(x_unknown, y_unknown, z_unknown, ax, wei):
        ax.plot(x_unknown, y_unknown, z_unknown, linewidth=wei * 3, c='b')
        return

    def _show_curve(x_unknown_li_all_tra, y_unknown_li_all_tra, z_unknown_li_all_tra, com_tra_li, com_tra_wei_li):
        ax = plt.figure().add_subplot(projection='3d')
        # 画轨迹连线
        # self.com_tra_li: [[1, 18, 10], [2, 12, 18], [3, 16, 0, 15, 12], [6, 7, 8, 19], [8, 11], [13, 4, 7, 9, 5, 17, 16], [9, 14]]
        for i, sin_tra in enumerate(com_tra_li):  # 对每条完整的轨迹
            for j in range(len(sin_tra) - 1):  # 对于这条轨迹每一个截断
                _plot_line(x_unknown_li_all_tra[i][j],
                           y_unknown_li_all_tra[i][j],
                           z_unknown_li_all_tra[i][j],
                           ax,
                           com_tra_wei_li[i][j])
        return

    def _show_straight(x_li, y_li, z_li, con_pair, wei_li):
        ax = plt.figure().add_subplot(projection='3d')
        for i in range(con_pair.shape[0]):
            _plot_line([x_li[i][0], x_li[i][1]],
                       [y_li[i][0], y_li[i][1]],
                       [z_li[i][0], z_li[i][1]],
                       ax, wei_li[i])
        return

    # 1. 准备数据
    # 1.1 读入，预处理数据
    dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/data/fruitfly_embryo/bin_recons_spot_level/concat'
    spatial_col = 'spatial_elas'
    ty_col = 'annotation'
    adata = anndata.read(os.path.join(dir, '3d.h5ad'))

    # 1.2 计算paga
    # notice: 已经做了pca和neighbors
    adata.obs[ty_col] = adata.obs[ty_col].astype('category')
    scanpy.tl.paga(adata, groups=ty_col)

    # 1.3 获取输入参数
    x_raw = adata.obsm[spatial_col][:, 0]
    y_raw = adata.obsm[spatial_col][:, 1]
    z_raw = adata.obsm[spatial_col][:, 2]
    ty = adata.obs[ty_col].to_numpy()
    con = adata.uns['paga']['connectivities'].todense()  # arr (n_clus, n_clus)
    con_tree = adata.uns['paga']['connectivities_tree'].todense()

    # 2 画图：目前只包括数据准备
    # 2.1 计算mesh，代表点坐标
    choose_ty = list(set(ty))
    choose_ty = ['CNS', 'foregut', 'salivary gland', 'carcass']
    type_traj = 'curve'

    ty_repre_xyz = {}  # type -- np.array(3,)
    mesh_each_ty = []  # each element: mesh of each type, element with the same order as choose_ty
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

    # 2.2 计算画轨迹需要的相应数据
    plot_cluster_traj_3d = PlotClusterTraj3D()
    if type_traj == 'curve':
        x_unknown_li_all_tra, y_unknown_li_all_tra, z_unknown_li_all_tra, com_tra_li, com_tra_wei_li \
            = plot_cluster_traj_3d.main(con_tree, x_raw, y_raw, z_raw, ty, choose_ty, ty_repre_xyz, type_traj=type_traj)
    else:
        x_li, y_li, z_li, con_pair, wei_li \
            = plot_cluster_traj_3d.main(con_tree, x_raw, y_raw, z_raw, ty, choose_ty, ty_repre_xyz, type_traj=type_traj)

    # 2.3 数据展示
    if type_traj == 'curve':
        _show_curve(x_unknown_li_all_tra, y_unknown_li_all_tra, z_unknown_li_all_tra, com_tra_li, com_tra_wei_li)
    else:
        _show_straight(x_li, y_li, z_li, con_pair, wei_li)


if __name__ == '__main__':
    test()



