import matplotlib.pyplot as plt

from .traj import Traj
from ..plot_base import PlotBase


class PlotClusterTraj3D(PlotBase):

    def plot_cluster_traj_3d(self, con, x_raw, y_raw, z_raw, ty,
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

        :return:
        """  # noqa
        traj = Traj(con, x_raw, y_raw, z_raw, ty, choose_ty)
        traj.gen_ty_all_no_dup_same_ord()

        mask_keep, keep_ty = traj.filter_minority(count_thresh)
        traj.revise_con_based_on_selection(keep_ty)

        if not choose_ty is None: # noqa
            traj.revise_con_based_on_selection(choose_ty)

        traj.gen_repre_x_y_z_by_ty(ty_repre_xyz)

        traj.get_con_pairs(lower_thresh_not_equal)

        if type_traj == 'curve':
            traj.compute_com_traj_li()
            print(traj.com_tra_li)
            print([[traj.ty_all_no_dup_same_ord[i] for i in li] for li in traj.com_tra_li])

            x_unknown_li_all_tra, y_unknown_li_all_tra, z_unknown_li_all_tra = traj.cal_position_param_curve(
                n_per_inter)
            com_tra_wei_li = traj.compute_weight_on_com_tra_li()
            print(com_tra_wei_li)
            return self._show_curve(x_unknown_li_all_tra, y_unknown_li_all_tra, z_unknown_li_all_tra, traj.com_tra_li,
                                    com_tra_wei_li)

        else:
            traj.compute_com_traj_li()
            print(traj.com_tra_li)
            print([[traj.ty_all_no_dup_same_ord[i] for i in li] for li in traj.com_tra_li])
            x_li, y_li, z_li = traj.cal_position_param_straight()
            wei_li = traj.compute_weight_on_pairs()
            return self._show_straight(x_li, y_li, z_li, traj.com_tra_li, wei_li)

    def _plot_line(self, x_unknown, y_unknown, z_unknown, ax, wei):
        ax.plot(x_unknown, y_unknown, z_unknown, linewidth=wei * 3, c='b')
        return

    def _show_curve(self, x_unknown_li_all_tra, y_unknown_li_all_tra, z_unknown_li_all_tra, com_tra_li, com_tra_wei_li):
        figure = plt.figure()
        ax = figure.add_subplot(projection='3d')
        # 画轨迹连线
        # self.com_tra_li: [[1, 18, 10], [2, 12, 18], [3, 16, 0, 15, 12], [6, 7, 8, 19], [8, 11], [13, 4, 7, 9, 5, 17, 16], [9, 14]]  # noqa
        for i, sin_tra in enumerate(com_tra_li):  # 对每条完整的轨迹
            for j in range(len(sin_tra) - 1):  # 对于这条轨迹每一个截断
                self._plot_line(x_unknown_li_all_tra[i][j],
                                y_unknown_li_all_tra[i][j],
                                z_unknown_li_all_tra[i][j],
                                ax,
                                com_tra_wei_li[i][j])
        return figure

    def _show_straight(self, x_li, y_li, z_li, con_pair, wei_li):
        figure = plt.figure()
        ax = figure.add_subplot(projection='3d')
        for i in range(con_pair.shape[0]):
            self._plot_line([x_li[i][0], x_li[i][1]],
                            [y_li[i][0], y_li[i][1]],
                            [z_li[i][0], z_li[i][1]],
                            ax, wei_li[i])
        return figure
