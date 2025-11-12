import matplotlib.pyplot as plt
import numpy as np

from .traj import Traj
from .. import base_scatter
from ..decorator import plot_scale
from ..plot_base import PlotBase


class PlotClusterTraj(PlotBase):

    @plot_scale
    def plot_cluster_traj(
            self,
            con,
            x_raw,
            y_raw,
            ty,
            count_thresh=0,
            eps_co=3,
            check_surr_co=0.75,  # 0.75
            choose_ty=None,
            type_traj='curve',
            lower_thresh_not_equal=0.5,
            show_scatter=True,
            line_alpha=1,
            line_width_co=1,
            line_color='#0000e5',
            uni_lwidth=False,
            text_size=5,
            n_per_inter=100,
            dpi_save=1000,
            palette='stereo_30',
            **kwargs
    ):
        """
        Plot trajectory between different cell clusters.
        
        :param con: connectivities matrix or connectivities_tree matrix output by PAGA
        :param x_raw: 1d NdArray, involving raw x coordinates of all cells, or bin sets
        :param y_raw: 1d NdArray, involving raw y coordinates of all cells, or bin sets
        :param ty: 1d NdArray, involving cell types of all cells, or bin sets, can take the format of either string, int, or float
        :param count_thresh: Threshold value that filters out all cell types with number of cells or bin sets lower than it
        :param eps_co: eps parameter for computing DBSCAN when postprocessing each cluster, check sklearn.cluster.DBSCAN for more details
        :param check_surr_co: a point has to have points that are check_surr_co* average spot-wise distance for it to be considered representative point of the cluster.
        :param choose_ty: cell types to plot, chosen by users
        :param type_traj: Type of visualization, either in curve (parameter value: 'curve'), or in straight lines (parameter value: 'straight')
        :param lower_thresh_not_equal: Threshold value that neglects all element in parameter: con with value lower than it
        :param show_scatter: show spots as scatters or not
        :param line_alpha: alpha of lines, 0-1
        :param line_width_co: linewidth
        :param line_color: color of line
        :param uni_lwidth: lines between different clusters to have different width or not
        :param text_size: size of labels of each cluster
        :param n_per_inter: number of interpolated points between each two clusters when plotting
        :param dpi_save: dpi when saving figures
        :param palette: palette
        """
        # generating data for plotting
        traj = Traj(con, x_raw, y_raw, ty)
        traj.assign_ty_rep()
        _, keep_ty = traj.filter_minority(count_thresh)
        traj.revise_con_based_on_selection(keep_ty)
        if not choose_ty is None:  # noqa
            traj.revise_con_based_on_selection(choose_ty)
        traj.estimate_avg_dis()
        traj.cal_repre_x_y_by_ty(eps_co, check_surr_co)
        traj.get_con_pairs(lower_thresh_not_equal)

        # plotting
        figure = plt.figure(dpi=dpi_save)
        if show_scatter:
            figure = base_scatter(x_raw, y_raw, palette=palette, hue=np.array(ty), **kwargs)

        traj.show_ty_label(text_size, choose_ty, keep_ty)

        if type_traj == 'curve':
            traj.compute_com_traj_li()
            x_unknown_li_all_tra, y_unknown_li_all_tra = traj.cal_position_param_curve(n_per_inter)
            com_tra_wei_li = traj.compute_weight_on_com_tra_li()
            traj.show_curve(x_unknown_li_all_tra, y_unknown_li_all_tra, com_tra_wei_li, line_alpha, line_width_co,
                            line_color, uni_lwidth)
        else:
            x_li, y_li = traj.cal_position_param_straight()
            wei_li = traj.compute_weight_on_pairs()
            traj.show_straight(x_li, y_li, wei_li, line_alpha, line_width_co, line_color, uni_lwidth)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        return figure
