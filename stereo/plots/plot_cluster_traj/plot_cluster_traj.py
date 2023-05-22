import matplotlib.pyplot as plt

from .traj import Traj
from ..plot_base import PlotBase


class PlotClusterTraj(PlotBase):
    def plot_cluster_traj(
            self,
            con, x_raw, y_raw, ty,
            save_dir,
            save_na,
            count_thresh=0,
            eps_co=3,
            check_surr_co=0.75,  # 0.75
            choose_ty=None,
            type_traj='curve',
            lower_thresh_not_equal=0.5,
            show_scatter=True,
            seed_val=0,
            num_legend_per_col=12,
            tick_step=2500,
            spot_alpha=0.7,
            spot_size=3,
            line_alpha=1,
            line_width_co=1,
            line_color='#7570b3',
            uni_lwidth=False,
            text_size=5,
            n_per_inter=100,
            dpi_save=1000):

        # TODO: 描述
        # TODO: 对输入进行断言

        # generating data for plotting
        traj = Traj(con, x_raw, y_raw, ty)
        traj.assign_ty_rep()

        mask_keep, keep_ty = traj.filter_minority(count_thresh)
        traj.revise_con_based_on_selection(keep_ty)

        if not choose_ty is None:
            traj.revise_con_based_on_selection(choose_ty)

        traj.estimate_avg_dis()

        traj.cal_repre_x_y_by_ty(eps_co, check_surr_co)

        traj.get_con_pairs(lower_thresh_not_equal)

        # plotting
        figure = plt.figure()

        if show_scatter:
            traj.show_scatter(spot_size, spot_alpha, seed_val, num_legend_per_col, tick_step, mask_keep)

        traj.show_ty_label(text_size, choose_ty, keep_ty)

        if type_traj == 'curve':
            traj.compute_com_traj_li()
            print([[traj.ty_rep[i] for i in li] for li in traj.com_tra_li])

            x_unknown_li_all_tra, y_unknown_li_all_tra = traj.cal_position_param_curve(n_per_inter)

            com_tra_wei_li = traj.compute_weight_on_com_tra_li()

            traj.show_curve(x_unknown_li_all_tra, y_unknown_li_all_tra, com_tra_wei_li,
                            line_alpha, line_width_co, line_color, uni_lwidth)
        else:
            x_li, y_li = traj.cal_position_param_straight()
            wei_li = traj.compute_weight_on_pairs()

            traj.show_straight(x_li, y_li, wei_li,
                               line_alpha, line_width_co, line_color, uni_lwidth)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        import os
        plt.savefig(os.path.join(save_dir, save_na), dpi=dpi_save, bbox_inches='tight')
        return figure
