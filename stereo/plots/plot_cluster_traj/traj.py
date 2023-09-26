import itertools
import math
from collections import Counter

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

from .interp import generate_cubic_interp_points
from .interp import generate_linear_interp_points


class Traj:
    def __init__(self, con, x_raw, y_raw, ty):
        self.con = con
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.ty = ty
        self.con_pair = None  # pi_index_li
        self.x_rep = None
        self.y_rep = None
        self.ty_rep = None  # same order as connectivity matrix index
        self.com_tra_li = None
        self.d_avg = None

    def assign_ty_rep(self):
        ty_rep_li = []
        for ty_val in list(dict.fromkeys(self.ty)):  # remove duplication while keeping the order
            ty_rep_li.append(ty_val)
        self.ty_rep = np.array(ty_rep_li)

    def filter_minority(self, count_thresh):
        """
        filter out spots with occurrences less than threshold value
        :param count_thresh:
        :return: mask with the same size as self.x_raw and self.ty
        """
        ty_count_dic = dict(Counter(self.ty))
        ty_arr = np.array(list(ty_count_dic.keys()))
        val_arr = np.array(list(ty_count_dic.values()))
        keep_ind = np.where(val_arr > count_thresh)

        keep_ty = ty_arr[keep_ind]
        mask_keep = np.isin(self.ty, keep_ty)
        return mask_keep, keep_ty

    def revise_con_based_on_selection(self, choose_ty):
        """
        revise self.con based on the selected types
        :param choose_ty: list, np.NdArray, or tuple
        :return: revise self.con
        """
        choose_i = np.array([np.where(self.ty_rep == ele)[0][0] for ele in choose_ty])
        choose_ind = np.array([ele for ele in itertools.permutations(choose_i, 2)])  # np.NdArray, (n,2)
        choose_mtx = np.zeros(self.con.shape, dtype=bool)
        choose_mtx[(choose_ind[:, 0], choose_ind[:, 1])] = 1

        self.con[~choose_mtx] = 0

    def estimate_avg_dis(self):
        """
        find average distance between points, based on the tentative hypothesis that points are evenly distributed

        :param x_arr:
        :param y_arr:
        :return: d_avg: average distance
        """

        x_arr = self.x_raw.copy()
        y_arr = self.y_raw.copy()

        x_arr = x_arr - x_arr.min()
        y_arr = y_arr - y_arr.min()

        x_arr = x_arr.astype(np.int32)
        y_arr = y_arr.astype(np.int32)

        shorter_len = np.array([y_arr.max() + 1, x_arr.max() + 1]).min()
        scale = 50 / shorter_len

        s_arr = np.zeros(shape=(math.ceil(y_arr.max() * scale + 1),
                                math.ceil(x_arr.max() * scale + 1)))  # e.g. 2.1 * 2.6 + 1 = 6.46 -> 7, 支持0~6的索引
        s_arr[np.ceil(y_arr * scale).astype(np.int32), np.ceil(x_arr * scale).astype(
            np.int32)] = 1  # e.g 最大值：2.1 * 2.6 = 5.46 -> 6

        s = np.sum(s_arr) / scale ** 2
        d_avg = math.sqrt(s / x_arr.shape[0])

        self.d_avg = d_avg
        return

    def cal_repre_x_y_by_ty(self, eps_co, check_surr_co):
        """
        calculate representative positions of spots for annotation, as central of gravity (CoG) of spots within a type of spots, if the CoG is located inside # noqa
        of spots region, or else as the same coordinate with a random spot within the type of spots

        :return: x_rep: (n_type,) y_rep：(n_type,), ty_rep: (n_type,)
        """
        x_rep_li = []
        y_rep_li = []

        for ty_val in list(dict.fromkeys(self.ty)):
            ind = np.where(self.ty == ty_val)
            x_clus = self.x_raw[ind]  # (n_spot_in_this_cluster,)
            y_clus = self.y_raw[ind]

            # 二次分群， 找到每个点的坐标分群标签
            X = np.concatenate((np.expand_dims(x_clus, axis=1), np.expand_dims(y_clus, axis=1)), axis=1)
            dbscan_clus = DBSCAN(eps=eps_co * self.d_avg, min_samples=3).fit(X)
            dbscan_labels_arr = dbscan_clus.labels_  # (n_spot_in_this_cluster,)

            # 预处理：生成dbscan标签和对应点数量两个矩阵
            dbscan_labels_c = dict(Counter(dbscan_labels_arr))
            dbscan_labels_c.pop(-1, None)  # 去掉离群点
            dbscan_labels = np.array(list(dbscan_labels_c.keys()))  # (n_new_clus-1,)
            dbscan_vals = np.array(list(dbscan_labels_c.values()))  # (n_new_clus-1,)  # 存：对应点数量

            # 全部点都被认为是离群点
            if dbscan_vals.shape[0] == 0:
                x_rep = x_clus[0]  # 随机找其中一个点作为代表
                y_rep = y_clus[0]
            # 不全是离群点
            else:
                # 找到点数最多的标签，作为最大群标签
                ind_clus = np.where(dbscan_vals == dbscan_vals.max())[0][0]
                # print(dbscan_labels)
                clus_label = dbscan_labels[ind_clus]

                # 找到最大群的群中心坐标
                ind_spot = np.where(dbscan_labels_arr == clus_label)

                x_mean = x_clus[ind_spot].mean()
                y_mean = y_clus[ind_spot].mean()

                # 判断该点能否作为代表点：其周围是否有spot
                ind_in_ran = np.where((x_clus > x_mean - check_surr_co * self.d_avg)
                                      & (x_clus < x_mean + check_surr_co * self.d_avg)
                                      & (y_clus > y_mean - check_surr_co * self.d_avg)
                                      & (y_clus < y_mean + check_surr_co * self.d_avg))
                # 重心附近没有spot
                if ind_in_ran[0].shape[0] == 0:
                    x_rep = x_clus[0]
                    y_rep = y_clus[0]
                # 重心附近有spot
                else:
                    x_rep = x_mean
                    y_rep = y_mean

            # 测试：重心位置
            # plt.figure()
            # plt.scatter(x_clus, y_clus, c=dbscan_clus.labels_, s=1, cmap='rainbow')
            # plt.colorbar()
            # plt.scatter(x_rep, y_rep, s=10, marker='x')
            # plt.xlim(left=self.x_raw.min(), right=self.x_raw.max())
            # plt.ylim(bottom=self.y_raw.min(), top=self.y_raw.max())
            # plt.gca().set_aspect('equal', adjustable='box')
            # sdir = 'E:/ANALYSIS_ALGORITHM/cell_trajectory_analysis/result/dbscan'
            # plt.savefig(os.path.join(sdir, str(ty_val) + '.tif'))
            # plt.close()

            x_rep_li.append(x_rep)
            y_rep_li.append(y_rep)
        self.x_rep = np.array(x_rep_li)
        self.y_rep = np.array(y_rep_li)
        return

    def get_con_pairs(self, lower_thresh_not_equal):
        """
        calculate pairs of connected nodes
        :param lower_thresh_not_equal:
        :return: ndarray, (n,2)
        """
        pi_index = np.where(self.con > lower_thresh_not_equal)
        pi_index_li = [[pi_index[0][i], pi_index[1][i]] for i in
                       range(pi_index[0].shape[0])]  # 将clus的关系从邻接矩阵转化成[[], []]存储
        pi_index_li = [li for li in pi_index_li if not li[0] == li[1]]  # 去掉起始和终点相同的li
        pi_index_li = [sorted(li) for li in pi_index_li]  # 是li从较小index到较大index
        pi_index_li = np.array([li for li, _ in itertools.groupby(pi_index_li)])  # 去掉重复的li

        self.con_pair = pi_index_li
        return

    def compute_com_traj_li(self):
        """
        Compute a list of complete trajectories
        :param lower_thresh_not_equal: threshold value that element on con matrix should surpass, to be considered a valid connection # noqa
        :return: com_tra_li: list of list of indices, e.g. [[0,1], [1,3,5,2]]
        """

        def find_end_p_gathering_from_intervals(pi_index_arr):
            """
            find point on the edge of trajectories
            :param pi_index_arr: a gathering of all intervals, list (without duplication) of list (without duplication)
                                    with 2 elements showing index in increasing order
            :return: a gathering of all edge points, arr: (n,)
            """
            uni_arr_ini, counts_ini = np.unique(pi_index_arr, return_counts=True)  # (num_unique,)
            edge_arr = uni_arr_ini[np.where(counts_ini == 1)]  # 在轨迹上属于起始或结束的cluster，每次循环均会更新 (n,)
            return edge_arr

        def find_a_complete_trajectory_and_renew_intervals(start, pi_index_arr):
            """
            As its name

            :param start: index of start point, integer
            :param pi_index_arr:  a gathering of all intervals, list (without duplication) of list (without duplication)
                                    with 2 elements showing index in increasing order
            :return: com_tra: a complete trajectory,  list (can have duplicated points)
                     pi_index_arr: updated pi_index_arr, with those appended to com_tra removed
            """
            com_tra = [start]
            next_ran = list(set(np.array([[ele for ele in li if not ele == start] for li in pi_index_arr if
                                          start in li]).flatten()))  # 所有下一个点的集合， list
            while len(next_ran) >= 1:
                next = next_ran[0]  # next point
                inter = np.array(sorted([start, next]))

                # 从pi_index_arr中去除此线段
                del_ind = np.where((pi_index_arr[:, 0] == inter[0]) & (
                        pi_index_arr[:, 1] == inter[1]))  # row index to be deleted, (1,)
                pi_index_arr = np.delete(pi_index_arr, del_ind, axis=0)

                # 把此线段信息加入到com_tra中
                com_tra.append(next)

                # 更新next_ran
                next_ran = list(set(np.array([[ele for ele in li if not ele == next] for li in pi_index_arr if
                                              next in li]).flatten()))  # 所有下一个点的集合， list

                # 更新start
                start = next
            return com_tra, pi_index_arr

        # 完整轨迹： 输入： con, lower_thresh_not_equal, 输出：com_tra_li: 轨迹 [[i1, i2, ...], [i7, i8, ...]]

        # 2. 计算完整轨迹的集合
        pi_index_arr = np.array(self.con_pair)  # 待选线段的集合，每次循环都会更新 (n, 2)
        edge_arr = find_end_p_gathering_from_intervals(pi_index_arr)
        com_tra_li = []  # list of complete trajectories, 每次循环都会更新, [[], [], ...]
        while edge_arr.shape[0] >= 1:
            # 找到一个边界点
            start = edge_arr[0]  # 起始或结束cluster的index
            # 找到该边界点对应的完整轨迹，同时修改interval的集合
            com_tra, pi_index_arr = find_a_complete_trajectory_and_renew_intervals(start, pi_index_arr)
            # 更新在轨迹上属于起始或结束的cluster
            edge_arr = find_end_p_gathering_from_intervals(pi_index_arr)
            # 更新所有完整轨迹的集合
            com_tra_li.append(com_tra)
        self.com_tra_li = com_tra_li
        return

    def cal_position_param_curve(self, n_per_inter):
        """
        calculate position parameter for plotting curve
        :param n_per_inter: number of interpolated points per interval
        :return: x_unknown_li_all_tra: [[np.NdArray]], y_unknown_li_all_tra: [[np.NdArray]]
        """
        # self.com_tra_li: [[1, 18, 10], [2, 12, 18], [3, 16, 0, 15, 12], [6, 7, 8, 19], [8, 11], [13, 4, 7, 9, 5, 17, 16], [9, 14]] # noqa
        x_unknown_li_all_tra = []
        y_unknown_li_all_tra = []
        for i, sin_tra in enumerate(self.com_tra_li):  # 对每条完整的轨迹
            x_known = self.x_rep[sin_tra]  # (n_sin_tra_point,)
            y_known = self.y_rep[sin_tra]

            # 1. 2个节点的情况
            if x_known.shape[0] == 2:
                x_unknown, y_unknown = generate_linear_interp_points(x_known, y_known, n_per_inter)
                x_unknown_li = [x_unknown]
                y_unknown_li = [y_unknown]

            # 2. 至少3个节点的情况
            else:
                # 2.1. 对正好有3个节点的情况进行额外处理
                if x_known.shape[0] == 3:
                    # 将前两个点中间点引入,将3个节点转化为4个节点
                    x_known = np.insert(x_known, 1, (x_known[0] + x_known[1]) / 2)
                    y_known = np.insert(y_known, 1, (y_known[0] + y_known[1]) / 2)

                    x_unknown_li, y_unknown_li = generate_cubic_interp_points(x_known, y_known,
                                                                              n_per_inter)  # [arr, arr, ...]

                    # 将多引入导致的两段，重新合并在一起
                    x_unknown_li = [np.concatenate((x_unknown_li[0], x_unknown_li[1]))] + x_unknown_li[2:]
                    y_unknown_li = [np.concatenate((y_unknown_li[0], y_unknown_li[1]))] + y_unknown_li[2:]

                else:
                    x_unknown_li, y_unknown_li = generate_cubic_interp_points(x_known, y_known,
                                                                              n_per_inter)  # [arr, arr, ...]

            x_unknown_li_all_tra.append(x_unknown_li)  # [[arr, arr, ...], [arr, arr, ...], ]
            y_unknown_li_all_tra.append(y_unknown_li)

        return x_unknown_li_all_tra, y_unknown_li_all_tra

    def cal_position_param_straight(self):
        """
        calculate position parameter for plotting straight lines
        :return: x_li: [arr, ...], each array is a np.array([x_start, x_end])
                 y_li: [arr, ...]
        """
        x_li = []  # [arr, arr]
        y_li = []
        for i in range(self.con_pair.shape[0]):  # for each pair of nodes
            sin_con_pair = self.con_pair[i]  # (2,)

            x_start = self.x_rep[sin_con_pair[0]]
            y_start = self.y_rep[sin_con_pair[0]]

            x_end = self.x_rep[sin_con_pair[1]]
            y_end = self.y_rep[sin_con_pair[1]]

            x_li.append(np.array([x_start, x_end]))
            y_li.append(np.array([y_start, y_end]))
        return x_li, y_li

    def compute_weight_on_com_tra_li(self):
        """
        compute the weight of each interval on all full trajectories
        :param com_tra_li: list of list
        :return: com_tra_wei_li (list of list, with the inner list one element shorter than the one in com_tra_li)
        """
        com_tra_wei_li = []  # complete trajectory weight list
        for i in range(len(self.com_tra_li)):
            sin_com_tra_wei_li = []  # single complete trajectory weight list
            for j in range(len(self.com_tra_li[i]) - 1):
                node_1 = self.com_tra_li[i][j]
                node_2 = self.com_tra_li[i][j + 1]

                if self.con[node_1, node_2] > 0 and self.con[node_2, node_1] > 0:
                    wei = np.min(np.array([self.con[node_1, node_2], self.con[node_2, node_1]]))
                else:
                    wei = np.max(np.array([self.con[node_1, node_2], self.con[node_2, node_1]]))

                sin_com_tra_wei_li.append(wei)
            com_tra_wei_li.append(sin_com_tra_wei_li)
        return com_tra_wei_li

    def compute_weight_on_pairs(self):
        wei_li = []
        for i in range(self.con_pair.shape[0]):  # 对每对点
            sin_con_pair = self.con_pair[i]  # (2,)
            edge_1 = sin_con_pair[0]
            edge_2 = sin_con_pair[1]
            wei_val = np.max(np.array([self.con[edge_1, edge_2], self.con[edge_2, edge_1]]))
            wei_li.append(wei_val)
        return wei_li

    def show_scatter(self, spot_size, spot_alpha, seed_val, num_legend_per_col, tick_step, mask_keep):
        """
        plot cells or bins as background

        :param spot_size:
        :param cmap:
        :param spot_alpha:
        :return:
        """
        # plot scatter
        np.random.seed(seed_val)
        cmap_val = mpl.colors.ListedColormap(np.random.rand(256, 3))
        dict_ty_float = dict(zip(set(self.ty[mask_keep]), np.arange(len(set(self.ty[mask_keep])))))  #

        im = plt.scatter(self.x_raw[mask_keep], self.y_raw[mask_keep],
                         c=[dict_ty_float[ele] for ele in self.ty[mask_keep]], s=spot_size, cmap=cmap_val,
                         alpha=spot_alpha, linewidths=0)

        # plot legend
        uni_ele = np.unique(self.ty[mask_keep])
        colors = [im.cmap(im.norm(dict_ty_float[val])) for val in uni_ele]

        patches = [mpatches.Patch(color=colors[i], label=uni_ele[i]) for i in
                   range(len(colors))]  # nan excluded:  if not np.isnan(uni_ele[i])

        ncols = len(patches) // (num_legend_per_col + 1) + 1
        plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncols,
                   framealpha=0)  # loc和bbox_to_anchor组合，loc表示legend的锚点，bbox_to_anchor表示锚点相对图的位置

        # set ticks
        x_tick = np.arange(np.floor(self.x_raw.min() / tick_step) * tick_step,
                           np.ceil(self.x_raw.max() / tick_step) * tick_step, step=tick_step).astype(np.int32)
        y_tick = np.arange(np.floor(self.y_raw.min() / tick_step) * tick_step,
                           np.ceil(self.y_raw.max() / tick_step) * tick_step, step=tick_step).astype(np.int32)
        plt.xticks(ticks=x_tick, labels=x_tick)
        plt.yticks(ticks=y_tick, labels=y_tick)

    def show_ty_label(self, text_size, choose_ty, keep_ty):
        """
        put labels on representative position of each type
        :param text_size:
        :return:
        """
        if choose_ty is None:
            if keep_ty is None:
                return
            else:
                plt_ty = keep_ty
        elif keep_ty is None:
            plt_ty = choose_ty
        else:
            plt_ty = [ele for ele in choose_ty if ele in keep_ty]

        for i in range(self.x_rep.shape[0]):
            if not plt_ty is None:  # types were selected # noqa
                if not self.ty_rep[i] in plt_ty:
                    continue
            try:
                int(self.ty_rep[i])
                if int(self.ty_rep[i]) == float(self.ty_rep[i]):
                    ty_val = str(int(self.ty_rep[i]))
                else:
                    ty_val = str(float(self.ty_rep[i]))
            except Exception:
                ty_val = self.ty_rep[i]

            plt.text(self.x_rep[i], self.y_rep[i], ty_val,
                     ha='center', va='center', fontsize=text_size,
                     bbox=dict(boxstyle='square', facecolor='white'))

    @staticmethod
    def _plot_line(x_unknown, y_unknown, line_alpha, line_width_co, wei, c_val, uni_lwidth):
        if uni_lwidth:
            plt.plot(x_unknown, y_unknown, alpha=line_alpha, linewidth=line_width_co, c=c_val)
        else:
            plt.plot(x_unknown, y_unknown, alpha=line_alpha, linewidth=line_width_co * wei, c=c_val)
        return

    def show_curve(self, x_unknown_li_all_tra, y_unknown_li_all_tra, com_tra_wei_li, line_alpha, line_width_co,
                   line_color, uni_lwidth):
        # 画轨迹连线
        # self.com_tra_li: [[1, 18, 10], [2, 12, 18], [3, 16, 0, 15, 12], [6, 7, 8, 19], [8, 11], [13, 4, 7, 9, 5, 17, 16], [9, 14]] # noqa
        for i, sin_tra in enumerate(self.com_tra_li):  # 对每条完整的轨迹
            for j in range(len(sin_tra) - 1):  # 对于这条轨迹每一个截断
                self._plot_line(x_unknown_li_all_tra[i][j], y_unknown_li_all_tra[i][j], line_alpha, line_width_co,
                                com_tra_wei_li[i][j], line_color, uni_lwidth)
        return

    def show_straight(self, x_li, y_li, wei_li, line_alpha, line_width_co, line_color, uni_lwidth):
        for i in range(self.con_pair.shape[0]):
            self._plot_line([x_li[i][0], x_li[i][1]], [y_li[i][0], y_li[i][1]], line_alpha, line_width_co, wei_li[i],
                            line_color, uni_lwidth)

        return
