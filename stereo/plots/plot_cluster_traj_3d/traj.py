import itertools
from collections import Counter

import numpy as np

from .interp import generate_cubic_interp_points
from .interp import generate_linear_interp_points


class Traj:
    def __init__(self, con, x_raw, y_raw, z_raw, ty, choose_ty):
        self.con = con
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.z_raw = z_raw
        self.ty = ty
        self.choose_ty = choose_ty
        self.con_pair = None  # pi_index_li
        self.x_rep = None  # same order as self.con matrix, filled with None if not in self.choose_ty
        self.y_rep = None
        self.z_rep = None
        self.ty_all_no_dup_same_ord = None  # same order as self.con matrix
        self.com_tra_li = None

    def gen_ty_all_no_dup_same_ord(self):
        self.ty_all_no_dup_same_ord = np.array(list(dict.fromkeys(self.ty)))

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
        choose_i = np.array([np.where(self.ty_all_no_dup_same_ord == ele)[0][0] for ele in choose_ty])
        choose_ind = np.array([ele for ele in itertools.permutations(choose_i, 2)])  # np.NdArray, (n,2)
        choose_mtx = np.zeros(self.con.shape, dtype=bool)
        choose_mtx[(choose_ind[:, 0], choose_ind[:, 1])] = 1

        self.con[~choose_mtx] = 0

    def gen_repre_x_y_z_by_ty(self, ty_repre):
        """
        calculate representative positions of spots for annotation, as central of gravity (CoG) of spots within a type of spots, if the CoG is located inside
        of spots region, or else as the same coordinate with a random spot within the type of spots
        :param: ty_repre: dictionary. keys mean name of type, values are np array of representing points coordinates
        :return: x_rep: (n_type,), y_rep：(n_type,), z_rep：(n_type,)
        """  # noqa
        x_rep_li = []
        y_rep_li = []
        z_rep_li = []
        for ty_val in self.ty_all_no_dup_same_ord:
            if ty_val in self.choose_ty:
                x_rep, y_rep, z_rep = ty_repre[ty_val]
            else:
                x_rep = y_rep = z_rep = None
            x_rep_li.append(x_rep)
            y_rep_li.append(y_rep)
            z_rep_li.append(z_rep)

        self.x_rep = np.array(x_rep_li)  # same order as choose_ty
        self.y_rep = np.array(y_rep_li)
        self.z_rep = np.array(z_rep_li)

    def get_con_pairs(self, lower_thresh_not_equal):
        """
        calculate pairs of connected nodes
        :param lower_thresh_not_equal:
        :return: ndarray, (n,2)
        """
        pi_index = np.where(self.con > lower_thresh_not_equal)
        # 将clus的关系从邻接矩阵转化成[[], []]存储
        pi_index_li = [[pi_index[0][i], pi_index[1][i]] for i in range(pi_index[0].shape[0])]
        pi_index_li = [li for li in pi_index_li if not li[0] == li[1]]  # 去掉起始和终点相同的li
        pi_index_li = [sorted(li) for li in pi_index_li]  # 是li从较小index到较大index
        pi_index_li = np.unique(np.array(pi_index_li), axis=0)
        self.con_pair = pi_index_li
        return

    def compute_com_traj_li(self):
        """
        Compute a list of complete trajectories
        :param lower_thresh_not_equal: threshold value that element on con matrix should surpass, to be considered a valid connection
        :return: com_tra_li: list of list of indices, e.g. [[0,1], [1,3,5,2]]
        """  # noqa

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
        :return: x_unknown_li_all_tra: [[np.NdArray]], y_unknown_li_all_tra: [[np.NdArray]], z_unknown_li_all_tra: [[np.NdArray]]
        """  # noqa
        # self.com_tra_li: [[1, 18, 10], [2, 12, 18], [3, 16, 0, 15, 12], [6, 7, 8, 19], [8, 11], [13, 4, 7, 9, 5, 17, 16], [9, 14]]  # noqa
        x_unknown_li_all_tra = []
        y_unknown_li_all_tra = []
        z_unknown_li_all_tra = []
        for i, sin_tra in enumerate(self.com_tra_li):  # 对每条完整的轨迹
            x_known = self.x_rep[sin_tra]  # (n_sin_tra_point,)
            y_known = self.y_rep[sin_tra]
            z_known = self.z_rep[sin_tra]
            # 1. 2个节点的情况
            if x_known.shape[0] == 2:
                x_unknown, y_unknown, z_unknown = generate_linear_interp_points(x_known, y_known, z_known, n_per_inter)
                x_unknown_li = [x_unknown]
                y_unknown_li = [y_unknown]
                z_unknown_li = [z_unknown]

            # 2. 至少3个节点的情况
            else:
                # 2.1. 对正好有3个节点的情况进行额外处理
                if x_known.shape[0] == 3:
                    # 将前两个点中间点引入,将3个节点转化为4个节点
                    x_known = np.insert(x_known, 1, (x_known[0] + x_known[1]) / 2)
                    y_known = np.insert(y_known, 1, (y_known[0] + y_known[1]) / 2)
                    z_known = np.insert(z_known, 1, (z_known[0] + z_known[1]) / 2)
                    # [arr, arr, ...]
                    x_unknown_li, y_unknown_li, z_unknown_li = generate_cubic_interp_points(x_known, y_known, z_known,
                                                                                            n_per_inter)

                    # 将多引入导致的两段，重新合并在一起
                    x_unknown_li = [np.concatenate((x_unknown_li[0], x_unknown_li[1]))] + x_unknown_li[2:]
                    y_unknown_li = [np.concatenate((y_unknown_li[0], y_unknown_li[1]))] + y_unknown_li[2:]
                    z_unknown_li = [np.concatenate((z_unknown_li[0], z_unknown_li[1]))] + z_unknown_li[2:]
                else:
                    # [arr, arr, ...]
                    x_unknown_li, y_unknown_li, z_unknown_li = generate_cubic_interp_points(x_known, y_known, z_known,
                                                                                            n_per_inter)

            x_unknown_li_all_tra.append(x_unknown_li)  # [[arr, arr, ...], [arr, arr, ...], ]
            y_unknown_li_all_tra.append(y_unknown_li)
            z_unknown_li_all_tra.append(z_unknown_li)
        return x_unknown_li_all_tra, y_unknown_li_all_tra, z_unknown_li_all_tra

    def cal_position_param_straight(self):
        """
        calculate position parameter for plotting straight lines
        :return: x_li: [arr, ...], each array is a np.array([x_start, x_end])
                 y_li: [arr, ...]
        """
        x_li = []  # [arr, arr]
        y_li = []
        z_li = []
        for i in range(self.con_pair.shape[0]):  # for each pair of nodes
            sin_con_pair = self.con_pair[i]  # (2,)

            x_start = self.x_rep[sin_con_pair[0]]
            y_start = self.y_rep[sin_con_pair[0]]
            z_start = self.z_rep[sin_con_pair[0]]

            x_end = self.x_rep[sin_con_pair[1]]
            y_end = self.y_rep[sin_con_pair[1]]
            z_end = self.z_rep[sin_con_pair[1]]

            x_li.append(np.array([x_start, x_end]))
            y_li.append(np.array([y_start, y_end]))
            z_li.append(np.array([z_start, z_end]))
        return x_li, y_li, z_li

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
