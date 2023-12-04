import math

import numpy as np
from scipy.interpolate import CubicSpline


def generate_linear_interp_points(x_known, y_known, n_per_inter):
    """
    generate linearly interpolated points between 2 physical points
    :param x_known: array-like, length of 2
    :param y_known: array-like, length of 2
    :param n_per_inter: number of interpolated points to generate
    :return:
    """
    x_unknown = np.linspace(x_known[0], x_known[1], num=n_per_inter)
    y_unknown = np.linspace(y_known[0], y_known[1], num=n_per_inter)
    return x_unknown, y_unknown


def _cal_knot_dis_on_path(x_known, y_known):
    """
    计算所有节点沿着轨迹的直线距离，称线段距离
    :param x_known:
    :param y_known:
    :param n_per_inter:
    :return:
    """
    # 每个节点的路径值存成li
    path_len = 0
    path_knot_len_li = [path_len]
    for i in range(len(x_known) - 1):
        inter_len = math.sqrt((x_known[i + 1] - x_known[i]) ** 2 + (y_known[i + 1] - y_known[i]) ** 2)
        path_len += inter_len
        path_knot_len_li.append(path_len)
    return path_knot_len_li


def _cal_segment_dis(i, path_knot_len_li, n_per_inter):
    """
    在一个节点段上均匀取若干个点(包括开始和最后一个点)，计算这些点在轨迹上的线段距离

    :param i: interval的序列位置，从0开始
    :param path_knot_len_li: 每个节点的直线段轨迹距离
    :param n_per_inter: interval上插值的个数
    :return:
    """
    s_inter = np.linspace(path_knot_len_li[i], path_knot_len_li[i + 1], num=n_per_inter)
    return list(s_inter)


def generate_cubic_interp_points(x_known, y_known, n_per_inter):
    """
    generate cubic interpolated points along a complete trajectory
    :param i:
    :param x_known: np.NdArray (n_nodes,)
    :param y_known: np.NdArray (n_nodes,)
    :param n_per_inter: number of interpolated points to generate between each pair of connected nodes
    :return: x_unknown_li [np.NdArray, np.NdArray, ...], each ele of the list includes the x of points along a interval between two nodes, plus the nodes themselves # noqa
             y_unknown_li
             com_tra_wei_li_plt: [np.NdArray, np.NdArray, ...]:
    """

    knot_path_len_li = _cal_knot_dis_on_path(x_known, y_known)

    c_x = CubicSpline(knot_path_len_li, x_known)
    c_y = CubicSpline(knot_path_len_li, y_known)

    x_unknown_li = []
    y_unknown_li = []
    for j in range(len(x_known) - 1):  # 对于这条轨迹每一个截断
        interp_path_len_li = _cal_segment_dis(j, knot_path_len_li, n_per_inter)

        x_unknown = c_x(interp_path_len_li)
        y_unknown = c_y(interp_path_len_li)

        x_unknown_li.append(x_unknown)
        y_unknown_li.append(y_unknown)

    return x_unknown_li, y_unknown_li
