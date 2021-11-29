#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:config.py
@time:2021/03/05
"""
from typing import Union, Optional
from pathlib import Path
import os
from matplotlib import rcParams, rcParamsDefault
import matplotlib.colors as mpl_colors
from colorcet import palette


class StereoConfig(object):
    """
    config of stereo.
    """

    def __init__(
            self,
            file_format: str = "h5ad",
            auto_show: bool = True,
            n_jobs=1,
            log_file: Union[str, Path, None] = None,
            log_level: str = "info",
            log_format: str = "%(asctime)s %(name)s %(levelname)s: %(message)s",
            output: str = "./output",
            data_dir: str = None
    ):
        self._file_format = file_format
        self._auto_show = auto_show
        self._n_jobs = n_jobs
        self._log_file = log_file
        self._log_level = log_level
        self._log_format = log_format
        self.out_dir = output
        self.data_dir = data_dir if data_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    @property
    def colormaps(self):
        colormaps = {n: palette[n] for n in ['glasbey', 'glasbey_bw', 'glasbey_cool', 'glasbey_warm', 'glasbey_dark',
                                             'glasbey_light', 'glasbey_category10', 'glasbey_hv']}
        colormaps['stereo_30'] = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628", "#FFFF33",
                                  "#F781BF", "#999999", "#E5D8BD", "#B3CDE3", "#CCEBC5", "#FED9A6", "#FBB4AE",
                                  "#8DD3C7", "#BEBADA", "#80B1D3", "#B3DE69", "#FCCDE5", "#BC80BD", "#FFED6F",
                                  "#8DA0CB", "#E78AC3", "#E5C494", "#CCCCCC", "#FB9A99", "#E31A1C", "#CAB2D6",
                                  "#6A3D9A", "#B15928"]
        return colormaps

    @property
    def linear_colormaps(self):
        colormaps = {n: palette[n] for n in ['rainbow', 'fire', 'bgy', 'bgyw', 'bmy', 'gray', 'kbc', 'CET_D4']}
        stmap_colors = ['#0c3383', '#0a88ba', '#f2d338', '#f28f38', '#d91e1e']
        nodes = [0.0, 0.25, 0.50, 0.75, 1.0]
        mycmap = mpl_colors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, stmap_colors)))
        color_list = [mpl_colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
        colormaps['stereo'] = color_list
        return colormaps

    def linear_colors(self, colors):
        if isinstance(colors, str):
            if colors not in self.linear_colormaps:
                raise ValueError(f'{colors} not in colormaps, color value range in {self.linear_colormaps.keys()}')
            else:
                return self.linear_colormaps[colors]
        elif isinstance(colors, list):
            return colors
        else:
            raise ValueError('colors should be str or list type')

    def get_colors(self, colors, n=None):
        if isinstance(colors, str):
            if colors not in self.colormaps:
                raise ValueError(f'{colors} not in colormaps, color value range in {self.colormaps.keys()}')
            if n is not None:
                if n > len(self.colormaps[colors]):
                    mycmap = mpl_colors.LinearSegmentedColormap.from_list("mycmap", self.colormaps[colors], N=n)
                    color_list = [mpl_colors.rgb2hex(mycmap(i)) for i in range(n)]
                else:
                    color_list = self.colormaps[colors][0: n]
                return color_list
            else:
                return self.colormaps[colors]
        else:
            return colors

    @property
    def log_file(self) -> Union[str, Path, None]:
        """
        get the file path of log.
        :return:
        """
        return self._log_file

    @log_file.setter
    def log_file(self, value):
        """
        set file path of log.
        :param value: value of log file path
        :return:
        """
        if value:
            dir_path = os.path.dirname(value)
            if not os.path.exists(dir_path):
                raise FileExistsError("folder does not exist, please check!")
        self._log_file = value

    @property
    def log_format(self) -> str:
        """
        get the format of log.
        :return:
        """
        return self._log_format

    @log_format.setter
    def log_format(self, value):
        """
        set file path of log.
        :param value: value of log format
        :return:
        """
        self._log_format = value

    @property
    def log_level(self) -> str:
        """
        get log level
        :return:
        """
        return self._log_level

    @log_level.setter
    def log_level(self, value):
        """
        set log level
        :param value: the value of log level
        :return:
        """
        if value.lower() not in ['info', 'warning', 'debug', 'error', 'critical']:
            print('the log level is out of range, please check and it is not modified.')
        else:
            self._log_level = value

    @property
    def auto_show(self):
        """
        Auto show figures if `auto_show == True` (default `True`).
        :return:
        """
        return self._auto_show

    @auto_show.setter
    def auto_show(self, value):
        """
        set value of auto_show
        :param value: value of auto_show
        :return:
        """
        self._auto_show = value

    @property
    def file_format(self) -> str:
        """
        file format of saving anndata object
        :return:
        """
        return self._file_format

    @file_format.setter
    def file_format(self, value):
        """
        set the value of file format
        :param value: the value of file format
        :return:
        """
        self._file_format = value

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @staticmethod
    def set_plot_param(fontsize: int = 14, figsize: Optional[int] = None, color_map: Optional[str] = None,
                       facecolor: Optional[str] = None, transparent: bool = False, ):
        if fontsize is not None:
            rcParams['font.size'] = fontsize
        if color_map is not None:
            rcParams['image.cmap'] = color_map
        if figsize is not None:
            rcParams['figure.figsize'] = figsize
        if facecolor is not None:
            rcParams['figure.facecolor'] = facecolor
            rcParams['axes.facecolor'] = facecolor
        if transparent is not None:
            rcParams["savefig.transparent"] = transparent

    @staticmethod
    def set_rcparams_defaults():
        """
        reset `matplotlib.rcParams` to defaults.
        :return:
        """
        rcParams.update(rcParamsDefault)


stereo_conf = StereoConfig()
