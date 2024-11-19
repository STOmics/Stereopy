#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:config.py
@time:2021/03/05
"""
import os
from pathlib import Path
from typing import Optional
from typing import Union
from collections import OrderedDict
from copy import deepcopy

import matplotlib.colors as mpl_colors
from colorcet import palette, aliases, cetnames_flipped
from matplotlib import rcParams
from matplotlib import rcParamsDefault
import numpy as np
import seaborn as sns


class StereoConfig(object):
    """
    config of stereo.

    log_format:
    https://docs.python.org/3.8/library/logging.html?highlight=logging#logrecord-attributes
    """

    def __init__(
            self,
            file_format: str = "h5ad",
            auto_show: bool = True,
            n_jobs=1,
            log_file: Union[str, Path, None] = None,
            log_level: str = "info",
            log_format: str = "[%(asctime)s][%(name)s][%(process)d][%(threadName)s][%(thread)d][%(module)s]"
                              "[%(lineno)d][%(levelname)s]: %(message)s",
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
        self._palette_custom = None


    @property
    def palette_custom(self):
        return self._palette_custom
    
    @palette_custom.setter
    def palette_custom(self, palette_custom):
        if not isinstance(palette_custom, list):
            raise ValueError('palette_custom should be a list of colors')
        self._palette_custom = palette_custom

    @property
    def colormaps(self):
        color_keys = sorted([k for k in palette.keys() if 'glasbey' in k and '_bw_' not in k])
        colormaps = OrderedDict([(k, palette[k]) for k in color_keys])
        # colormaps = {k: v for k, v in palette.items() if 'glasbey' in k and '_bw_' not in k}
        colormaps['stereo_30'] = [
            "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628",
            "#FFFF33", "#F781BF", "#999999", "#E5D8BD", "#B3CDE3", "#CCEBC5",
            "#FED9A6", "#FBB4AE", "#8DD3C7", "#BEBADA", "#80B1D3", "#B3DE69",
            "#FCCDE5", "#BC80BD", "#FFED6F", "#8DA0CB", "#E79AD3", "#E5C494",
            "#CCCCCC", "#FB9A99", "#10E03C", "#DAB2D6", "#6A3D9A", "#D15928"
        ]
        if self.palette_custom is not None:
            colormaps['custom'] = self.palette_custom
        return colormaps

    @property
    def linear_colormaps(self):
        color_keys = sorted([k for k in palette.keys() if not ('glasbey' in k or k in aliases or k in cetnames_flipped)])
        colormaps = OrderedDict([(k, palette[k]) for k in color_keys])

        stmap_colors = ['#0c3383', '#0a88ba', '#f2d338', '#f28f38', '#d91e1e']
        nodes = [0.0, 0.25, 0.50, 0.75, 1.0]
        mycmap = mpl_colors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, stmap_colors)))
        color_list = [mpl_colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
        colormaps['stereo'] = color_list
        return colormaps

    def linear_colors(self, colors, reverse=False):
        if isinstance(colors, str):
            linear_colormaps = deepcopy(palette)
            linear_colormaps.update(self.linear_colormaps)
            if colors not in linear_colormaps:
                # raise ValueError(f'{colors} not in colormaps, color value range in {self.linear_colormaps.keys()}')
                mycmap = sns.color_palette(colors, as_cmap=True)
                colors = [mpl_colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
                return colors[::-1] if reverse else colors
            else:
                return linear_colormaps[colors][::-1] if reverse else linear_colormaps[colors]
        elif isinstance(colors, (list, tuple, np.ndarray)):
            mycmap = mpl_colors.LinearSegmentedColormap.from_list("mycmap", colors)
            colors = [mpl_colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
            return colors[::-1] if reverse else colors
        else:
            raise ValueError('colors should be str or list type')

    def get_colors(self, colors, n=None, order=None):
        if isinstance(colors, str):
            colormaps = deepcopy(palette)
            colormaps.update(self.colormaps)
            if colors not in colormaps:
                # raise ValueError(f'{colors} not in colormaps, color value range in {self.colormaps.keys()}')
                mycmap = sns.color_palette(colors, as_cmap=True)
                colormaps[colors] = colormaps_selected = [mpl_colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
            else:
                colormaps_selected = colormaps[colors]
        elif isinstance(colors, (dict, OrderedDict)):
            if order is not None:
                colormaps_selected = [colors[k] for k in order if k in colors]
            else:
                colormaps_selected = list(colors.values())
        elif isinstance(colors, (list, tuple, np.ndarray)):
            colormaps_selected = list(colors)
        else:
            raise ValueError('colors should be str, dict, list, tuple or np.ndarray type')
        
        if n is not None:
            if n > len(colormaps_selected):
                mycmap = mpl_colors.LinearSegmentedColormap.from_list("mycmap", colormaps_selected, N=n)
                colormaps_selected = [mpl_colors.rgb2hex(mycmap(i)) for i in range(n)]
            else:
                if colors == 'stereo_30':
                    colormaps_selected = colormaps_selected[:n]
                else:
                    index_selected = np.linspace(0, len(colormaps_selected), n, endpoint=False, dtype=int)
                    colormaps_selected = [colormaps_selected[i] for i in index_selected]
        
        return colormaps_selected

    @property
    def log_file(self) -> Union[str, Path, None]:
        """
        get the file path of log.
        """
        return self._log_file

    @log_file.setter
    def log_file(self, value):
        """
        set file path of log.

        :param value: value of log file path
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
        """
        return self._log_format

    @log_format.setter
    def log_format(self, value):
        """
        set file path of log.

        :param value: value of log format
        """
        self._log_format = value

    @property
    def log_level(self) -> str:
        """
        get log level
        """
        return self._log_level

    @log_level.setter
    def log_level(self, value):
        """
        set log level

        :param value: the value of log level
        """
        if value.lower() not in ['info', 'warning', 'debug', 'error', 'critical']:
            print('the log level is out of range, please check and it is not modified.')
        else:
            self._log_level = value

    @property
    def auto_show(self):
        """
        Auto show figures if `auto_show == True` (default `True`).
        """
        return self._auto_show

    @auto_show.setter
    def auto_show(self, value):
        """
        set value of auto_show

        :param value: value of auto_show
        """
        self._auto_show = value

    @property
    def file_format(self) -> str:
        """
        file format of saving anndata object
        """
        return self._file_format

    @file_format.setter
    def file_format(self, value):
        """
        set the value of file format

        :param value: the value of file format
        """
        self._file_format = value

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @staticmethod
    def set_plot_param(
            fontsize: int = 14,
            figsize: Optional[int] = None,
            color_map: Optional[str] = None,
            facecolor: Optional[str] = None,
            transparent: bool = False
    ):
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
        """
        rcParams.update(rcParamsDefault)


stereo_conf = StereoConfig()
