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
        from colorcet import palette
        colormaps = {n: palette[n] for n in ['glasbey', 'glasbey_bw', 'glasbey_cool', 'glasbey_warm', 'glasbey_dark',
                                             'glasbey_light', 'glasbey_category10', 'glasbey_hv']}
        colormaps['st'] = ['violet', 'turquoise', 'tomato', 'teal', 'tan', 'silver', 'sienna', 'red', 'purple', 'plum',
                           'pink',
                           'orchid', 'orangered', 'orange', 'olive', 'navy', 'maroon', 'magenta', 'lime',
                           'lightgreen', 'lightblue', 'lavender', 'khaki', 'indigo', 'grey', 'green', 'gold', 'fuchsia',
                           'darkgreen', 'darkblue', 'cyan', 'crimson', 'coral', 'chocolate', 'chartreuse', 'brown',
                           'blue', 'black',
                           'beige', 'azure', 'aquamarine', 'aqua',
                           ]
        colormaps['stereo_30'] = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628", "#FFFF33",
                                  "#F781BF", "#999999", "#E5D8BD", "#B3CDE3", "#CCEBC5", "#FED9A6", "#FBB4AE",
                                  "#8DD3C7", "#BEBADA", "#80B1D3", "#B3DE69", "#FCCDE5", "#BC80BD", "#FFED6F",
                                  "#8DA0CB", "#E78AC3", "#E5C494", "#CCCCCC", "#FB9A99", "#E31A1C", "#CAB2D6",
                                  "#6A3D9A", "#B15928"]
        colormaps['stereo_150'] = ["#E41A1C", "#C22D3A", "#A04058", "#7E5477", "#5D6795", "#3B7BB3", "#3A86A5",
                                   "#3E8F90", "#43997A", "#47A265", "#4BAC4F", "#57A156", "#668E67", "#747B79",
                                   "#83688A", "#91559B", "#A35390", "#B75D70", "#CB6650", "#DF7031", "#F37911",
                                   "#F77B03", "#E5730B", "#D46B13", "#C3631A", "#B15B22", "#AB6028", "#BC812A",
                                   "#CEA12C", "#DFC22F", "#F0E331", "#FEFA37", "#FDE252", "#FBC96E", "#FAB189",
                                   "#F898A4", "#F681BE", "#E485B7", "#D18AAF", "#BF8FA8", "#AD93A1", "#9A9899",
                                   "#A6A39F", "#B5B0A6", "#C3BCAD", "#D2C8B4", "#E1D5BB", "#DDD6C2", "#D3D4CA",
                                   "#CAD2D1", "#C0CFD8", "#B6CDE0", "#B6D0DF", "#BAD6D9", "#BFDCD3", "#C4E2CD",
                                   "#C9E7C8", "#D0E9C2", "#DAE5BC", "#E4E2B6", "#EDDEAF", "#F7DBA9", "#FDD6A6",
                                   "#FDCFA8", "#FCC8A9", "#FCC0AB", "#FBB9AC", "#F6B5AF", "#E1BBB3", "#CBC1B8",
                                   "#B6C7BD", "#A0CDC2", "#8DD2C7", "#97CDCA", "#A0C8CE", "#AAC4D2", "#B3BFD6",
                                   "#BDBAD9", "#B2B8D8", "#A6B6D7", "#9AB4D6", "#8EB3D4", "#82B1D3", "#87B7C2",
                                   "#91C0AE", "#9BC999", "#A5D284", "#AFDA70", "#BCDB78", "#CAD890", "#D8D5A9",
                                   "#E6D1C1", "#F5CED9", "#F5C5E0", "#E9B6D9", "#DCA7D1", "#D098C9", "#C389C1",
                                   "#C088B7", "#CD9DA8", "#DBB298", "#E8C789", "#F5DC7A", "#F9E973", "#E3DA85",
                                   "#CDCB97", "#B7BCA9", "#A0ADBA", "#8E9FCA", "#A09BC9", "#B196C7", "#C392C6",
                                   "#D48EC4", "#E68AC3", "#E694BA", "#E6A0B1", "#E5ABA7", "#E5B69E", "#E5C295",
                                   "#E0C59D", "#DCC6A7", "#D7C8B2", "#D2C9BD", "#CDCBC8", "#D2C5C5", "#DBBBBB",
                                   "#E4B1B1", "#EDA8A7", "#F69E9D", "#F88C8B", "#F37373", "#EF5A5A", "#EA4142",
                                   "#E5282A", "#E0262A", "#DC434F", "#D76173", "#D27E97", "#CD9CBB", "#C4ABD2",
                                   "#B294C7", "#9F7EBB", "#8C67AF", "#7A50A4", "#6B3D96", "#794380", "#87486A",
                                   "#954E54", "#A3533E", "#B15928"]
        return colormaps

    def get_colors(self, colors):
        if isinstance(colors, str):
            if colors not in self.colormaps:
                raise ValueError(f'{colors} not in colormaps, color value range in {self.colormaps.keys()}')
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
