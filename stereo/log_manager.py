#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:log_manager.py
@time:2021/03/05
"""

import logging
from .config import stereo_conf


class LogManager(object):
    def __init__(self, log_path=None, level=None):
        self.level_map = {'debug': logging.DEBUG,
                          'info': logging.INFO,
                          'warning': logging.WARNING,
                          'error': logging.ERROR,
                          'critical': logging.CRITICAL}
        self.format = stereo_conf.log_format
        self.formatter = logging.Formatter(self.format, "%Y-%m-%d %H:%M:%S")
        self.log_path = log_path
        self.level = level.lower() if level else stereo_conf.log_level.lower()
        if self.log_path:
            self.file_handler = logging.FileHandler(self.log_path)
            self.file_handler.setLevel(self.level_map[self.level])
            self.file_handler.setFormatter(self.formatter)
        else:
            self.stream_handler = logging.StreamHandler()
            self.stream_handler.setLevel(self.level_map[self.level])
            self.stream_handler.setFormatter(self.formatter)

    def get_logger(self, name="Stereo"):
        """
        get logger object
        :param name: logger name
        :return: logger object
        """
        alogger = logging.getLogger(name)
        alogger.propagate = 0
        alogger.setLevel(self.level_map[self.level])
        self._add_handler(alogger)
        return alogger

    def _add_handler(self, alogger):
        """
        add handler of logger
        :param alogger: logger object
        :return:
        """
        if self.log_path:
            alogger.addHandler(self.file_handler)
        else:
            alogger.addHandler(self.stream_handler)


logger = LogManager().get_logger(name='Stereo')
