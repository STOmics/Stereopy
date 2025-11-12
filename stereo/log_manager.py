#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:log_manager.py
@time:2021/03/05
"""

import logging

from .stereo_config import stereo_conf


class LogManager(object):
    __instance = None

    def __init__(self, log_path=None, level=None, only_log_to_file=False):
        self.level_map = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        self.format = stereo_conf.log_format
        self.formatter = logging.Formatter(self.format, "%Y-%m-%d %H:%M:%S")
        self.log_path = log_path
        self.only_log_to_file = only_log_to_file
        self.level = level.lower() if level else stereo_conf.log_level.lower()
        self.file_handler = None
        self.stream_handler = None

    def get_logger(self, name="Stereo"):
        """
        get logger object

        :param name: logger name

        :return: logger object
        """
        alogger = logging.getLogger(name)
        alogger.propagate = 0
        alogger.setLevel(self.level_map[self.level])
        self._set_handler()
        self._remove_handler(alogger)
        self._add_handler(alogger)
        return alogger

    def _set_handler(self):
        self.file_handler, self.stream_handler = None, None
        if self.log_path:
            self.file_handler = logging.FileHandler(self.log_path)
            self.file_handler.setLevel(self.level_map[self.level])
            self.file_handler.setFormatter(self.formatter)
        if self.log_path is None or not self.only_log_to_file:
            self.stream_handler = logging.StreamHandler()
            self.stream_handler.setLevel(self.level_map[self.level])
            self.stream_handler.setFormatter(self.formatter)

    def _remove_handler(self, alogger: logging.Logger):
        alogger.handlers.clear()

    def _add_handler(self, alogger: logging.Logger):
        """
        add handler of logger

        :param alogger: logger object

        :return: None
        """
        if self.file_handler is not None:
            alogger.addHandler(self.file_handler)
        if self.stream_handler is not None:
            alogger.addHandler(self.stream_handler)

    @staticmethod
    def get_instance(log_path=None, level=None, only_log_to_file=False):
        if LogManager.__instance is None:
            LogManager.__instance = LogManager(log_path=log_path, level=level, only_log_to_file=only_log_to_file)
        return LogManager.__instance

    @staticmethod
    def logger(name='Stereo'):
        return LogManager.get_instance().get_logger(name=name)

    @staticmethod
    def set_level(level='info'):
        global logger
        level = level.lower()
        if level not in ['debug', 'info', 'warning', 'error', 'critical']:
            raise ValueError("Invalid log level, available values are ['info', 'warning', 'debug', 'error',"
                             " 'critical'].")
        log_manager = LogManager.get_instance()
        log_manager.level = level
        logger = log_manager.get_logger(name='Stereo')

    @staticmethod
    def log_to_file(log_path=None, only_log_to_file=False):
        global logger
        log_manager = LogManager.get_instance()
        log_manager.log_path = log_path
        log_manager.only_log_to_file = only_log_to_file
        logger = log_manager.get_logger(name='Stereo')

    @staticmethod
    def stop_logging():
        global logger
        log_manager = LogManager.get_instance()
        log_manager._remove_handler(logger)

    @staticmethod
    def start_logging():
        global logger
        log_manager = LogManager.get_instance()
        log_manager._add_handler(logger)


logger = LogManager.logger()
