#!/usr/bin/env python3
# coding: utf-8
"""
@file: data.py
@description: the base class of data.
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2021/06/11  create file.
"""
from pathlib import Path
from ..log_manager import logger
from typing import Optional


class Data(object):
    def __init__(self, file_path: Optional[str] = None, file_format: Optional[str] = None, partitions: int = 1):
        self._file = Path(file_path)
        self._partitions = int(partitions)
        self._file_format = file_format
        self.format_range = ['txt', 'csv', 'mtx', 'h5ad']
        self.logger = logger

    def check(self):
        """
        checking whether the params is in the range.

        :return:
        """
        self.file_check(file=self.file)
        self.format_check(f_format=self.file_format)

    def file_check(self, file):
        """
        Check if the file exists.

        :param file: the Path of file.
        :return:
        """
        if file is not None and not file.exists():
            self.logger.error(f"{str(file)} is not exist, please check!")
            raise FileExistsError

    def format_check(self, f_format):
        """
        Check whether the file format is in the range.

        :param f_format: the format of file.
        :return:
        """
        if f_format is not None and f_format not in self.format_range:
            self.logger.warning(f"the file format `{f_format}` is not in the range, please check!")

    @property
    def file(self):
        """
        get the file property

        :return:
        """
        return self._file

    @file.setter
    def file(self, path):
        """
        set the file property

        :param path: the file path
        :return:
        """
        if isinstance(path, str):
            file = Path(path)
        elif isinstance(path, Path):
            file = path
        else:
            raise TypeError
        self.file_check(file=file)
        self._file = file

    @property
    def file_format(self):
        """
        get the file_format property

        :return:
        """
        return self._file_format

    @file_format.setter
    def file_format(self, f_format):
        """
        set the file_format property

        :param f_format: the file format
        :return:
        """
        self.format_check(f_format)
        self._file_format = f_format

    @property
    def partitions(self):
        """
        get the partitions property

        :return:
        """
        return self._partitions

    @partitions.setter
    def partitions(self, partition):
        """
        set the partitions property

        :param partition: the partitions number, which define the cores of multi processes.
        :return:
        """
        self._partitions = partition

    def read(self, *args, **kwargs):
        raise NotImplementedError

    def write(self, *args, **kwargs):
        raise NotImplementedError
