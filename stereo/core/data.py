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


class Data(object):
    def __init__(self, file_path, file_format, partitions=1):
        self.file = Path(file_path)
        self.partitions = int(partitions)
        self.file_format = file_format

    def check(self):
        """
        checking whether the params is in the range.

        :return:
        """
        if not self.file.exists():
            logger.error(f"{str(self.file)} is not exist, please check!")
            raise FileExistsError
        if self.file_format not in ['stereo_exp', '10x']:
            logger.warning(f"the file format `{self.file_format}` is not in the range, please check!")

    @property
    def file(self):
        """
        get the file property

        :return:
        """
        return self.file

    @file.setter
    def file(self, path):
        """
        set the file property

        :param path: the file path
        :return:
        """
        if isinstance(path, str):
            self.file = Path(path)
        elif isinstance(path, Path):
            self.file = path
        else:
            raise TypeError

    @property
    def file_format(self):
        """
        get the file_format property

        :return:
        """
        return self.file_format

    @file_format.setter
    def file_format(self, f_format):
        """
        set the file_format property

        :param f_format: the file format
        :return:
        """
        self.file_format = f_format

    def read(self):
        raise NotImplementedError

    def write(self):
        raise NotImplementedError
