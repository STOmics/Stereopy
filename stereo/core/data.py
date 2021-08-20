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
import os


class Data(object):
    def __init__(self,
                 file_path: Optional[str] = None,
                 file_format: Optional[str] = None,
                 output: Optional[str] = None,
                 partitions: int = 1):
        self._file = Path(file_path) if file_path is not None else None
        self._partitions = int(partitions)
        self._file_format = file_format
        self.format_range = ['gem', 'gef', 'mtx', 'h5ad', 'scanpy_h5ad']
        self._output = output

    def check(self):
        """
        checking whether the params is in the range.

        :return:
        """
        self.file_check(file=self.file)
        self.format_check(f_format=self.file_format)
        if self.file is not None and self.file_format is None:
            logger.error('the file format must be not None , if the file path is set.')
            raise Exception

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, path):
        self.output_check(path)
        self._output = path

    @staticmethod
    def output_check(path):
        """
        check if the output dir is exists. It will be created if not exists.

        :param path:
        :return:
        """
        if path is None:
            logger.warning(f'the output path is set as None.')
            return
        out_dir = os.path.dirname(path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if os.path.exists(path):
            logger.warning(f'the output file is exists, we will replace it with new file.')

    @staticmethod
    def file_check(file):
        """
        Check if the file exists.

        :param file: the Path of file.
        :return:
        """
        if file is not None and not file.exists():
            logger.error(f"{str(file)} is not exist, please check!")
            raise FileExistsError

    def format_check(self, f_format):
        """
        Check whether the file format is in the range.

        :param f_format: the format of file.
        :return:
        """
        if f_format is not None and f_format not in self.format_range:
            logger.warning(f"the file format `{f_format}` is not in the range, please check!")

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
        if path is None:
            file = path
        elif isinstance(path, str):
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
