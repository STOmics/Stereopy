#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/21 11:22
# @Author  : payne
# @File    : read_write_utils.py
# @Description : utils for reading file
import os
from functools import wraps


class ReadWriteUtils(object):
    @staticmethod
    def check_file_exists(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            if kwargs:
                # case (a=1, b=2, c=3, d=4)
                if kwargs.get('file_path'):
                    if not os.path.exists(kwargs.get('file_path')):
                        raise FileNotFoundError("Please ensure there is a file")
                else:
                    # case ('/test/test/test.file', b=2)
                    if args:
                        path = args[0]
                        if not os.path.exists(path):
                            raise FileNotFoundError("Please ensure there is a file")
                    else:
                        raise FileNotFoundError("Please ensure there is a file")
            else:
                if args:
                    # case (1, 2, 3, 4)
                    path = args[0]
                    if not os.path.exists(path):
                        raise FileNotFoundError("Please ensure there is a file")
                else:
                    raise FileNotFoundError("Please ensure there is a file")
            return func(*args, **kwargs)

        return wrapped
