#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/9/23 5:14 PM
# @Author  : zhangchao
# @File    : time_decorator.py
# @Email   : zhangchao5@genomics.cn
import time


def get_running_time(func):
    def func_time(*args, **kwargs):
        t0 = time.time()
        print(f"{get_format_time()} Method: '{func.__name__}' Running...")
        res = func(*args, **kwargs)
        t1 = time.time()
        print(f"  Running time: {((t1 - t0) // 60)} min {((t1 - t0) % 60):.4f} s")
        return res

    return func_time


def get_format_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
