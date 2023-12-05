#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:58
# @Author  : zhangchao
# @File    : print_time.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import time


def print_time(function):
    def func_time(*args, **kwargs):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} BatchQC Starting")
        t0 = time.time()
        res = function(*args, **kwargs)
        t1 = time.time()
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} BatchQC Done!")
        t = t1 - t0
        print(f"Total Running Time: {t // 60:}min {t % 60:.4f}s")
        return res

    return func_time
