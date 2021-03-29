#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file:__init__.py.py
@time:2021/03/05
"""
from . import plots as plt
import sys
from .config import StereoConfig
from .log_manager import logger
from . import io
from . import preprocess
from . import tools
from . import utils

# do the end.
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['plt']})

