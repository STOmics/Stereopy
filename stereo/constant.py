#!/usr/bin/env python3
# coding: utf-8
"""
@author: wenzhenbin  wenzhenbin@genomics.cn
@last modified by: wenzhenbin
@file:constant.py
@time:2023/07/03
"""
from enum import Enum

TMP = "tmp"
PAGA = "paga"
BATCH = "batch"
GROUP = "group"
INDEX = "index"
LOG_FC = "logFC"
LESS_P = "less_p"
SCORES = "scores"
SANKEY = "sankey"
SIMPLE = "simple"
CATEGORY = "category"
GREATER_P = "greater_p"
FEATURE_P = "feature_p"
ANNOTATION = "annotation"
LESS_PVALUE = "less_pvalue"
END_CELLPOSE = '.cellpose'
_LOG_PVALUE = "_log_pvalue"
TOTAL_COUNTS = "total_counts"
CELLTYPE_STD = "celltype_std"
PCT_COUNTS_MT = "pct_counts_mt"
CELLTYPE_MEAN = "celltype_mean"
FUZZY_C_WEIGHT = "fuzzy_C_weight"
FUZZY_C_RESULT = "fuzzy_C_result"
GREATER_PVALUE = "greater_pvalue"
N_GENES_BY_COUNTS = "n_genes_by_counts"
CELLTYPE_MEAN_SCALE = "celltype_mean_scale"
CONNECTIVITIES_TREE = "connectivities_tree"
PLOT_SCATTER_SIZE_FACTOR = 120000
PLOT_BASE_IMAGE_EXPANSION = 500

MODEL_URL = 'https://www.cellpose.org/models'
CELLPOSE_GUI_PNG_URL = 'https://www.cellpose.org/static/images/cellpose_gui.png'
STYLE_CHOICE_NPY_URL = 'https://www.cellpose.org/static/models/style_choice.npy'
CELLPOSE_TRANSPARENT_PNG_URL = 'https://www.cellpose.org/static/images/cellpose_transparent.png'


class BatchColType(Enum):
    sample_name = "sample_name"
    timepoint = "timepoint"
    time = 'time'


class AlternativeType(Enum):
    less = "less"
    greater = "greater"


class UseColType(Enum):
    celltype = "celltype"
    timepoint = "timepoint"


class PaletteType(Enum):
    tab20 = "tab20"


class DptColType(Enum):
    dpt_pseudotime = "dpt_pseudotime"


class DirectionType(Enum):
    left = "left"
    right = "right"
    center = 'center'
    bottom = 'bottom'


class PValCombinationType(Enum):
    mean = "mean"
    fdr = "FDR"
    fisher = "fisher"


class ColorType(Enum):
    red = "red"
    grey = "grey"
    black = "black"
    green = "green"


class RunMethodType(Enum):
    tvg_marker = "tvg_marker"


class VersionType(Enum):
    v1 = 'v1'
    v1_pro = 'v1_pro'
    v3 = 'v3'

    @staticmethod
    def get_version_list():
        return [key.value for key in VersionType]
