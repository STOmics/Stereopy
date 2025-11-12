#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:28
# @Author  : zhangchao
# @File    : describe_data.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import numpy as np
import pandas as pd


def description_data(merge_data, condition=None):
    """Description Data Basic Information

    Parameters
    --------------------------------
    merge_data: `Anndata`
        data matrix, concat data matrix with rows for cells and columns for genes.
    condition: `str`
        The designed experimental conditions. Default, each data is separate experimental design.

    Returns
    --------------------------------

    """
    if condition is None:
        merge_data.obs["condition"] = merge_data.obs["batch"]
    else:
        merge_data.obs["condition"] = condition
        merge_data.obs["condition"] = merge_data.obs["condition"].astype("category")
    describe_df = pd.DataFrame(index=merge_data.obs["condition"].cat.categories.tolist())

    for i, c in enumerate(merge_data.obs["batch"].cat.categories.tolist()):
        if i == 0:
            describe_df[f"batch{c}"] = merge_data.obs[
                merge_data.obs["batch"] == c]["condition"].value_counts(sort=False)
        else:
            t_df = pd.DataFrame(index=merge_data.obs["condition"].cat.categories.tolist())
            t_df[f"batch{c}"] = merge_data.obs[
                merge_data.obs["batch"] == c]["condition"].value_counts(sort=False)
            describe_df = pd.merge(describe_df, t_df, how="outer", left_index=True, right_index=True)
    describe_df.index = [f"condition{i}" for i in range(1, merge_data.obs["condition"].cat.categories.size + 1)]

    ideal_distribution = merge_data.obs["batch"].value_counts(sort=False, normalize=True).values
    expected_df = describe_df * ideal_distribution

    chi = ((describe_df.values - expected_df.values) ** 2 / (expected_df.values + 1e-20)).sum()

    # standardized pearson correlation coefficient
    pearson_coefficient = np.sqrt(chi * min(describe_df.shape) / (
            (chi + merge_data.shape[0]) * (min(describe_df.shape) - 1)))

    # cramer's V coefficient
    cramer_v_coefficient = np.sqrt(chi / (merge_data.shape[0] * (min(describe_df.shape) - 1)))

    confound_df = pd.DataFrame(data={"Pearson Correlation Coefficient": pearson_coefficient,
                                     "Cramer's V Coefficient": np.around(cramer_v_coefficient, decimals=4)},
                               index=["Confounding Coefficients"])
    confound_df["describe_note"] = "0=No Confounding, 1=Complete Confounding. The larger the value, the higher the " \
                                   "correlation between data batch and experimental conditions."
    describe_df.loc["Total"] = merge_data.obs["batch"].value_counts(sort=False).values
    describe_df["describe_note"] = "'condition' is the different control variables designed in the experiment. " \
                                   "By default, a batch of data studies the same control variable."
    return describe_df, confound_df
