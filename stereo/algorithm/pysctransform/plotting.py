# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from scipy.stats.stats import pearsonr


def is_outlier(x, snr_threshold=25):
    """
    Mark points as outliers
    """
    if isinstance(x, pd.Series):
        x = x.values
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    median = np.median(x, axis=0)
    diff = np.sum((x - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > snr_threshold


def plot_fit(pysct_results, xaxis="gmean", fig=None):
    """
    Parameters
    ----------
    pysct_results: dict
                   obsect returned by pysctransform.vst
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 3))
    genes_log10_mean = pysct_results["genes_log10_{}".format(xaxis)]
    genes_log10_mean_step1 = pysct_results["genes_log10_{}_step1".format(xaxis)]
    model_params = pysct_results["model_parameters"]
    model_params_fit = pysct_results["model_parameters_fit"]

    total_params = model_params_fit.shape[1]

    for index, column in enumerate(model_params_fit.columns):
        ax = fig.add_subplot(1, total_params, index + 1)
        model_param_col = model_params[column]
        # model_param_outliers = is_outlier(model_param_col)
        if column != "theta":
            ax.scatter(
                genes_log10_mean_step1,  # [~model_param_outliers],
                model_param_col,  # [~model_param_outliers],
                s=1,
                label="single gene estimate",
                color="#2b8cbe",
            )
            ax.scatter(
                genes_log10_mean,
                model_params_fit[column],
                s=2,
                label="regularized",
                color="#de2d26",
            )
            ax.set_ylabel(column)
        else:
            ax.scatter(
                genes_log10_mean_step1,  # [~model_param_outliers],
                np.log10(model_param_col),  # [~model_param_outliers],
                s=1,
                label="single gene estimate",
                color="#2b8cbe",
            )
            ax.scatter(
                genes_log10_mean,
                np.log10(model_params_fit[column]),
                s=2,
                label="regularized",
                color="#de2d26",
            )
            ax.set_ylabel("log10(" + column + ")")
        if column == "od_factor":
            ax.set_ylabel("log10(od_factor)")

        ax.set_xlabel("log10(gene_{})".format(xaxis))
        ax.set_title(column)
        ax.legend(frameon=False)
    _ = fig.tight_layout()
    return fig


def plot_residual_var(pysct_results, topngenes=30, label_genes=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = None

    gene_attr = pysct_results["gene_attr"]
    gene_attr_sorted = gene_attr.sort_values(
        by=["residual_variance"], ascending=False
    ).reset_index()
    # TODO: error check
    topn = gene_attr_sorted.iloc[:topngenes]
    gene_attr = gene_attr_sorted.iloc[topngenes:]
    ax.set_xscale("log")

    ax.scatter(
        gene_attr["gene_gmean"], gene_attr["residual_variance"], s=1.5, color="black"
    )
    ax.scatter(topn["gene_gmean"], topn["residual_variance"], s=1.5, color="deeppink")
    ax.axhline(1, linestyle="dashed", color="red")
    ax.set_xlabel("Gene gmean")
    ax.set_ylabel("Residual variance")
    if label_genes:
        texts = [
            plt.text(row["gene_gmean"], row["residual_variance"], row["index"])
            for index, row in topn.iterrows()
        ]
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))
    # fig.tight_layout()
    return fig


def compare_with_sct(
    vst_out, sct_modelparsfit_file, sct_geneattr_file, sct_modelpars_file=None
):
    sct_modelparsfit = pd.read_csv(sct_modelparsfit_file, index_col=0)
    sct_geneattr = pd.read_csv(sct_geneattr_file, index_col=0)
    nplots = 3
    if sct_modelpars_file:
        nplots = 4
        sct_modelpars = pd.read_csv(sct_modelpars_file, index_col=0)
        sct_modelpars.columns = ["sct_" + x for x in sct_modelpars.columns]
        model_pars_merged = vst_out["model_parameters"].join(sct_modelpars, how="inner")

    sct_modelparsfit.columns = ["sct_" + x for x in sct_modelparsfit.columns]
    sct_geneattr.columns = ["sct_" + x for x in sct_geneattr.columns]

    model_parsfit_merged = vst_out["model_parameters_fit"].join(
        sct_modelparsfit, how="inner"
    )
    gene_attr_merged = vst_out["gene_attr"].join(sct_geneattr, how="inner")

    fig = plt.figure(figsize=(4 * nplots, 4))
    ax = fig.add_subplot(1, nplots, 1)
    ax.scatter(
        model_parsfit_merged["sct_theta"],
        model_parsfit_merged["theta"],
        s=1,
        color="black",
    )
    ax.axline([0, 0], [1, 1], linestyle="dashed", color="red")
    ax.set_xlabel("SCT theta (regularized)")
    ax.set_ylabel("pySCT theta (regularized)")

    ax = fig.add_subplot(1, nplots, 2)
    ax.scatter(
        gene_attr_merged["sct_residual_mean"],
        gene_attr_merged["residual_mean"],
        s=1,
        color="black",
    )
    ax.axline([0, 0], [1, 1], linestyle="dashed", color="red")
    ax.set_xlabel("SCT residual mean")
    ax.set_ylabel("pySCT residual mean")

    ax = fig.add_subplot(1, nplots, 3)
    ax.scatter(
        gene_attr_merged["sct_residual_variance"],
        gene_attr_merged["residual_variance"],
        s=1,
        color="black",
    )
    gene_attr_merged_var = gene_attr_merged[
        ["sct_residual_variance", "residual_variance"]
    ].dropna()
    cor = pearsonr(
        gene_attr_merged_var["sct_residual_variance"],
        gene_attr_merged_var["residual_variance"],
    )[0]

    ax.axline(
        [0, 0], [1, 1], linestyle="dashed", color="red", label="r={:.2f}".format(cor)
    )
    ax.set_xlabel("SCT residual variance")
    ax.set_ylabel("pySCT residual variance")
    ax.legend(frameon=False)

    if nplots == 4:
        ax = fig.add_subplot(1, nplots, 4)
        ax.scatter(
            model_pars_merged["sct_theta"],
            model_pars_merged["theta"],
            s=1,
            color="black",
        )
        ax.axline([0, 0], [1, 1], linestyle="dashed", color="red")
        model_pars_merged = model_pars_merged.replace(np.inf, 1e5)  # dropna()
        cor = pearsonr(model_pars_merged["sct_theta"], model_pars_merged["theta"])[0]
        ax.set_xlabel("SCT theta")
        ax.set_ylabel("pySCT theta")

        ax.axline(
            [0, 0],
            [1, 1],
            linestyle="dashed",
            color="red",
            label="r={:.2f}".format(cor),
        )
        ax.legend(frameon=False)

    fig.tight_layout()
    return fig
