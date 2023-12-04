import matplotlib.pyplot as plt
import numpy as np


def plot_fit(pysct_results, xaxis="gmean", fig=None):
    if fig is None:
        fig = plt.figure(figsize=(12, 3))
    genes_log10_mean = pysct_results[1]["gene_attr"]['gmean']
    genes_log10_mean_step1 = pysct_results[1]['genes_log_gmean_step1'],
    model_params = pysct_results[1]['model_pars']
    model_params_fit = pysct_results[1]['model_pars_fit']

    total_params = model_params_fit.shape[1]

    for index, column in enumerate(model_params_fit.columns):
        ax = fig.add_subplot(1, total_params, index + 1)
        model_param_col = model_params[column]
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

    gene_attr = pysct_results[1]["gene_attr"]
    gene_attr_sorted = gene_attr.sort_values(by=["residual_variance"], ascending=False).reset_index()
    # TODO: error check
    topn = gene_attr_sorted.iloc[:topngenes]
    gene_attr = gene_attr_sorted.iloc[topngenes:]
    ax.set_xscale("log")

    ax.scatter(
        gene_attr["gmean"], gene_attr["residual_variance"], s=1.5, color="black"
    )
    ax.scatter(topn["gmean"], topn["residual_variance"], s=1.5, color="deeppink")
    ax.axhline(1, linestyle="dashed", color="red")
    ax.set_xlabel("Gene GMean")
    ax.set_ylabel("Pearson Residual Variance")
    if label_genes:
        _ = [
            plt.text(row["gmean"], row["residual_variance"], row["index"])
            for index, row in topn.iterrows()
        ]
    return fig


def plot_log_normalize_var(log_normal_results, topngenes=30, label_genes=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = None

    gene_attr = log_normal_results
    gene_attr_sorted = gene_attr.sort_values(by=["log_normalize_variance"], ascending=False).reset_index()
    # TODO: error check
    topn = gene_attr_sorted.iloc[:topngenes]
    gene_attr = gene_attr_sorted.iloc[topngenes:]
    ax.set_xscale("log")

    ax.scatter(
        gene_attr["gmean"], gene_attr["log_normalize_variance"], s=1.5, color="black"
    )
    ax.scatter(topn["gmean"], topn["log_normalize_variance"], s=1.5, color="deeppink")
    ax.axhline(1, linestyle="dashed", color="red")
    ax.set_xlabel("Gene GMean")
    ax.set_ylabel("Log Normalize Variance")
    if label_genes:
        _ = [
            plt.text(row["gmean"], row["log_normalize_variance"], row["index"])
            for index, row in topn.iterrows()
        ]
    return fig


colors = ['red', 'blue', 'green', 'grey', 'black', 'pink']
bar_width = 0.1
opacity = 0.5
error_config = {'ecolor': '0.3'}


def plot_genes_var_contribution(stereo_exp_data, gene_names=None):
    exp_matrix_df = stereo_exp_data.to_df()

    # SCTransform
    stereo_exp_data.tl.sctransform(res_key='sctransform', inplace=True, filter_hvgs=True)

    genes_var_series = stereo_exp_data.tl.result['sctransform'][0]['scale.data'].var(1)

    if gene_names:
        genes_var_series = genes_var_series.loc[gene_names]
        gene_names = genes_var_series.index.values
    else:
        genes_var_series = genes_var_series.sample(6)
        gene_names = genes_var_series.index.values

    genes_var_series_sum = genes_var_series.sum()
    genes_var_contribution = genes_var_series / genes_var_series_sum * 100

    fig, axes = plt.subplots()

    bottom = 0
    for di in range(len(genes_var_contribution.values)):
        _ = axes.bar(0, genes_var_contribution.values[di], bar_width, bottom=bottom, alpha=opacity,
                     color=colors[di],
                     error_kw=error_config, label=gene_names[di])
        bottom += genes_var_contribution.values[di]

    # Log1p-Normalization
    exp_matrix_df = np.log1p(exp_matrix_df)
    genes_var_series = exp_matrix_df.T.loc[gene_names,].var(1)
    genes_var_series_sum = genes_var_series.sum()
    genes_var_contribution = genes_var_series / genes_var_series_sum * 100

    bottom = 0
    for di in range(len(genes_var_contribution.values)):
        _ = axes.bar(1, genes_var_contribution.values[di], bar_width, bottom=bottom, alpha=opacity,
                     color=colors[di],
                     error_kw=error_config)
        bottom += genes_var_contribution.values[di]

    axes.set_xticks([0, 1])
    axes.set_xticklabels(['SCTransform', 'Log1p'])
    fig.supylabel('Variance Contribution %')
    plt.legend()
    fig.tight_layout()
    plt.show()
