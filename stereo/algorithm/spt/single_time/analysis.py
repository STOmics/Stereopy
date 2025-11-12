import statsmodels.api as sm
import numpy as np
import pandas as pd
import math
from scipy import stats
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc


def glm_window(adata, cell_numer=20, top_rela_gene_number=30):
    """Display the most relevant data in window mode

    Parameters
    ----------
    adata
        Anndata
    cell_numer, optional
        Expected number of cells in a window, by default 20
    top_rela_gene_number, optional
        The number of most relevant genes, by default 30

    Returns
    -------
        Heatmap
    """
    # 1. sort
    adata = adata[np.argsort(adata.obs["ptime"].values), :].copy()
    ptime_sort_matrix = adata.X.copy()

    # 2. prepare the glm input
    exog = pd.DataFrame(
        data=ptime_sort_matrix, index=adata.obs.index, columns=adata.var.index
    )
    endog = adata.obs["ptime"]
    gamma_model = sm.GLM(endog, exog, family=sm.families.Gamma())
    gamma_results = gamma_model.fit()
    pvalues = gamma_results.pvalues

    # 3. extract the most correlated genes
    rela_genes = pvalues[np.argsort(pvalues.values)].index[:top_rela_gene_number]
    max_cell = pd.DataFrame(index=rela_genes, columns=["max"])
    rela_exog = exog.loc[:, rela_genes]
    rela_exog_matrix = np.array(rela_exog)

    # 4. calculate the cell number of each window (rounded up), the number of windows
    cell_number = cell_numer
    window_number = math.ceil(adata.n_obs / cell_number)

    # 5. reshape
    rela_exog_matrix.resize(
        (cell_number * window_number, len(rela_genes)), refcheck=False
    )

    # 6. divide into blocks
    block_matrix = rela_exog_matrix.reshape(
        (window_number, cell_number, len(rela_genes))
    )

    # 7. add each block column
    window_matrix = np.sum(block_matrix, axis=1)

    # 8. calculate the cell number of each window
    cell_in_window = np.array(
        [[cell_number]] * (window_number - 1)
        + [[adata.n_obs - cell_number * (window_number - 1)]]
    )

    # 9. get window average
    mean_window_matrix = window_matrix / cell_in_window
    window_exog = pd.DataFrame(
        data=mean_window_matrix,
        index=["window_" + str(i) for i in range(window_number)],
        columns=rela_genes,
    )

    # 10. rank related genes
    for i in rela_genes:
        max_cell.loc[i, "max"] = window_exog[i].idxmax()

    ptime = np.array(adata.obs["ptime"])
    ptime.resize((window_number, cell_number), refcheck=False)
    mean_ptime = ptime.sum(axis=1) / cell_in_window.T
    endog = pd.DataFrame(
        data=mean_ptime.T,
        index=["window_" + str(i) for i in range(window_number)],
        columns=["ptime"],
    )
    max_cell["ptime"] = endog.loc[max_cell["max"].values].values
    max_cell = max_cell.iloc[np.argsort(max_cell["ptime"].values), :]
    sort_window_exog = window_exog.loc[:, max_cell.index]

    # 11. z-score
    zscore_matrix = stats.zscore(sort_window_exog, axis=0)

    # 12. normalization
    norm_matrix = preprocessing.normalize(zscore_matrix, axis=0, norm="max")
    last_pd = pd.DataFrame(
        data=norm_matrix, index=sort_window_exog.index, columns=sort_window_exog.columns
    )

    # 13. heatmap
    plt.figure(figsize=(15, 3))
    gg = sns.heatmap(last_pd, cmap="plasma")

    return adata


def one_gene(adata, gene_name):
    """Scatter plot of one selected gene

    Parameters
    ----------
    adata
        Anndata
    gene_name
        Selected gene

    Returns
    -------
        Scatter plot
    """
    ptime = adata.obs["ptime"]
    gene = adata[:, gene_name].X.T[0]

    parameter = np.polyfit(ptime, gene, deg=3)

    p = np.poly1d(parameter)

    ax = sc.pl.scatter(
        adata, x="ptime", y=gene_name, color=gene_name, color_map="plasma", show=False
    )
    ax.plot(ptime, p(ptime), color="black")

    return ax
