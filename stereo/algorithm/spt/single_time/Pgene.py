import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
from scipy import stats
from sklearn import preprocessing
import multiprocessing as mp
from pygam import LinearGAM, s, f
import statsmodels.stats as stat
from scipy.signal import savgol_filter
import statsmodels.formula.api as smf
import scipy.stats
from scipy import sparse
import gc
from anndata import AnnData
from matplotlib.axes import Axes

from stereo.core.stereo_exp_data import StereoExpData
from stereo.preprocess.qc import cal_n_cells
from stereo.preprocess.filter import filter_genes, filter_by_clusters
from stereo.log_manager import logger


"""
Caculate JS score to determine data trend is increase or decrease

"""


def js_score(gam_fit, grid_X):
    """
    Parameters
    ----------
    gam_fit :
        Fitted model by pyGAM

    grid_X: array
        An array value grided by pyGAM's generate_X_grid function


    Returns
    -------
    trend: str
        Mark the fitted model is increase or decrease

    """

    def JS_divergence(p, q):
        """
        Parameters
        ----------
        p,q :
                Two same length arrays
                p: array fitted by model
                q: standard distribution array

        Returns
        -------
            JS score. More smaller value indicate the distribution of inputed data is more similar with standard distribution
        """
        M = (p + q) / 2
        return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)

    x = [i for i in range(100)]
    l = np.array(x).reshape(-1, 1)
    increase_trend = preprocessing.MaxAbsScaler().fit_transform(l).reshape(1, -1)[0]
    decrease_trend = increase_trend[::-1]
    decrease_score = JS_divergence(gam_fit.predict(grid_X), decrease_trend)
    increase_score = JS_divergence(gam_fit.predict(grid_X), increase_trend)
    pattern_dict = {"decrease": decrease_score, "increase": increase_score}
    gene_trend = min(pattern_dict, key=pattern_dict.get)
    return gene_trend


##Fit gene expression and ptime by generalized additive model
##Identify pesudotime-dependent genes may drive cell transition

"""
Filter genes by minimum expression proporation and cluster differential expression.
Cluster differential expression is used to as a reference to order gene.
"""


def filter_gene(
    data: StereoExpData,
    use_col: str,
    min_exp_prop: float,
    hvg_gene: int = 2000
)->AnnData:
    """
    Filter genes by minimum expression proporation and cluster differential expression.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    min_exp_prop
        Minimum expression proporation.
    abs_FC
        Log2 foldchange in differential expression.
        
    Returns
    ----------
    :class:`~anndata.AnnData`
        The :class:`~anndata.AnnData` object formed by filtered genes.
    """
    ptime_list = list(data.cells['ptime'])
    if sorted(ptime_list) == ptime_list:
        pass
    else:
        raise Exception("error: Please sort adata by ptime.")

    # cluster_order = data.cells.obs.groupby([use_col]).mean().sort_values(["ptime"]).index
    n_cells_in_genes = cal_n_cells(data.exp_matrix if data.raw is None else data.raw.exp_matrix)
    min_prop_filtered_genes = n_cells_in_genes > int(data.n_cells * min_exp_prop)
    data.tl.highly_variable_genes(n_top_genes=hvg_gene)
    gene_list_lm = data.gene_names[min_prop_filtered_genes & data.genes['highly_variable']]
    data.sub_by_index(gene_index=min_prop_filtered_genes) 
    logger.info("Cell number" + "\t" + str(data.n_cells))
    logger.info("Gene number" + "\t" + str(len(gene_list_lm)))
    return data


def GAM_gene_fit(exp_gene_list):

    """
    Parameters
    ----------
    exp_gene_list : multi layer list

    exp_gene_list[0]: dataframe
                    columns : ptime,gene_expression
    exp_gene_list[1]: gene_name

    """

    r_list = list()
    trend_list = list()
    gene_list = list()
    pvalue_list = list()

    df_new = exp_gene_list[0]
    gene = exp_gene_list[1]
    x = df_new[["ptime"]].values
    y = df_new[gene]
    gam = LinearGAM(s(0, n_splines=8))
    gam_fit = gam.gridsearch(x, y, progress=False)
    grid_X = gam_fit.generate_X_grid(term=0)
    r_list.append(gam_fit.statistics_["pseudo_r2"]["explained_deviance"])
    pvalue_list.append(gam_fit.statistics_["p_values"][0])
    gene_list.append(gene)

    trend_list.append(js_score(gam_fit, grid_X))

    ## sort gene by fdr and R2
    df_batch_res = pd.DataFrame(
        {
            "gene": gene_list,
            "pvalue": pvalue_list,
            "model_fit": r_list,
            "pattern": trend_list,
        }
    )
    return df_batch_res


"""
function:

Call ptime_gene_GAM() by  multi-process computing to improve operational speed

"""


def ptime_gene_GAM(data: StereoExpData, core_number: int = 3) -> pd.DataFrame:
    """
    Fit GAM model by formula gene_exp ~ Ptime.

    Call GAM_gene_fit() by multi-process computing to improve operational speed.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    core_number
        Number of processes for caculating.

    Returns
    -------
    :class:`~pandas.DataFrame`
        An :class:`~pandas.DataFrame` object, each column is one index.

        - pvalue: calculated from GAM
        - R2: a goodness-of-fit measure. larger value means better fit
        - pattern: increase or decrease. drection of gene expression changes across time
        - fdr: BH fdr

    """
    # perform GAM model on each gene
    gene_list_for_gam = data.gene_names[data.genes['highly_variable']]

    # df_exp_filter = pd.DataFrame(data=data.exp_matrix.toarray() if data.issparse() else data.exp_matrix,
    #                              index=data.cell_names,
    #                              columns=data.gene_names)

    logger.info(f"Genes number fitted by GAM model:  {len(gene_list_for_gam)}")
    if core_number >= 1:
        para_list = list()
        for gene in gene_list_for_gam:
            gene_expression = data.get_exp_matrix(gene_list=gene)
            if data.issparse():
                gene_expression = gene_expression.toarray()
            gene_expression = gene_expression.flatten()
            df_new = pd.DataFrame({
                "ptime": list(data.cells['ptime']),
                gene: gene_expression
            })
            # df_new=df_new.loc[df_new[gene]>0]
            para_list.append((df_new, gene))
        p = mp.Pool(core_number)
        df_res = p.map(GAM_gene_fit, para_list)
        p.close()
        p.join()
        df_res = pd.concat(df_res)

        del para_list
        gc.collect()
    fdr = stat.multitest.fdrcorrection(np.array(df_res["pvalue"]))[1]
    df_res["fdr"] = fdr
    df_res.index = list(df_res["gene"])
    return df_res


"""
function:

Split cells sorted by ptime into widonws.

Order genes according number id of the maximum expression window

"""


def order_trajectory_genes(data:StereoExpData, df_sig_res:pd.DataFrame, cell_number:int):
    """
    Split cells sorted by ptime into widonws.

    Order genes according number id of the maximum expression window.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    df_sig_res
        Return dataframe by ptime_gene_GAM() after filtering as significat gene dataframe.
    cell_number
        Cell number within splited window.

    Returns
    -------
    :class:`~pandas.DataFrame`

        - columns:Sortted significant genes expression matrix according to mean expression value in windows
        - index: cell_id
    """
    ptime_sort_exp_matrix = data.exp_matrix.toarray() if data.issparse() else data.exp_matrix

    df_exp_filter = pd.DataFrame(
        data=ptime_sort_exp_matrix, index=data.cell_names, columns=data.gene_names
    )

    df_one_cell_exp_sig = df_exp_filter.loc[:, df_sig_res.index]
    sig_genes = df_one_cell_exp_sig.columns
    max_cell = pd.DataFrame(index=sig_genes, columns=["max"])
    df_one_cell_exp_matrix = np.array(df_one_cell_exp_sig)

    # windows number
    window_number = math.ceil(len(df_one_cell_exp_sig) / cell_number)

    df_one_cell_exp_matrix.resize(
        (cell_number * window_number, len(sig_genes)), refcheck=False
    )
    # divide block
    block_matrix = df_one_cell_exp_matrix.reshape(
        (window_number, cell_number, len(sig_genes))
    )
    window_matrix = np.sum(block_matrix, axis=1)

    # cell number in each window
    cell_in_window = np.array(
        [[cell_number]] * (window_number - 1)
        + [[len(df_one_cell_exp_sig) - cell_number * (window_number - 1)]]
    )

    # mean expression in each window
    mean_window_matrix = window_matrix / cell_in_window
    window_exp = pd.DataFrame(
        data=mean_window_matrix,
        index=["window_" + str(i) for i in range(window_number)],
        columns=sig_genes,
    )

    for i in sig_genes:
        max_cell.loc[i, "max"] = window_exp[i].idxmax()

    ptime = np.array(data.cells["ptime"])
    ptime.resize((window_number, cell_number), refcheck=False)
    mean_ptime = ptime.sum(axis=1) / cell_in_window.T
    endog = pd.DataFrame(
        data=mean_ptime.T,
        index=["window_" + str(i) for i in range(window_number)],
        columns=["ptime"],
    )
    max_cell["ptime"] = endog.loc[max_cell["max"].values].values
    max_cell = max_cell.iloc[np.argsort(max_cell["ptime"].values), :]
    sort_window_exp = window_exp.loc[:, max_cell.index]
    logger.info(f"Finally selected {len(sort_window_exp.columns)} genes.")
    ## return gene order
    gene_sort_list = sort_window_exp.columns
    df_one_cell_exp_sort = df_one_cell_exp_sig[gene_sort_list]
    return df_one_cell_exp_sort


"""
function:

Plot ordered gene expression heatmap of the selected candidate trajectory genes

"""


def plot_trajectory_gene_heatmap(
    sig_gene_exp_order: pd.DataFrame,
    smooth_length:int,
    cmap_name: str ="twilight_shifted",
    gene_label_size:int =30,
    fig_width=8,
    fig_height=10
):
    """
    Parameters
    ----------
    sig_gene_exp_order
        Gene ordered expression dataframe.
    smooth_length
        length of smoothing window
    cmap_name
        Color palette
    fig_width,fig_height
        The width and height of figure
    Returns
    -------
        A heatmap plot, column-representing cells, row-representing genes.

    """
    ## only show TF gene
    # TF_file=pd.read_table('hs_hgnc_tfs.txt',header=None)
    # cell_TF_exp=cell_exp[cell_exp.columns[cell_exp.columns.isin(TF_file[0])]]

    sort_window_exog_z = stats.zscore(sig_gene_exp_order, axis=0)
    last_pd = pd.DataFrame(
        data=sort_window_exog_z.T,
        columns=sort_window_exog_z.index,
        index=sort_window_exog_z.columns,
    )

    # smooth data
    last_pd_smooth = savgol_filter(last_pd, smooth_length, 1)
    last_pd_smooth = pd.DataFrame(last_pd_smooth)
    last_pd_smooth.columns = last_pd.columns
    last_pd_smooth.index = last_pd.index

    #fig = plt.figure(figsize=(8, 10))
    fig = plt.figure(figsize=(fig_width,fig_height))
    #ax1 = plt.subplot2grid((8, 10), (0, 0), colspan=10, rowspan=8)
    pseudotime_gene_heatmap = sns.heatmap(
        last_pd_smooth,
        cmap=cmap_name,
        cbar_kws={"shrink": 0.3, "label": "normalized expression"},
    )
    cbar = pseudotime_gene_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    ## add cell type
    # df_cell=pd.DataFrame(sig_gene_exp_order.index)
    # df_cell[1]=list(adata.obs['cluster'])
    # plt.axis('off')
    # cell_line_plot = sns.histplot(data = df_cell, x = 0,hue=1,ax=ax2)

    # cell_line_plot.set_frame_on(False)
    # cell_line_plot.get_legend().remove()
    # cell_line_plot._legend.remove()
    pseudotime_gene_heatmap.figure.axes[-1].yaxis.label.set_size(25)
    pseudotime_gene_heatmap.xaxis.tick_top()
    pseudotime_gene_heatmap.set_xticks([])
    pseudotime_gene_heatmap.yaxis.set_tick_params(labelsize=gene_label_size)
    plt.xticks(rotation=90)
    fig.tight_layout()
    return fig


"""
function:

Plot one trajectory gene

"""


def plot_trajectory_gene(
    data: StereoExpData,
    use_col: str,
    gene_name: str,
    line_width: int = 5,
    show_cell_type: bool = False, 
    point_size=20)->Axes:
    """
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    gene_name
        Gene used to plot.
    line_width
        Widthe of fitting line.
    show_cell_type
        Whether to show cell type in plot.
    point_size
        The size of point
    Returns
    -------
    :class:`~matplotlib.axes.Axes`
        An :class:`~matplotlib.axes.Axes` object. X axis indicates pseduotime and y axis indicates gene expression value.

    """

    # gene_expression = pd.DataFrame(
    #     data=data.exp_matrix, 
    #     index=data.cell_names,
    #     columns=data.gene_names
    # )
    gene_expression = data.get_exp_matrix(gene_list=[gene_name])
    if data.issparse():
        gene_expression = gene_expression.toarray().flatten()
    df_new = pd.DataFrame(
        {
            "ptime": list(data.cells["ptime"]),
            gene_name: gene_expression,
            "cell_type": list(data.cells[use_col]),
        }
    )
    # df_new=df_new.loc[df_new[gene_name]>0]
    x_ptime = df_new[["ptime"]].values
    y_exp = df_new[gene_name]
    gam = LinearGAM(s(0, n_splines=10))
    gam_res = gam.gridsearch(x_ptime, y_exp,progress=False)

    fig, axs = plt.subplots(figsize=(10,6))
    XX = gam_res.generate_X_grid(term=0)
    axs.plot(XX, gam.predict(XX), color="#aa4d3d", linewidth=line_width)
    if show_cell_type == True:
        sns.scatterplot(
            x="ptime", y=gene_name, palette="deep", ax=axs, data=df_new,s=point_size, hue="cell_type"
        )
        plt.gca().legend().set_title("")
        plt.legend(fontsize="xx-large", loc=(1.01, 0.5))
    else:
        sns.scatterplot(
            x="ptime", y=gene_name, cmap="plasma", ax=axs, s=point_size,data=df_new, c=x_ptime
        )
        norm = plt.Normalize(df_new['ptime'].min(), df_new['ptime'].max())
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        axs.figure.colorbar(sm,ax=axs)
    # if show_cell_type == True:
    #     plt.gca().legend().set_title("")
    #     plt.legend(fontsize="xx-large", loc=(1.01, 0.5))

    plt.title(gene_name, fontsize=30)
    plt.xlabel("ptime", fontsize=30)
    plt.ylabel("expression", fontsize=30)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # norm = plt.Normalize(df_new['ptime'].min(), df_new['ptime'].max())
    # sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    # sm.set_array([])
    #axs.get_legend().remove()
    # axs.figure.colorbar(sm,ax=axs)
    fig.tight_layout()

    return fig


"""
funtion:

Plot a group of trajectory genes

"""


def plot_trajectory_gene_list(
    adata,
    gene_name_list,
    col_num=4,
    title_fontsize=25,
    label_fontsize=22,
    line_width=5,
    fig_legnth=10,
    fig_width=8,
):
    """
    Parameters
    ----------
    adata : AnnData object.

    gene_name_list: List object
             gene list used to plot
    col_num: int
            Number of genes displayed per line in picature
            (Default: 4)
    title_fontsize: int
            title fontsize of picture
            (Default: 25)
    label_fontsize: int
             x and y label fontsize
            (Default: 22)
    fig_legnth,fig_width:
            The legenth and width of picture size
            (Default: 10,8)


    Returns
    -------

    ax: fig object
        x axis indicate pseduotime; y axis indicate gene expression value

    """

    gene_number = len(gene_name_list)
    row_num = math.ceil(gene_number / col_num)
    fig, axs = plt.subplots(
        ncols=col_num,
        nrows=row_num,
        figsize=(fig_legnth, fig_width),
        sharey=False,
        sharex=True,
    )
    i = -1

    gene_expression = pd.DataFrame(
        data=adata.X, index=adata.obs.index, columns=adata.var.index
    )

    for m in range(row_num):
        for n in range(col_num):
            i = i + 1
            if i > gene_number - 1:
                break
            gene_name = gene_name_list[i]

            ax = axs[m, n]
            df_new = pd.DataFrame(
                {
                    "ptime": list(adata.obs["ptime"]),
                    gene_name: list(gene_expression[gene_name]),
                }
            )
            # df_new=df_new.loc[df_new[gene_name]>0]
            x_ptime = df_new[["ptime"]].values
            y_exp = df_new[gene_name]
            gam = LinearGAM(s(0, n_splines=10))
            gam_res = gam.gridsearch(x_ptime, y_exp)
            XX = gam_res.generate_X_grid(term=0)
            ax.plot(XX, gam.predict(XX), color="#aa4d3d", linewidth=line_width)
            ax.scatter(x_ptime, y_exp, cmap="plasma", c=x_ptime)
            ax.set_title(gene_name, fontsize=title_fontsize)
    for pos in range(gene_number, row_num * col_num):
        axs.flat[pos].set_visible(False)

    # fig.tight_layout()
    fig.text(0.5, -0.04, "ptime", ha="center", fontsize=label_fontsize)
    fig.text(
        0.01,
        0.5,
        "expression ",
        va="center",
        rotation="vertical",
        fontsize=label_fontsize,
    )
    # plt.tight_layout()

    fig.tight_layout()
    fig.subplots_adjust(left=0.06)
    return ax


# 01 filter gene by expression
# sub_adata=sti.Pgene.filter_gene(sub_adata,min_exp_prop=0.1,hvg_gene=3000)

# 02 fit GAM model
# df_res  = sti.Pgene.ptime_gene_GAM(sub_adata,core_number=5)

# 03 filter gene by GAM model indicators
# df_sig_res = df_res.loc[(df_res['model_fit']>0.05) & (df_res['fdr']<0.05)]

# 04 order trajectory genes
# sort_exp_sig = sti.Pgene.order_trajectory_genes(sub_adata,df_sig_res,cell_number=20)

# 05 plot trajectory gene heatmap
# sti.Pgene.plot_trajectory_gene_heatmap(sort_exp_sig,smooth_length=100,gene_label_size=20)

# 06 plot one or multiple trajectory genes
# sti.Pgene.plot_trajectory_gene(sub_adata,gene_name='APOE',show_cell_type=False)
# sti.Pgene.plot_trajectory_gene_list(sub_adata,gene_name_list=['COL1A1','ACTB','TNC','AQP1'],col_num=2)
