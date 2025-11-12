'''
utils.py is created for wrappers, timeit decorator and
everything else that should be globally accessible and does not belong
to any specific class.

'''
import time
from functools import wraps

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.axes import Axes

from stereo.log_manager import logger


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f'Function {func.__name__} took {total_time:.4f}s')
        return result

    return timeit_wrapper


@timeit
def plot_spatial(
        adata,
        annotation,
        ax: Axes,
        spot_size: float,
        palette=None,
        title: str = ""
):
    """
    Scatter plot in spatial coordinates.

    Parameters:
        - adata (AnnData): Annotated data object which represents the sample
        - annotation (str): adata.obs column used for grouping
        - ax (Axes): Axes object used for plotting
        - spot_size (int): Size of the dot that represents a cell. We are passing it as a diameter of the spot, while
                the plotting library uses radius therefore it is multiplied by 0.5
        - palette (dict): Dictionary that represents a mapping between annotation categories and colors
        - title (str): Title of the figure

    """
    s = spot_size * 0.2
    data = adata
    ax = sns.scatterplot(
        data=data.obs, hue=annotation, x=data.obsm['spatial'][:, 0], y=data.obsm['spatial'][:, 1],
        ax=ax, s=s, linewidth=0, palette=palette, marker='.'
    )
    ax.set(yticklabels=[], xticklabels=[], title=title)
    ax.tick_params(bottom=False, left=False)
    ax.set_aspect("equal")
    sns.despine(bottom=True, left=True, ax=ax)


original_rcParams = None


def set_figure_params(
        dpi: int,
        facecolor: str,
):
    global original_rcParams
    if original_rcParams is None:
        from copy import deepcopy
        original_rcParams = deepcopy(rcParams)
    rcParams['figure.facecolor'] = facecolor
    rcParams['axes.facecolor'] = facecolor
    rcParams["figure.dpi"] = dpi


def reset_figure_params():
    global original_rcParams
    if original_rcParams is not None:
        for key in rcParams.keys():
            if key in original_rcParams:
                rcParams[key] = original_rcParams[key]
        # original_rcParams = None


def csv_to_anndata(csv_file_path: str, annotation: str):
    """
    Convert csv data with cell ID, spatial coordinates (x and y), and cell type annotation
    to an Anndata object with empty X layer, for CCD analysis.
    cell ID data should be converted to string type.

    Parameters:
        - csv_file_path (str): path to .csv file with sample data
        - annotation (str): name of the column with cell type labels

    """
    df = pd.read_csv(csv_file_path)
    adata = ad.AnnData(np.zeros(shape=(df.shape[0], 1), dtype=np.float32), dtype=np.float32)
    adata.obs_names = df.loc[:, 'cell_ID'].values.astype('str').copy()
    adata.obs[annotation] = df.loc[:, annotation].values.copy()
    adata.obsm['spatial'] = df.loc[:, ['x', 'y']].values.copy()
    del df
    return adata
