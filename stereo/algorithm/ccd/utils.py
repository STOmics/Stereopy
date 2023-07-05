'''
utils.py is created for wrappers, timeit decorator and
everything else that should be globally accessible and does not belong
to any specific class.

'''
import time
import seaborn as sns

from matplotlib import rcParams
from matplotlib.axes import Axes
from functools import wraps


import scanpy as sc

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f}s')
        return result
    return timeit_wrapper

@timeit
def plot_spatial(
    adata,
    annotation,
    ax: Axes,
    spot_size: float,
    palette = None,
    title: str = "",
    groups=None
):
    """
    Scatter plot in spatial coordinates.

    Parameters:
        - adata (AnnData): Annotated data object which represents the sample
        - annotation (str): adata.obs column used for grouping
        - ax (Axes): Axes object used for plotting
        - spot_size (int): Size of the dot that represents a cell. We are passing it as a diameter of the spot, while the plotting library uses radius therefore it is multiplied by 0.5 
        - palette (dict): Dictionary that represents a mapping between annotation categories and colors
        - title (str): Title of the figure
        - groups (list): If we want to plot only specific groups from annotation categories we will include only the categories present in groups parameter

    """
    s = spot_size * 0.1  # TODO: Ugly: consider using only one of: matplotlib or seaborn plots to have same spot size
    data = adata[adata.obs[annotation].isin(groups)] if groups else adata
    ax = sns.scatterplot(data=data.obs, hue=annotation, x=data.obsm['spatial'][:, 0], y=data.obsm['spatial'][:, 1], ax=ax, s=s, linewidth=0, palette=palette, marker='.')
    ax.set(yticklabels=[], xticklabels=[], title=title)
    ax.tick_params(bottom=False, left=False)
    ax.set_aspect("equal")
    sns.despine(bottom=True, left=True, ax=ax)

def set_figure_params(
        dpi: int,
        facecolor: str,
):
    rcParams['figure.facecolor'] = facecolor
    rcParams['axes.facecolor'] = facecolor
    rcParams["figure.dpi"] = dpi
