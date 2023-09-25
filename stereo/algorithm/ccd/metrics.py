import numpy as np
import pandas as pd
import scipy.ndimage
import skimage.measure


def entropy2D(image):
    """
    Calculate the 2D Shannon entropy of an image.

    Parameters:
    - image (numpy.ndarray): The input image as a 2D NumPy array.

    Returns:
    - float: The Shannon entropy value of the image.
    """
    return skimage.measure.shannon_entropy(image)


def scatteredness2D(image, kernel):
    """
    Calculate the scatteredness measure of a 2D binary image.

    The scatteredness measure quantifies the degree of scattered or isolated regions in a binary image. It is calculated
    as the ratio of the number of connected components (objects) in the image to the total number of non-zero elements.
    The measure is further corrected based on the size of the neighborhood kernel.

    Parameters:
    - image (numpy.ndarray): The input binary image as a 2D NumPy array.
    - kernel (numpy.ndarray): The neighborhood kernel used for connectivity. It should be a 2D binary array,
                               typically representing 4 or 8 neighbors connectivity.

    Returns:
    - float: The scatteredness measure of the image.
    """

    _, num_objects = scipy.ndimage.label(image, structure=kernel, output=None)  # this assumes 4 neighbors connectivity
    # idea for scatteredness was to compute the number of connected components and divide it with number of existing non-zero elements # noqa
    # but this measure does not contain the information on percentage of non-zero elements in the matrix.
    # thus we multiply it with non-zero percentage (num non-zero / total num) creating just this formula
    # num_object/image.size
    # max value is based on neighbors size (if 4 then 1/4, if 8, 1/8), min value is 0 if there are no non-zero elements
    # [NOTE] add neighbourhood size for scatteredness calculation to params
    # [NOTE] try to find a heuristic to control the downsampling rate based on the proportion of cell number to area pixel number # noqa
    scatteredness = num_objects / image.size * (
            np.sum(kernel.ravel()) - 1)  # it is corrected with -1 for the central point # noqa
    return scatteredness


def calculate_spatial_metrics(adata, unique_cell_type, downsample_rate, annotation):
    """
    This function calculates cell type-specific global metrics, including entropy and scatteredness.

    Parameters:
    - adata (anndata.AnnData): Annotated data object containing spatial transcriptomics data.
    - unique_cell_type (list or array-like): List of unique cell types to calculate metrics for.
    - downsample_rate (int): The downsampling rate used to resize the tissue window.
    - annotation (str): Name of the column in 'adata.obs' that contains the cell type annotations.

    Returns:
    - list: A list containing the following spatial metrics for each cell type:
            * entropy (pd.Series): Entropy values for each cell type.
            * scatteredness (pd.Series): Scatteredness values for each cell type.
            * cell_t_images (dict): Dictionary mapping cell types to their corresponding tissue window images.
    """

    # calculate cell type specific global metrics
    adata.obs['x_coor'] = (adata.obsm['spatial'][:, 0])
    adata.obs['y_coor'] = (adata.obsm['spatial'][:, 1])
    cx_min = np.min(adata.obs['x_coor'])
    cx_max = np.max(adata.obs['x_coor'])
    cy_min = np.min(adata.obs['y_coor'])
    cy_max = np.max(adata.obs['y_coor'])

    # scatt_kernel is an odd number sized square matrix. Should always have 1 in the central position
    scatt_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    entropy = pd.Series(index=unique_cell_type, name='entropy', dtype=np.float64)
    scatteredness = pd.Series(index=unique_cell_type, name='scatteredness', dtype=np.float64)
    cell_t_images = {}
    for cell_t in unique_cell_type:
        tissue_window = np.zeros(shape=(
            int(np.ceil((cy_max - cy_min + 1) / downsample_rate)),
            int(np.ceil((cx_max - cx_min + 1) / downsample_rate))),
            dtype=np.int8)
        tissue_window[((adata.obs['y_coor'][adata.obs[annotation] == cell_t] - cy_min) / downsample_rate).astype(int), (
                (adata.obs['x_coor'][adata.obs[annotation] == cell_t] - cx_min) / downsample_rate).astype(int)] = 1
        cell_t_images[cell_t] = tissue_window

        entropy.loc[cell_t] = entropy2D(tissue_window)
        scatteredness.loc[cell_t] = scatteredness2D(tissue_window, kernel=scatt_kernel)
    return [entropy, scatteredness, cell_t_images]
