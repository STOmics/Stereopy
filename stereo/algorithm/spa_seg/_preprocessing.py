import numpy as np
from anndata import AnnData

from ._constant import CellbinSize, SpotSize


def add_spot_pos(
    adata: AnnData,
    bin_type: str,
    spatial_key: str
):
    adata.obs["array_row"] = adata.obsm[spatial_key][:, 0]
    adata.obs["array_col"] = adata.obsm[spatial_key][:, 1]

    if np.min(adata.obsm[spatial_key]) < 0:
        adata.obs['array_col'] = (adata.obs['array_col'].values - adata.obs['array_col'].values.min())
        adata.obs['array_row'] = (adata.obs['array_row'].values - adata.obs['array_row'].values.min())

    """
    The scale factor refer to the code in stLearn:
    https://github.com/BiomedicalMachineLearning/stLearn/blob/master/stlearn/wrapper/read.py
    """

    if bin_type == 'cell_bins':
        scale = 1.0 / CellbinSize.CELLBIN_SIZE.value
        # adata.uns["spot_size"] = SpotSize.STEREO_SPOT_SIZE.value
    elif bin_type == "bins":
        scale = 1
        # adata.uns["spot_size"] = SpotSize.STEREO_SPOT_SIZE.value
    else:
        raise ValueError("Invalid bin type, available options: 'cell_bins', 'bins'")

    adata.obs['array_col'] = adata.obs['array_col'] * scale
    adata.obs['array_row'] = adata.obs['array_row'] * scale
    adata.obsm['spatial_original'] = adata.obsm[spatial_key].copy()
    adata.obsm[spatial_key][:, 0] = adata.obs['array_row'].to_numpy(copy=True)
    adata.obsm[spatial_key][:, 1] = adata.obs['array_col'].to_numpy(copy=True)

    return adata