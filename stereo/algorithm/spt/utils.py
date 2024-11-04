from typing import Optional, Union

import numpy as np
import pandas as pd

from stereo.core.stereo_exp_data import StereoExpData

def get_cell_coordinates(data: StereoExpData, basis='spatial', ndmin=None):
    if basis not in data.cells_matrix:
        if f'X_{basis}' not in data.cells_matrix:
            raise ValueError(f"Please give the right basis, choose from {list(data.cells_matrix.keys())}.")
        else:
            position = data.cells_matrix[f'X_{basis}']
    else:
        position = data.cells_matrix[basis]
    if isinstance(position, pd.DataFrame):
        position = position.to_numpy()
    assert position.shape[1] >= 2, "The shape of the position should be at least (n_cells, 2)."

    if ndmin is not None:
        assert ndmin >= 2, "The ndmin should be at least 2."

    return position[:, :ndmin] if ndmin is not None else position

def get_lap_neighbor_data(
    data: StereoExpData,
    focused_cell_types: Optional[Union[np.ndarray, list, str]] = None
):
    assert 'is_lap_neighbor' in data.cells, 'Please run map_cell_to_LAP first.'
    cluster_res_key = data.tl.result['spa_track']['cluster_res_key']
    cells_sorted = data.cells.obs.sort_values(by='ptime')
    neighbor_cells = cells_sorted.index[cells_sorted['is_lap_neighbor']]
    neighbor_cell_types = data.cells.loc[neighbor_cells, cluster_res_key]
    if isinstance(focused_cell_types, str):
        focused_cell_types = [focused_cell_types]
    if focused_cell_types is not None:
        neighbor_cells = neighbor_cells[np.isin(neighbor_cell_types, focused_cell_types)]
    return data.sub_by_name(cell_name=neighbor_cells)