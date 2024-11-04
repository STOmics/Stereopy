from typing import Tuple, Optional, Literal, Union

import numpy as np
import pandas as pd

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.core.stereo_exp_data import StereoExpData

from stereo.algorithm.spt.single_time import (
    assess_start_cluster,
    set_start_cells,
    # Lasso,
    auto_estimate_para,
    calc_alpha_by_moransI,
    get_ot_matrix,
    get_ptime,
    get_velocity_grid,
    get_velocity,
    auto_get_start_cluster,
    lasso_select,
    VectorField,
    nearest_neighbors,
    least_action,
    map_cell_to_LAP,
    filter_gene,
    ptime_gene_GAM,
    order_trajectory_genes,
    Trainer
)
# from stereo.algorithm.spt.plot import PlotSpaTrack
from stereo.algorithm.spt.utils import get_lap_neighbor_data, get_cell_coordinates
from stereo.plots.plot_spa_track import PlotSpaTrack
    
class SpaTrack(AlgorithmBase):
    def main(
        self,
        cluster_res_key: str = 'cluster'
    ):
        if cluster_res_key not in self.pipeline_res:
            raise KeyError(f'Cannot find clustering result by key {cluster_res_key}')
        if 'spa_track' not in self.pipeline_res:
            self.pipeline_res['spa_track'] = {}
        self.cluster_res_key = cluster_res_key
        self.pipeline_res['spa_track']['cluster_res_key'] = cluster_res_key
        self.plot = PlotSpaTrack(stereo_exp_data=self.stereo_exp_data, pipeline_res=self.pipeline_res)
        return self

    def assess_start_cluster(self):
        mean_entropy_sorted_in_cluster = assess_start_cluster(self.stereo_exp_data, use_col=self.cluster_res_key)
        self.pipeline_res['Mean_Entropy_sorted_in_cluster'] = mean_entropy_sorted_in_cluster
        
    
    def set_start_cells(
        self,
        select_way: Literal["coordinates", "cell_type"],
        cell_type: Optional[str] = None,
        start_point: Optional[Tuple[int, int]] = None,
        spatial_key: Optional[str] = None,
        split: bool = False,
        n_clusters: int = 2,
        n_neigh: int = 5
    ):
        spatial_key = self.stereo_exp_data.spatial_key if spatial_key is None else spatial_key
        start_cells = set_start_cells(
            self.stereo_exp_data,
            select_way=select_way,
            cell_type=cell_type,
            start_point=start_point,
            basis=spatial_key,
            use_col=self.cluster_res_key,
            split=split,
            n_clusters=n_clusters,
            n_neigh=n_neigh
        )
        res_key: str = 'start_cells'
        self.stereo_exp_data.cells[res_key] = False
        self.stereo_exp_data.cells.loc[self.stereo_exp_data.cell_names[start_cells], res_key] = True
        self.stereo_exp_data.cells[res_key] = self.stereo_exp_data.cells[res_key].astype('category')

        return start_cells

    def auto_estimate_para(
        self,
        spatial_key: Optional[str] = None,
        hvg_gene_number: int = 2000,
        # hvg_key: Optional[str] = None
        
    ):
        spatial_key = self.stereo_exp_data.spatial_key if spatial_key is None else spatial_key
        return auto_estimate_para(
            self.stereo_exp_data, basis=spatial_key, hvg_gene_number=hvg_gene_number
        )
    
    def calc_alpha_by_moransI(
        self,
        spatial_key: Optional[str] = None
    ):
        spatial_key = self.stereo_exp_data.spatial_key if spatial_key is None else spatial_key
        return calc_alpha_by_moransI(self.stereo_exp_data, basis=spatial_key)
    
    def get_ot_matrix(
        self,
        data_type: str,
        alpha1: int = 0.5,
        alpha2: int = 0.5,
        spatial_key: Optional[str] = None,
        n_pcs: int = 50,
        pca_res_key: Optional[str] = None,
    ):
        res_key: str = 'trans'
        spatial_key = self.stereo_exp_data.spatial_key if spatial_key is None else spatial_key
        self.pipeline_res[res_key] = get_ot_matrix(
            self.stereo_exp_data, data_type, alpha1=alpha1, alpha2=alpha2, basis=spatial_key,
            pattern='run', n_pcs=n_pcs, pca_res_key=pca_res_key
        )
        return self.pipeline_res[res_key]
    
    def get_ptime(
        self
    ):
        start_cells_key: str = 'start_cells'
        ot_mtx_key: str = 'trans'
        res_key: str = 'ptime'
        self.stereo_exp_data.cells[res_key] = get_ptime(
            self.stereo_exp_data, start_cells_key=start_cells_key, ot_mtx_key=ot_mtx_key
        )
        return self.stereo_exp_data.cells[res_key].to_numpy()
    
    def get_velocity_grid(
        self,
        spatial_key: Optional[str] = None,
        grid_num: int = 50,
        smooth: float = 0.5,
        density: float = 1.0,
    ):
        spatial_key = self.stereo_exp_data.spatial_key if spatial_key is None else spatial_key
        P = get_cell_coordinates(self.stereo_exp_data, basis=spatial_key, ndmin=2)
        V = get_cell_coordinates(self.stereo_exp_data, basis=f'velocity_{spatial_key}', ndmin=2)
        P_grid, V_grid = get_velocity_grid(self.stereo_exp_data, P, V, grid_num=grid_num, smooth=smooth, density=density)
        self.pipeline_res['P_grid'] = P_grid
        self.pipeline_res['V_grid'] = V_grid
        return P_grid, V_grid
    
    def get_velocity(
        self,
        spatial_key: Optional[str] = None,
        n_neigh_pos: int = 10,
        n_neigh_gene: int = 0,
        grid_num: int = 50,
        smooth: float = 0.5,
        density: float = 1.0
    ):
        pseudotime_key: str = 'ptime'
        ot_mtx_key: str = 'trans'
        spatial_key = self.stereo_exp_data.spatial_key if spatial_key is None else spatial_key
        P_grid, V_grid, trans_neigh, velocity = get_velocity(
            self.stereo_exp_data, basis=spatial_key, n_neigh_pos=n_neigh_pos, n_neigh_gene=n_neigh_gene,
            grid_num=grid_num, smooth=smooth, density=density, pseudotime_key=pseudotime_key,
            ot_mtx_key=ot_mtx_key
        )
        self.pipeline_res['spa_track']['velocity_key'] = f'velocity_{spatial_key}'
        self.pipeline_res['spa_track']['velocoty_spatial_key'] = spatial_key
        # self.pipeline_res[f'velocity_{spatial_key}'] = velocity
        self.pipeline_res.set_value(f'velocity_{spatial_key}', velocity)
        self.pipeline_res['P_grid'] = P_grid
        self.pipeline_res['V_grid'] = V_grid
        self.pipeline_res['trans_neigh'] = trans_neigh

    def auto_get_start_cluster(
        self,
        clusters: Optional[list] = None
    ):
        ot_mtx_key: str = 'trans'
        return auto_get_start_cluster(
            self.stereo_exp_data, use_col=self.cluster_res_key, ot_mtx_key=ot_mtx_key, clusters=clusters
        )
    
    def lasso_select(
        self,
        cell_type: Optional[str] = None,
        spatial_key: Optional[str] = None,
        bg_color: str = '#2F2F4F',
        palette: Union[str, list] = 'stereo_30',
        marker: str = 'o',
        marker_size: int = 5,
        width: int = 500,
        height: int = 500,
        invert_y: bool = True
    ):
        spatial_key = self.stereo_exp_data.spatial_key if spatial_key is None else spatial_key
        return lasso_select(
            self.stereo_exp_data, use_col=self.cluster_res_key, cell_type=cell_type, basis=spatial_key,
            bg_color=bg_color, palette=palette, marker=marker, marker_size=marker_size,
            width=width, height=height, invert_y=invert_y
        )
    
    def create_vector_field(
        self,
        normalize: bool = False,
        **kwargs,
    ):
        if 'velocity_key' not in self.pipeline_res['spa_track']:
            raise KeyError('Cannot find velocity in result, please run get_velocity first.')
        if self.pipeline_res['spa_track']['velocity_key'] not in self.pipeline_res:
            raise KeyError('Cannot find velocity in result, please run get_velocity first.')
        
        spatial_key = self.pipeline_res['spa_track']['velocoty_spatial_key']
        self.pipeline_res['vector_field'] = VectorField(
            self.stereo_exp_data, basis=spatial_key, normalize=normalize, **kwargs
        )
        self.pipeline_res['spa_track']['vf_spaital_key'] = spatial_key

    def set_lap_endpoints(
        self,
        start_coordinate: Tuple[Union[int, float], Union[int, float]],
        end_coordinate: Tuple[Union[int, float], Union[int, float]],
    ):
        if len(start_coordinate) != 2 or len(end_coordinate) != 2:
            raise ValueError('Coordinates must be a tuple of 2 elements')
        
        start_cell = nearest_neighbors(start_coordinate, self.stereo_exp_data.position[:, :2])[0][0]
        end_cell = nearest_neighbors(end_coordinate, self.stereo_exp_data.position[:, :2])[0][0]

        self.stereo_exp_data.cells['lap_endpoints'] = 'others'
        self.stereo_exp_data.cells.loc[self.stereo_exp_data.cell_names[start_cell], 'lap_endpoints'] = 'start'
        self.stereo_exp_data.cells.loc[self.stereo_exp_data.cell_names[end_cell], 'lap_endpoints'] = 'end'
        self.stereo_exp_data.cells['lap_endpoints'] = self.stereo_exp_data.cells['lap_endpoints'].astype('category')

    def least_action(
        self,
        n_points: int = 25,
        n_neighbors: int = 100,
        dt_0=1,
        EM_steps=2
    ):
        if 'vector_field' not in self.pipeline_res:
            raise KeyError('Cannot find vector field in result, please run create_vector_field first.')
        if 'lap_endpoints' not in self.stereo_exp_data.cells:
            raise KeyError('Cannot find LAP endpoints in cells/obs, please run set_lap_endpoints first.')
        lap_endpoints = self.stereo_exp_data.cells['lap_endpoints'].to_numpy()
        init_cells = self.stereo_exp_data.cell_names[lap_endpoints == 'start']
        if len(init_cells) == 1:
            init_cells = init_cells[0]
        target_cells = self.stereo_exp_data.cell_names[lap_endpoints == 'end']
        if len(target_cells) == 1:
            target_cells = target_cells[0]
        spaital_key = self.pipeline_res['spa_track']['vf_spaital_key']
        self.pipeline_res['spa_track']['lap_spaital_key'] = spaital_key
        return least_action(
            self.stereo_exp_data, init_cells, target_cells, basis=spaital_key, vf_key=f'VecFld_{spaital_key}',
            vecfld=self.pipeline_res['vector_field'], n_points=n_points, n_neighbors=n_neighbors, dt_0=dt_0, EM_steps=EM_steps
        )
    
    def map_cell_to_LAP(self, cell_neighbors=150):
        if (spatial_key := self.pipeline_res['spa_track'].get('lap_spaital_key')) is None:
            raise KeyError('Cannot find LAP spatial key in result, please run least_action first.')
        lap_key = f"LAP_{spatial_key}"
        if lap_key not in self.pipeline_res:
            raise KeyError('Cannot find LAP in result, please run least_action first.')
        
        LAP_ptime, LAP_neighbor_cells = map_cell_to_LAP(self.stereo_exp_data, basis=spatial_key, cell_neighbors=cell_neighbors)
        neighbor_cell_names = self.stereo_exp_data.cell_names[LAP_neighbor_cells]
        self.stereo_exp_data.cells.loc[neighbor_cell_names, 'ptime'] = LAP_ptime
        self.stereo_exp_data.cells['is_lap_neighbor'] = False
        self.stereo_exp_data.cells.loc[neighbor_cell_names, 'is_lap_neighbor'] = True
    
    def filter_genes(
        self,
        min_exp_prop_in_genes: float,
        n_hvg: int = 2000,
        focused_cell_types: Optional[Union[np.ndarray, list, str]] = None
    ):
        lap_neighbor_data = get_lap_neighbor_data(self.stereo_exp_data, focused_cell_types)

        self.pipeline_res['spa_track']['filter_genes'] = {
            'min_exp_prop_in_genes': min_exp_prop_in_genes,
            'n_hvg': n_hvg,
            'focused_cell_types': focused_cell_types
        }

        return filter_gene(
            lap_neighbor_data, use_col=self.cluster_res_key, min_exp_prop=min_exp_prop_in_genes, hvg_gene=n_hvg
        )
    
    def ptime_gene_GAM(
        self,
        data: StereoExpData,
        n_jobs: int = -1
    ):
        from multiprocessing import cpu_count
        if n_jobs <= 0:
            n_jobs = cpu_count()
        
        return ptime_gene_GAM(data, core_number=n_jobs)

    def order_trajectory_genes(
        self,
        gam_result: pd.DataFrame,
        min_model_fit: float = 0,
        max_fdr: float = 1,
        cell_number: int = 20
    ):
        lap_neighbor_data = get_lap_neighbor_data(self.stereo_exp_data)
        gam_result_sig = gam_result.loc[(gam_result['fdr'] < max_fdr) & (gam_result['model_fit'] > min_model_fit)]
        return order_trajectory_genes(lap_neighbor_data, gam_result_sig, cell_number)

    def gr_training(
        self,
        tfs_path: str = None,
        ptime_path: str = None,
        min_cells: int = None,
        cell_divide_per_time: int = 80,
        cell_select_per_time: int = 10,
        cell_generate_per_time: int = 500,
        train_ratio: float = 0.8,
        use_gpu: bool = True,
        random_state: int = 0,
        training_times: int = 10,
        iter_times: int = 30,
        mapping_num: int = 3000,
        filename: str = "weights.csv",
        lr_ratio: float = 0.1
    ):
        trainer = Trainer(
            data_type="p_time",
            data=self.stereo_exp_data,
            tfs_path=tfs_path,
            ptime_path=ptime_path,
            min_cells=min_cells,
            cell_divide_per_time=cell_divide_per_time,
            cell_select_per_time=cell_select_per_time,
            cell_generate_per_time=cell_generate_per_time,
            train_ratio=train_ratio,
            use_gpu=use_gpu,
            random_state=random_state
        )
        trainer.run(
            training_times=training_times,
            iter_times=iter_times,
            mapping_num=mapping_num,
            filename=filename,
            lr_ratio=lr_ratio
        )
        return trainer