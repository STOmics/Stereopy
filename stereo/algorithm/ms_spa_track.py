from typing import Tuple, Optional, Literal, Union, List
from copy import deepcopy
import numpy as np
import pandas as pd

from stereo.core.stereo_exp_data import StereoExpData
from stereo.algorithm.ms_algorithm_base import MSDataAlgorithmBase
from stereo.algorithm.spt.multiple_time import (
    transfer_matrix,
    generate_animate_input,
    map_data
)
from stereo.algorithm.spt.single_time import Trainer
# from stereo.algorithm.spt.plot import PlotSpaTrack
from stereo.plots.plot_spa_track import PlotSpaTrack


class MSSpaTrack(MSDataAlgorithmBase):
    def main(
        self,
        cluster_res_key: str = 'cluster'
    ):
        # if cluster_res_key not in self.pipeline_res:
        #     raise KeyError(f'Cannot find clustering result by key {cluster_res_key}')
        if 'spa_track' not in self.pipeline_res:
            self.pipeline_res['spa_track'] = {}
        self.cluster_res_key = cluster_res_key
        self.pipeline_res['spa_track']['cluster_res_key'] = cluster_res_key
        self.plot = PlotSpaTrack(ms_data=self.ms_data, pipeline_res=self.pipeline_res)
        return self
    
    def transfer_matrix(
        self,
        data_indices: List[Union[str, int]] = None,
        layer: str = None,
        spatial_key: str = 'spatial',
        alpha: float = 0.1, 
        epsilon = 0.01,
        rho = np.inf,
        G_1 = None,
        G_2 = None,
        **kwargs
    ):
        assert spatial_key is not None, 'spatial_key must be provided'
        if data_indices is None:
            data_indices = self.ms_data.names
        data_list_to_calculate: List[StereoExpData] = [
            self.ms_data[di] for di in data_indices
        ]
        data_names = [
            self.ms_data.names[di] if isinstance(di, int) else di for di in data_indices
        ]
        sp_existed = [spatial_key in data.cells_matrix for data in data_list_to_calculate]
        assert all(sp_existed), 'spatial_key must be existed in all data'

        transfer_matrices = {}
        for i in range(len(data_indices) - 1):
            data1 = data_list_to_calculate[i]
            data2 = data_list_to_calculate[i + 1]
            transfer_matrices[(data_names[i], data_names[i + 1])] = transfer_matrix(
                data1, data2, layer=layer, spatial_key=spatial_key, alpha=alpha, epsilon=epsilon,
                rho=rho, G_1=G_1, G_2=G_2, **kwargs
            )
        self.pipeline_res['spa_track']['transfer_spatial_key'] = spatial_key
        self.pipeline_res['spa_track']['transfer_matrices'] = transfer_matrices
    
    def generate_animate_input(
        self,
        data_indices: List[Union[str, int]] = None,
        time_key: str = 'batch'
    ):
        data_names = [
            self.ms_data.names[di] if isinstance(di, int) else di for di in data_indices
        ]
        data_list = [
            self.ms_data[data_name] for data_name in data_names
        ]
        pi_list = []
        for i in range(len(data_indices) - 1):
            pi_list.append(self.pipeline_res['spa_track']['transfer_matrices'][(data_names[i], data_names[i + 1])])
        spatial_key = self.pipeline_res['spa_track']['transfer_spatial_key']
        self.pipeline_res['spa_track']['transfer_data'] = generate_animate_input(
            pi_list, data_list, spatial_key=spatial_key, time=time_key, annotation=self.pipeline_res['spa_track']['cluster_res_key']
        )
    
    def map_data(
        self,
        data1_index: Union[str, int],
        data2_index: Union[str, int]
    ):
        data1 = self.ms_data[data1_index]
        data2 = self.ms_data[data2_index]
        data1_name = self.ms_data.names[data1_index] if isinstance(data1_index, int) else data1_index
        data2_name = self.ms_data.names[data2_index] if isinstance(data2_index, int) else data2_index
        transfer_matrix = self.pipeline_res['spa_track']['transfer_matrices'][(data1_name, data2_name)]
        pi = map_data(transfer_matrix, data1, data2)
        if 'mapped_data' not in self.pipeline_res['spa_track']:
            self.pipeline_res['spa_track']['mapped_data'] = {}
        self.pipeline_res['spa_track']['mapped_data'][(data1_name, data2_name)] = pi
        return pi
    
    def gr_training(
        self,
        data1_index: Union[str, int],
        data2_index: Union[str, int],
        tfs_path: str = None,
        min_cells_1: int = None,
        min_cells_2: int = None,
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
        data_list = [deepcopy(self.ms_data[data1_index]), deepcopy(self.ms_data[data2_index])]
        min_cells = [min_cells_1, min_cells_2]
        cell_mapping = self.map_data(data1_index, data2_index)

        trainer = Trainer(
            data_type="2_time",
            data=data_list,
            tfs_path=tfs_path,
            cell_mapping=cell_mapping,
            min_cells=min_cells,
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