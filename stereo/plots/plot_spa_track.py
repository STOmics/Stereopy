from typing import Union, Optional, List, Mapping
import matplotlib.pylab as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from stereo.constant import PLOT_SCATTER_SIZE_FACTOR
from stereo.core.stereo_exp_data import StereoExpData
from stereo.core.ms_data import MSData
from stereo.plots.scatter import base_scatter
from stereo.plots.decorator import download
from stereo.stereo_config import stereo_conf

from stereo.algorithm.spt.utils import get_cell_coordinates, get_lap_neighbor_data
from stereo.algorithm.spt.single_time import (
    assess_start_cluster_plot,
    plot_least_action_path,
    plot_trajectory_gene_heatmap,
    plot_trajectory_gene,
    filter_gene
)

from stereo.algorithm.spt.multiple_time import (
    animate_transfer,
    visual_3D_mapping_3,
    visual_3D_mapping_2
)


class PlotSpaTrack:

    def __init__(
        self,
        stereo_exp_data: StereoExpData = None,
        ms_data: MSData = None,
        pipeline_res: Mapping = None
    ):
        self.stereo_exp_data = stereo_exp_data
        self.ms_data = ms_data
        self.pipeline_res = pipeline_res

    @download
    def quiver(
        self,
        palette: str = 'stereo_30',
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        dot_size: int = None,
        marker: str = 'o',
        invert_y: bool = True,
        width: int = None,
        height: int = None,
        alpha: float = 0.4,
        quiver_scale: float = 0.008,
        quiver_kwargs: dict = {},
        **kwargs
    ):
        cluster_res_key = self.pipeline_res['spa_track']['cluster_res_key']
        if cluster_res_key not in self.pipeline_res:
            raise KeyError(f'Cannot find clustering result by key {cluster_res_key}')
        group_list = self.pipeline_res[cluster_res_key]['group'].to_numpy(copy=True)
        hue_order = self.pipeline_res[cluster_res_key]['group'].cat.categories
        spatial_key = self.pipeline_res['spa_track']['velocoty_spatial_key']
        position = get_cell_coordinates(self.stereo_exp_data, basis=spatial_key)
        fig: Figure = base_scatter(
            position[:, 0],
            position[:, 1],
            hue=group_list,
            palette=palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            dot_size=dot_size,
            marker=marker,
            invert_y=invert_y,
            hue_order=hue_order,
            width=width,
            height=height,
            foreground_alpha=alpha,
            **kwargs
        )
        ax: Axes = plt.gca()

        ax.quiver(
            self.pipeline_res['P_grid'][0], self.pipeline_res['P_grid'][1],
            self.pipeline_res['V_grid'][0], self.pipeline_res['V_grid'][1],
            color='black', scale=quiver_scale, **quiver_kwargs
        )
        return fig

    
    def __stremplot(
        self,
        palette: str = 'stereo_30',
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        dot_size: int = None,
        marker: str = 'o',
        invert_y: bool = True,
        width: int = None,
        height: int = None,
        alpha: float = 0.4,
        stream_lines_density: float = 1.8,
        stream_lines_width: float = 2.5,
        stream_arrows_size: float = 1.5,
        streamplot_kwargs: dict = {},
        **kwargs
    ):
        cluster_res_key = self.pipeline_res['spa_track']['cluster_res_key']
        if cluster_res_key not in self.pipeline_res:
            raise KeyError(f'Cannot find clustering result by key {cluster_res_key}')
        group_list = self.pipeline_res[cluster_res_key]['group'].to_numpy(copy=True)
        hue_order = self.pipeline_res[cluster_res_key]['group'].cat.categories
        spatial_key = self.pipeline_res['spa_track']['velocoty_spatial_key']
        position = get_cell_coordinates(self.stereo_exp_data, basis=spatial_key)
        fig: Figure = base_scatter(
            position[:, 0],
            position[:, 1],
            hue=group_list,
            palette=palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            dot_size=dot_size,
            marker=marker,
            invert_y=invert_y,
            hue_order=hue_order,
            width=width,
            height=height,
            foreground_alpha=alpha,
            **kwargs
        )
        ax: Axes = plt.gca()

        ax.streamplot(
            self.pipeline_res['P_grid'][0], self.pipeline_res['P_grid'][1],
            self.pipeline_res['V_grid'][0], self.pipeline_res['V_grid'][1],
            density=stream_lines_density, linewidth=stream_lines_width, arrowsize=stream_arrows_size,
            color='black', **streamplot_kwargs
        )
        return fig
    
    @download
    def stremplot(
        self,
        palette: str = 'stereo_30',
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        dot_size: int = None,
        marker: str = 'o',
        invert_y: bool = True,
        width: int = None,
        height: int = None,
        alpha: float = 0.4,
        stream_lines_density: float = 1.8,
        stream_lines_width: float = 2.5,
        stream_arrows_size: float = 1.5,
        streamplot_kwargs: dict = {},
        **kwargs
    ):
        return self.__stremplot(
            palette=palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            dot_size=dot_size,
            marker=marker,
            invert_y=invert_y,
            width=width,
            height=height,
            alpha=alpha,
            stream_lines_density=stream_lines_density,
            stream_lines_width=stream_lines_width,
            stream_arrows_size=stream_arrows_size,
            streamplot_kwargs=streamplot_kwargs,
            **kwargs
        )
    
    @download
    def lap_endpoints(
        self,
        x_label: str = None,
        y_label: str = None,
        dot_size: int = None,
        marker: str = 'o',
        invert_y: bool = True,
        width: int = None,
        height: int = None,
        **kwargs
    ):
        if 'lap_endpoints' not in self.stereo_exp_data.cells:
            raise KeyError('Cannot find LAP endpoints in stereo_exp_data.cells')
        
        lap_endpoints = self.stereo_exp_data.cells['lap_endpoints'].to_numpy()

        endpoints_index = (lap_endpoints == 'start') | (lap_endpoints == 'end')

        position = np.concatenate(
            [
                self.stereo_exp_data.position[~endpoints_index],
                self.stereo_exp_data.position[endpoints_index]
            ]
        )

        lap_endpoints = np.concatenate(
            [lap_endpoints[~endpoints_index], lap_endpoints[endpoints_index]]
        )

        fig: Figure = base_scatter(
            position[:, 0],
            position[:, 1],
            hue=lap_endpoints,
            palette=['red', 'blue', '#828282'],
            title='LAP endpoints',
            x_label=x_label,
            y_label=y_label,
            dot_size=dot_size,
            marker=marker,
            invert_y=invert_y,
            hue_order=['start', 'end', 'others'],
            width=width,
            height=height,
            **kwargs
        )
        
        return fig
    
    @download
    def least_action_path(
        self,
        palette: str = 'stereo_30',
        title: str = None,
        x_label: str = None,
        y_label: str = None,
        dot_size: int = None,
        marker: str = 'o',
        invert_y: bool = True,
        width: int = None,
        height: int = None,
        alpha: float = 0.4,
        stream_lines_density: float = 1.8,
        stream_lines_width: float = 2.5,
        stream_arrows_size: float = 1.5,
        streamplot_kwargs: dict = {},
        neighbors_palette: str = 'YlGnBu',
        **kwargs
    ):
        dot_size = PLOT_SCATTER_SIZE_FACTOR / self.stereo_exp_data.n_cells if dot_size is None else dot_size
        fig: Figure = self.__stremplot(
            palette=palette,
            title=title,
            x_label=x_label,
            y_label=y_label,
            dot_size=dot_size,
            marker=marker,
            invert_y=False,
            width=width,
            height=height,
            alpha=alpha,
            stream_lines_density=stream_lines_density,
            stream_lines_width=stream_lines_width,
            stream_arrows_size=stream_arrows_size,
            streamplot_kwargs=streamplot_kwargs,
            show_legend=False,
            **kwargs
        )
        ax: Axes = plt.gca()
        spatial_key = self.pipeline_res['spa_track']['velocoty_spatial_key']
        plot_least_action_path(self.stereo_exp_data, basis=spatial_key, ax=ax, point_size=dot_size+50, linewidth=(dot_size+50)/25)

        is_lap_neighbor = self.stereo_exp_data.cells['is_lap_neighbor'].to_numpy()
        base_scatter(
            x=self.stereo_exp_data.position[is_lap_neighbor, 0],
            y=self.stereo_exp_data.position[is_lap_neighbor, 1],
            hue=self.stereo_exp_data.cells['ptime'][is_lap_neighbor],
            x_label=x_label,
            y_label=y_label,
            title=title,
            dot_size=dot_size,
            palette=neighbors_palette,
            color_bar=True,
            marker=marker,
            width=width,
            height=height,
            ax=ax,
            invert_y=False,
            **kwargs
        )
        if invert_y:
            ax.invert_yaxis()
        return fig
    
    @download
    def trajectory_gene_heatmap(
        self,
        sig_gene_exp_order: pd.DataFrame,
        smooth_length:int,
        palette: str ="twilight_shifted",
        gene_label_size:int =30,
        width=8,
        height=10
    ):
        return plot_trajectory_gene_heatmap(
            sig_gene_exp_order,
            smooth_length,
            cmap_name=palette,
            gene_label_size=gene_label_size,
            fig_width=width,
            fig_height=height
        )
    

    @download
    def trajectory_gene(
        self,
        gene_name: str,
        line_width: int = 5,
        show_cell_type: bool = False,
        dot_size=20
    ) -> Axes:
        cluster_res_key = self.pipeline_res['spa_track']['cluster_res_key']
        if 'filter_genes' in self.pipeline_res['spa_track']:
            focused_cell_types = self.pipeline_res['spa_track']['filter_genes']['focused_cell_types']
            lap_neighbor_data = get_lap_neighbor_data(self.stereo_exp_data, focused_cell_types)
            min_exp_prop_in_genes = self.pipeline_res['spa_track']['filter_genes']['min_exp_prop_in_genes']
            n_hvg = self.pipeline_res['spa_track']['filter_genes']['n_hvg']
            data = filter_gene(
                lap_neighbor_data, use_col=cluster_res_key, min_exp_prop=min_exp_prop_in_genes, hvg_gene=n_hvg
            )
        else:
            data = get_lap_neighbor_data(self.stereo_exp_data)
        return plot_trajectory_gene(
            data,
            cluster_res_key,
            gene_name,
            line_width,
            show_cell_type,
            dot_size
        )
    
    @download
    def assess_start_cluster_plot(
        self,
        palette: str = 'stereo_30',
        width: int = 10,
        height: int = 9
    ):
        return assess_start_cluster_plot(
            self.stereo_exp_data,
            use_col=self.pipeline_res['spa_track']['cluster_res_key'],
            palette=palette,
            width=width, height=height
        )
    
    def animate_transfer(
        self,
        data_indices: List[Union[str, int]] = None,
        title: str = None,
        save_path: str = None,
        palette: Union[str, List[str]] = 'tab20b',
        time_key: str = 'time',
        N: int = 2,
        n: int = 6, 
    ):
        data_names = [
            self.ms_data.names[di] if isinstance(di, int) else di for di in data_indices
        ]
        data_list = [
            self.ms_data[data_name] for data_name in data_names
        ]
        spatial_key = self.pipeline_res['spa_track']['transfer_spatial_key']
        if title is None:
            title = f'Transfer data: {", ".join(data_names[0:-1])} to {data_names[-1]}'

        cluster_res_key = self.pipeline_res['spa_track']['cluster_res_key']
        clusters = []
        for data in data_list:
            clusters = np.intersect1d(clusters, data.cells.obs[cluster_res_key].unique())
        color = stereo_conf.get_colors(palette, n=len(clusters))
        
        return animate_transfer(
            data_list=data_list,
            transfer_data=self.pipeline_res['spa_track']['transfer_data'],
            fig_title=title,
            save_path=save_path,
            spatial_key=spatial_key,
            color=color,
            time=time_key,
            annotation=self.pipeline_res['spa_track']['cluster_res_key'],
            times=data_names,
            N=N,
            n=n
        )

    @download
    def visual_3D_mapping_3(
        self,
        data_indices: List[Union[str, int]] = None,
        dot_size: float = 0.8,
        line_width: float = 0.03,
        line_alpha: float = 0.8,
        palette: Union[str, List[str]]='tab20b',
        width: Optional[int] = None,
        height: Optional[int] = None,
        axis=None,
    ):
        data_names = [
            self.ms_data.names[di] if isinstance(di, int) else di for di in data_indices
        ]
        obs_list = [
            self.ms_data[data_name].cells.obs[['cell_id','annotation','x','y']].copy() \
                for data_name in data_names
        ]
        slice_batches = []
        clusters = []
        # cluster_res_key = self.pipeline_res['spa_track']['cluster_res_key']
        for i, obs in enumerate(obs_list):
            obs['batch'] = f'slice{i+1}'
            clusters = np.union1d(clusters, obs['annotation'].unique())
            slice_batches.append(f'slice{i+1}')
        clusters = np.unique(clusters)
        obs_concat = pd.concat(obs_list)
        df_mapping_cell=self.pipeline_res['spa_track']['transfer_data'][slice_batches]
        colors = stereo_conf.get_colors(palette, n=len(clusters))
        fig = visual_3D_mapping_3(
            df_concat=obs_concat,
            df_mapping_res=df_mapping_cell,
            point_size=dot_size,
            line_width=line_width,
            line_alpha=line_alpha,
            view_axis=axis,
            cell_color_list=colors
        )
        if width is not None:
            fig.set_figwidth(width)
        if height is not None:
            fig.set_figheight(height)
        return fig

    @download
    def visual_3D_mapping_2(
        self,
        data1_index,
        data2_index,
        dot_size: float = 0.8,
        line_width: float = 0.03,
        line_alpha: float = 0.8,
        palette: Union[str, List[str]]='tab20b',
        width: Optional[int] = None,
        height: Optional[int] = None,
        axis=None,
    ):
        data1_name = self.ms_data.names[data1_index] if isinstance(data1_index, int) else data1_index
        data2_name = self.ms_data.names[data2_index] if isinstance(data2_index, int) else data2_index
        obs_list = [
            self.ms_data[data_name].cells.obs[['cell_id','annotation','x','y']].copy() \
                for data_name in (data1_name, data2_name)
        ]
        slice_batches = []
        clusters = []
        for i, obs in enumerate(obs_list):
            obs['batch'] = f'slice{i+1}'
            clusters = np.union1d(clusters, obs['annotation'].unique())
            slice_batches.append(f'slice{i+1}')
        clusters = np.unique(clusters)
        obs_concat = pd.concat(obs_list)
        colors = stereo_conf.get_colors(palette, n=len(clusters))

        df_mapping = self.pipeline_res['spa_track']['mapped_data'][(data1_name, data2_name)][['slice1','slice2']]

        fig = visual_3D_mapping_2(
            df_concat=obs_concat,
            df_mapping=df_mapping,
            point_size=dot_size,
            line_width=line_width,
            line_alpha=line_alpha,
            view_axis=axis,
            cell_color_list=colors
        )
        if width is not None:
            fig.set_figwidth(width)
        if height is not None:
            fig.set_figheight(height)
        return fig
