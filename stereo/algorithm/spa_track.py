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
        """
        Create an object of SpaTrack for single sample.

        :param cluster_res_key: the key of clustering result to be used in cells/obs
        """

        if cluster_res_key not in self.pipeline_res:
            raise KeyError(f'Cannot find clustering result by key {cluster_res_key}')
        if 'spa_track' not in self.pipeline_res:
            self.pipeline_res['spa_track'] = {}
        self.cluster_res_key = cluster_res_key
        self.pipeline_res['spa_track']['cluster_res_key'] = cluster_res_key
        self.plot = PlotSpaTrack(stereo_exp_data=self.stereo_exp_data, pipeline_res=self.pipeline_res)
        return self

    def assess_start_cluster(self):
        """
        Assess the entropy value to identify the starting cluster
        """
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
        """
        Use coordinates or cell type to manually select starting cells.

        :param select_way: Ways to select starting cells.
                            `cell_type`: select by cell type.  
                            `coordinates`: select by coordinates.  
        :param cell_type: Restrict the cell type of starting cells, defaults to None
        :param start_point: The coordinates of the start point in **coordinates** mode, defaults to None
        :param spatial_key: The key to get position information of cells, by default, the `data.spatial_key` will be used.
        :param split: Whether to split the specific type of cells into several small clusters according to cell density, defaults to False
        :param n_clusters: The number of cluster centers after splitting, defaults to 2
        :param n_neigh: The number of neighbors next to the start point/cluster center selected as the starting cell, defaults to 5

        :return: The indices of the starting cells
        """
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

    def auto_estimate_param(
        self,
        spatial_key: Optional[str] = None,
        n_hvg: int = 2000,
    ):
        """
        Automatically estimate the alpha1 and alpha2 for `spt.get_ot_matrix` based on highly variable genes.

        :param spatial_key: The key to get position information of cells, by default, the `data.spatial_key` will be used.
        :param n_hvg: the number of highly variable genes to be calculated, defaults to 2000

        return: The estimated alpha1 and alpha2
        """
        spatial_key = self.stereo_exp_data.spatial_key if spatial_key is None else spatial_key
        return auto_estimate_para(
            self.stereo_exp_data, basis=spatial_key, hvg_gene_number=n_hvg
        )
    
    def calc_alpha_by_moransI(
        self,
        spatial_key: Optional[str] = None
    ):
        """
        Estimate alpah1 and alpah2 for `spt.get_ot_matrix` by using Moran's I.

        :param spatial_key: The key to get position information of cells, by default, the `data.spatial_key` will be used.

        return: The estimated alpha1 and alpha2
        """
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
        """
        Calculate transfer probabilities between cells.

        Using optimal transport theory based on gene expression and/or spatial location information.

        :param data_type: The type of sequencing data.
                            * - `spatial`: for the spatial transcriptome data.
                            * - `single-cell`: for the single-cell sequencing data.
        :param alpha1: The proportion of spatial location information, defaults to 0.5.
        :param alpha2: The proportion of gene expression information, defaults to 0.5.
        :param spatial_key: The key to get position information of cells, by default, the `data.spatial_key` will be used.
        :param n_pcs: _description_, defaults to 50
        :param pca_res_key: The number of used pcs to be calculated, defaults to None

        The transfer matrix stored with the key `trans` in `data.tl.result`.
        """
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
        """
        Get the cell pseudotime based on transition probabilities from initial cells.

        The pseudotime stored with the key `ptime` in `data.cells` or `data.adata.obs`.
        """
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
        """
        Convert cell velocity to grid velocity for streamline display.

        The visualization of vector field borrows idea from scTour: https://github.com/LiQian-XC/sctour/blob/main/sctour.


        :param spatial_key: The key to get position information of cells, by default, the `data.spatial_key` will be used.
        :param grid_num: the number of grids, defaults to 50
        :param smooth: The factor for scale in Gaussian pdf, defaults to 0.5
        :param density: grid density, defaults to 1.0

        :return: A tuple containing the embedding and unitary displacement vectors in grid level.
        """
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
        """
        Get the velocity of each cell and simultaneously convert cell velocity to grid velocity for streamline display.

        The speed can be determined in terms of the cell location and/or gene expression.

        :param spatial_key: The key to get position information of cells, by default, the `data.spatial_key` will be used.
        :param n_neigh_pos: Number of neighbors based on cell positions such as spatial or umap coordinates, defaults to 10
        :param n_neigh_gene: Number of neighbors based on gene expression, defaults to 0
        :param grid_num: the number of grids, defaults to 50
        :param smooth: The factor for scale in Gaussian pdf, defaults to 0.5
        :param density: grid density, defaults to 1.0
        """
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
        """
        Automatically select the start cluster with the largest sum of transfer probability.        

        :param clusters: Give clusters to find, by default, each cluster will be traversed and calculated.

        :return: One cluster with maximum sum of transition probabilities
        """
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
        """
        Select starting cells manually by lasso.

        :param cell_type: Only retain cells of the specified type in lasso selection,
                            by default, all cells will be retained.
        :param spatial_key: The key to get position information of cells, by default, the `data.spatial_key` will be used.
        :param bg_color: The background color of the plot, defaults to '#2F2F4F'
        :param palette: The color palette of the plot, defaults to 'stereo_30'
        :param marker: The dot marker of the plot, defaults to 'o'
        :param marker_size: The size of the dot marker, defaults to 5
        :param width: The width in pixels of the plot, defaults to 500
        :param height: The height in pixels of the plot, defaults to 500
        :param invert_y: Whether to invert the y-axis, defaults to True
        

        """
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
        """
        Learn the function of vector filed.

        :param normalize: Logic flag to determine whether to normalize the data 
                            to have zero means and unit covariance, defaults to False

        """
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
        """
        Set the start and end cells of the least action path (LAP).

        :param start_coordinate: The coordinates of the start cell.
        :param end_coordinate: The coordinates of the end cell.

        """
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
        **kwags
    ):
        """
        Calculate the optimal paths between any two cell states.

        :param n_points: The number of points on the least action path, defaults to 25
        :param n_neighbors: The number of neighbors, defaults to 100

        :return: A trajectory class containing the least action paths information.
        """
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
            vecfld=self.pipeline_res['vector_field'], n_points=n_points, n_neighbors=n_neighbors, **kwags
        )
    
    def map_cell_to_LAP(self, cell_neighbors=150):
        """
        Assign a new pseudotime value to each of these cells based on their position along the LAP.

        :param cell_neighbors: The number of cell neighbors, defaults to 150

        """
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
        """
        Filter genes based on the proportion of cells expressing the gene and the highly variable genes.

        Only running on the subset of cells that are the neighbors of the LAP.

        :param min_exp_prop_in_genes: The minimum proportion of cells expressing the gene, defaults to 0.1
        :param n_hvg: The number of highly variable genes to be calculated, defaults to 2000
        :param focused_cell_types: The cell types to be focused on, by default, all cell types will be used.

        :return: the object of data with filtered genes
        """
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
        """
        Fit GAM model by formula gene_exp ~ Ptime.

        Call GAM_gene_fit() by multi-process computing to improve operational speed.

        :param data: the object of data after `spt.filter_genes`.
        :param n_jobs: The number of cores used for computing, by default, -1 means using all cores.
        :return: A DataFrame containing the columns as follows:
                    - pvalue: calculated from GAM
                    - model_fit: a goodness-of-fit measure. larger value means better fit
                    - pattern: increase or decrease. drection of gene expression changes across time
                    - fdr: BH fdr
        """
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
        """
        Split cells sorted by ptime into widonws.

        Order genes according number id of the maximum expression window.

        Only running on the subset of cells that are the neighbors of the LAP.

        :param gam_result: the result of `spt.ptime_gene_GAM`.
        :param min_model_fit: the minimum model fit for filtering the gam result, defaults to 0
        :param max_fdr: the maximum fdr for filtering the gam result, defaults to 1
        :param cell_number: Cell number within splited window, defaults to 20

        :return: A DataFrame whose columns are sortted significant genes expression matrix
                    according to mean expression value in windows and index is cells.
        """
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
        """
        Create and run a trainer for gene regulatory network training with a single sample contating pesudotime.

        :param tfs_path: The path of the tf names file, defaults to None
        :param ptime_path: The path of the ptime file, used to determine 
                            the sequence of the ptime data, defaults to None
        :param min_cells: The minimum number of cells for gene filtration, defaults to None
        :param cell_divide_per_time: The cell number generated at each time point 
                                        using the meta-analysis method, defaults to 80
        :param cell_select_per_time: The number of randomly selected cells at each time point, defaults to 10
        :param cell_generate_per_time: The number of cells generated at each time point, defaults to 500
        :param train_ratio:  Ratio of training data, defaults to 0.8
        :param use_gpu: Whether to use gpu, by default, to use if available.
        :param random_state: Random seed of numpy and torch, fixed for reproducibility, defaults to 0
        :param training_times: Number of times to randomly initialize the model and retrain, defaults to 10
        :param iter_times: The number of iterations for each training model, defaults to 30
        :param mapping_num: The number of top weight pairs you want to extract, defaults to 3000
        :param filename: The name of the file to save the weights, defaults to "weights.csv"
        :param lr_ratio: The learning rate, defaults to 0.1


        :return: A trainer object for gene regulatory network training.
        """
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