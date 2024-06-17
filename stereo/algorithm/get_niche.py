import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import cdist

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.preprocess.filter import filter_by_clusters
from stereo.preprocess.filter import filter_cells
from stereo.algorithm.cell_cell_communication.exceptions import InvalidNicheMethod
from stereo.core.stereo_exp_data import AnnBasedStereoExpData


class GetNiche(AlgorithmBase):
    def main(
            self,
            niche_distance: float,
            cluster_1: str,
            cluster_2: str,
            cluster_res_key: str = None,
            method: str = 'fixed',
            theta: float = 0.1,
            filter_raw: bool = True,
            inplace: bool = False
    ):
        """
        To ensure the accuracy and specificity of this juxtacrine signaling model,
        we extract cells bordering their niches and statistically calculate their CCC activity scores of L-R pairs
        under the assumption that intercellular L-R communications routinely exist among closely neighboring cells

        :param niche_distance: the maximum distance between cells in order to form a niche.
        :param cluster_1: one cell cluster in the interaction.
        :param cluster_2: the other cell cluster in the interaction.
        :param cluster_res_key: the key which specifies the clustering result in data.tl.result.
        :param method: method for calculating niche, choose from 'fixed' or 'adaptive'.
        :param theta: the parameter used to control border region selection, only available for 'adaptive' method.
        :param filter_raw: this function will create a new data object by filtering cells,
                            this parameter determine whether to filter raw data meanwhile, default to True.
        :param inplace: whether to replace the previous express matrix or get a new StereoExpData object with the new express matrix, default by False. # noqa

        return: a new StereoExpData or AnnBasedStereoExpData object representing a niche.
        """
        assert cluster_1 != cluster_2, "cluster_1 can not equal to cluster_2."

        data_full = self.stereo_exp_data
        cluster = self.pipeline_res[cluster_res_key]
        data_1, _ = filter_by_clusters(data_full, cluster_res=cluster, groups=cluster_1, inplace=False)
        data_2, _ = filter_by_clusters(data_full, cluster_res=cluster, groups=cluster_2, inplace=False)

        coord_1 = data_1.position.astype(np.int64)
        coord_2 = data_2.position.astype(np.int64)
        if data_1.position_z is not None:
            coord_1 = np.concatenate([coord_1, data_1.position_z.astype(np.int64)], axis=1)
        if data_2.position_z is not None:
            coord_2 = np.concatenate([coord_2, data_2.position_z.astype(np.int64)], axis=1)

        if method == 'fixed':
            dist_matrix = cdist(coord_1, coord_2)
            dist_df = pd.DataFrame(dist_matrix, index=data_1.cell_names, columns=data_2.cell_names)

            result_target_sender = dist_df.where(dist_df < niche_distance, other=np.nan)
            result_target_sender = result_target_sender.dropna(how='all', axis=1).dropna(how='all', axis=0)

            cell_list = list(result_target_sender.index) + list(result_target_sender.columns)
            # data_result = filter_cells(data_full, cell_list=cell_list, inplace=inplace)

        elif method == 'adaptive':
            coord_12: np.ndarray = np.concatenate([coord_1, coord_2], axis=0)
            coord_all = data_full.position.astype(np.int64)
            if data_full.position_z is not None:
                coord_all = np.concatenate([coord_all, data_full.position_z.astype(np.int64)], axis=1)

            n1 = coord_1.shape[0]  # number of cells in cluster_1
            n12 = coord_12.shape[0]  # number of cells in cluster_1 + cluster_2
            
            # a matrix indicating the points falling inside the neighboring cubic for each cell of cluster_1
            neighbors = np.zeros((n1, n12), dtype=int)
            shift = np.zeros(n1, dtype=np.float64)

            cluster_label = cluster['group']
            info_entropy = np.zeros(n1, dtype=np.float64)
            
            for i in range(n1):
                """
                adaptive step 1: get the shift of the centroid for each cell in cluster_1, 
                considering only cluster_1 and cluster_2
                """
                dist = np.abs(coord_1[i] - coord_12)
                flag = np.all(dist <= niche_distance, axis=1)
                neighbors[i, flag] = 1
                neighbors[i, i] = 0  # exclude the cell itself
                n_neighbors = np.sum(neighbors[i])
                n_neighbors_2 = np.sum(neighbors[i, n1:])
                shift[i] = n_neighbors_2 / n_neighbors if n_neighbors != 0 else 0

                """
                adaptive step 2: calculate local information entropy for each cell in cluster_1, 
                considering all cell types
                """
                dist = np.abs(coord_1[i] - coord_all)
                flag = np.all(dist <= niche_distance, axis=1)
                neighbor_cluster = cluster_label[flag]
                _, encoded_neighbor_cluster = np.unique(neighbor_cluster, return_inverse=True)
                entropy_value = entropy(encoded_neighbor_cluster, base=2)
                info_entropy[i] = entropy_value

            """
            adaptive step 3: select cells belonging to the border region, out of cluster_1
            """
            border_index = shift > (theta * info_entropy)
            cell_name_border = data_1.cell_names[border_index]
            """
            adaptive step 4: construct the niche for cluster_1 and cluster_2
            """
            neighbor_border = neighbors[border_index, n1:]  # filter border rows and columns of cluster_2
            neighbor_index = np.any(neighbor_border, axis=0)
            cell_name_neighbor = data_2.cell_names[neighbor_index]
            cell_list = list(cell_name_border) + list(cell_name_neighbor)
        else:
            raise InvalidNicheMethod(method)
        
        data_result = filter_cells(data_full, cell_list=cell_list, inplace=inplace)
        if filter_raw and data_result.raw is not None:
            filter_cells(data_result.raw, cell_list=cell_list, inplace=True)
            if isinstance(data_result, AnnBasedStereoExpData):
                data_result.adata.raw = data_result.raw.adata


        return data_result
