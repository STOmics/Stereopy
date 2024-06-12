import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import cdist

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.preprocess.filter import filter_by_clusters
from stereo.preprocess.filter import filter_cells
from stereo.algorithm.cell_cell_communication.exceptions import InvalidNicheMethod


class GetNiche(AlgorithmBase):
    def main(
            self,
            niche_distance: float,
            cluster_1: str,
            cluster_2: str,
            cluster_res_key: str = None,
            inplace: bool = False,
            method: str = 'fixed',
            theta: float = 0.5
    ):
        """
        To ensure the accuracy and specificity of this juxtacrine signaling model,
        we extract cells bordering their niches and statistically calculate their CCC activity scores of L-R pairs
        under the assumption that intercellular L-R communications routinely exist among closely neighboring cells

        :param niche_distance: the maximum distance between cells in order to form a niche.
        :param cluster_1: one cell cluster in the interaction.
        :param cluster_2: the other cell cluster in the interaction.
        :param cluster_res_key: the key which specifies the clustering result in data.tl.result.
        :param inplace: whether to replace the previous express matrix or get a new StereoExpData object with the new express matrix, default by False. # noqa
        :param method: method for calculating niche, choose from 'fixed' or 'adaptive'.
        :param theta: the parameter used to control border region selection.
        """
        assert cluster_1 != cluster_2, "cluster_1 can not equal to cluster_2."

        data_full = self.stereo_exp_data
        cluster = self.pipeline_res[cluster_res_key]
        data_1, _ = filter_by_clusters(data_full, cluster_res=cluster, groups=cluster_1, inplace=False)
        data_2, _ = filter_by_clusters(data_full, cluster_res=cluster, groups=cluster_2, inplace=False)

        coord_1 = data_1.position
        coord_2 = data_2.position
        if data_1.position_z is not None:
            coord_1 = np.concatenate([coord_1, data_1.position_z], axis=1)
        if data_2.position_z is not None:
            coord_2 = np.concatenate([coord_2, data_2.position_z], axis=1)

        if method == 'fixed':
            dist_matrix = cdist(coord_1, coord_2)
            dist_df = pd.DataFrame(dist_matrix, index=data_1.cell_names, columns=data_2.cell_names)

            result_target_sender = dist_df.where(dist_df < niche_distance, other=np.nan)
            result_target_sender = result_target_sender.dropna(how='all', axis=1).dropna(how='all', axis=0)

            cell_list = list(result_target_sender.index) + list(result_target_sender.columns)
            data_result = filter_cells(data_full, cell_list=cell_list, inplace=inplace)

        elif method == 'adaptive':
            coord_12 = np.concatenate((coord_1, coord_2), axis=0)
            coord_all = data_full.position
            if data_full.position_z is not None:
                coord_all = np.concatenate(coord_all, coord_all.position_z, axis=1)

            n1 = coord_1.shape[0]  # number of cells in cluster_1
            n12 = coord_12.shape[0]  # number of cells in cluster_1 + cluster_2
            n = coord_all.shape[0]  # number of all cells
            """
            adaptive step 1: get the shift of the centroid for each cell in cluster_1, 
            considering only cluster_1 and cluster_2
            """
            # a matrix indicating the points falling inside the neighboring cubic for each cell of cluster_1
            neighbors = np.zeros((n1, n12), dtype=int)
            for i in range(n1):
                for j in range(n12):
                    dist_x = abs(coord_1[i, 0] - coord_12[i, 0])
                    dist_y = abs(coord_1[i, 1] - coord_12[i, 1])
                    if data_full.position_z is not None:
                        dist_z = abs(coord_1[i, 2] - coord_12[i, 2])
                    else:
                        dist_z = 0
                    # if the distance in all three dimensions are less than given distance (in a cubic), append 1
                    if dist_x <= niche_distance and dist_y <= niche_distance and dist_z <= niche_distance and i != j:
                        neighbors[i, j] = 1
            # calculate centroid shift for each cell in cluster_1, equalling proportion of cluster_2 in the cubic
            n_neighbor = np.sum(neighbors, axis=1)
            n_neighbor_1 = np.sum(neighbors[:, range(n1)], axis=1)
            shift = 1 - np.divide(n_neighbor_1, n_neighbor, out=np.zeros_like(n_neighbor, dtype=float),
                                  where=(n_neighbor != 0))  # force to 0 if no neighbors
            """
            adaptive step 2: calculate local information entropy for each cell in cluster_1, 
            considering all cell types
            """
            # calculate information entropy for each cell in cluster_1
            cluster_label = data_full.cells[cluster_res_key]

            info_entropy = np.array([])
            for i in range(n1):
                neighbor_index = np.array([])
                for j in range(n):
                    dist_x = abs(coord_1[i, 0] - coord_12[i, 0])
                    dist_y = abs(coord_1[i, 1] - coord_12[i, 1])
                    if data_full.position_z is not None:
                        dist_z = abs(coord_1[i, 2] - coord_12[i, 2])
                    else:
                        dist_z = 0
                    # if the distance in all three dimensions are less than given distance (in a cubic), append 1
                    if dist_x <= niche_distance and dist_y <= niche_distance and dist_z <= niche_distance:
                        neighbor_index = np.append(neighbor_index, j)
                neighbor_cluster = cluster_label[neighbor_index]
                _, encoded_neighbor_cluster = np.unique(neighbor_cluster, return_inverse=True)
                entropy_value = entropy(encoded_neighbor_cluster, base=2)
                info_entropy = np.append(info_entropy, entropy_value)
            """
            adaptive step 3: select cells belonging to the border region, out of cluster_1
            """
            border_index = np.where(shift > theta * info_entropy)[0]
            cell_name_border = data_1.cell_names[border_index]
            """
            adaptive step 4: construct the niche for cluster_1 and cluster_2
            """
            neighbor_border = neighbors[border_index, n1:]  # filter border rows and columns of cluster_2
            neighbor_index = np.where(np.any(neighbor_border, axis=0))[0]  # determine cluster_2 neighbors
            cell_name_neighbor = data_2.cell_names[neighbor_index]
            cell_list = list(cell_name_border) + list(cell_name_neighbor)
            data_result = filter_cells(data_full, cell_list=cell_list, inplace=False)
        else:
            raise InvalidNicheMethod(method)

        for res_key in (data_result.tl.key_record['pca'] + data_result.tl.key_record['umap']):
            if data_result.tl.result[res_key].shape[0] == data_result.shape[0]:
                continue
            flag = np.isin(data_full.cell_names, cell_list)
            res_filtered = data_result.tl.result[res_key][flag]
            res_filtered.reset_index(drop=True, inplace=True)
            data_result.tl.result[res_key] = res_filtered
        
        for column in data_result.cells.obs.columns:
            if data_result.cells.obs[column].dtype.name == 'category':
                data_result.cells.obs[column] = data_result.cells.obs[column].cat.remove_unused_categories()

        return data_result
