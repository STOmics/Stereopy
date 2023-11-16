import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.preprocess.filter import filter_by_clusters
from stereo.preprocess.filter import filter_cells


class GetNiche(AlgorithmBase):
    def main(
            self,
            niche_distance: float,
            cluster_1: str,
            cluster_2: str,
            cluster_res_key: str = None,
            inplace: bool = False
    ):
        """
        To ensure the accuracy and specificity of this juxtacrine signaling model,
        we extract cells bordering their niches and statistically calculate their CCC activity scores of L-R pairs
        under the assumption that intercellular L-R communications routinely exist among closely neighboring cells

        :param niche_distance: the maximum distance between cells in order to form a niche.
        :param cluster_1: one cell cluster in the interaction.
        :param cluster_2: the other cell cluster in the interaction.
        :param coord_key: the key which specifies the coordiate of cells.
        :param cluster_res_key: the key which specifies the clustering result in data.tl.result.
        :param inplace: whether to inplace the previous express matrix or get a new StereoExpData object with the new express matrix, default by False. # noqa
        """
        assert cluster_1 != cluster_2, "cluster_1 can not equal to cluster_2."

        data_full = self.stereo_exp_data
        # adata_full = self.stereo_exp_data._ann_data
        cluster = self.pipeline_res[cluster_res_key]
        data_1, _ = filter_by_clusters(data_full, cluster_res=cluster, groups=cluster_1, inplace=False)
        data_2, _ = filter_by_clusters(data_full, cluster_res=cluster, groups=cluster_2, inplace=False)
        # adata_1 = adata_full[adata_full.obs['celltype'] == cluster_1, :]
        # adata_2 = adata_full[adata_full.obs['celltype'] == cluster_2, :]

        coord_1 = data_1.position
        coord_2 = data_2.position
        if data_1.position_z is not None:
            coord_1 = np.concatenate([coord_1, data_1.position_z], axis=1)
        if data_2.position_z is not None:
            coord_2 = np.concatenate([coord_2, data_2.position_z], axis=1)
        dist_matrix = cdist(coord_1, coord_2)
        dist_df = pd.DataFrame(dist_matrix, index=data_1.cell_names, columns=data_2.cell_names)

        result_target_sender = dist_df.where(dist_df < niche_distance, other=np.nan)
        result_target_sender = result_target_sender.dropna(how='all', axis=1).dropna(how='all', axis=0)

        # adata_result = adata_full[(list(result_target_sender.index) + list(result_target_sender.columns)), :]
        cell_list = list(result_target_sender.index) + list(result_target_sender.columns)
        data_result = filter_cells(data_full, cell_list=cell_list, inplace=inplace)
        if not inplace:
            data_result.tl.result.set_item_callback = None

        for res_key in data_result.tl.key_record['pca']:
            if data_result.tl.result[res_key].shape[0] == data_result.shape[0]:
                continue
            data_result.tl.result[res_key] = data_result.tl.result[res_key][
                np.isin(data_full.cell_names, cell_list)].copy()
        if cluster_res_key in data_result.cells:
            data_result.cells[cluster_res_key] = pd.Series(
                data_result.cells[cluster_res_key].to_numpy(),
                index=data_result.cell_names,
                dtype='category'
            )
        else:
            data_result.tl.result[cluster_res_key]['group'] = pd.Series(
                data_result.tl.result[cluster_res_key]['group'].to_numpy(),
                index=data_result.cell_names,
                dtype='category'
            )

        if not inplace:
            data_result.tl.result.contain_method = None
            data_result.tl.result.get_item_method = None
        return data_result
