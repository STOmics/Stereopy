import os
import time

import pandas as pd
import panel as pn

from stereo.algorithm.algorithm_base import AlgorithmBase
from .cell_cell_communication.exceptions import InvalidMicroEnvInput
from .cell_cell_communication.exceptions import PipelineResultInexistent
from .cell_cell_communication.spatial_scoloc import GetMicroEnvs


class GenCccMicroEnvs(AlgorithmBase):
    def main(
            self,
            cluster_res_key: str = 'cluster',
            n_boot: int = 20,
            boot_prop: float = 0.8,
            dimension: int = 3,
            fill_rare: bool = True,
            min_num: int = 30,
            binsize: float = 2,
            eps: float = 1e-20,
            show_dividing_by_thresholds: bool = True,
            method: str = 'split',
            threshold: float = None,
            output_path: str = None,
            res_key: str = 'ccc_micro_envs'
    ):
        """
        Generate the micro-environment used for the CCC analysis.

        This function should be ran twice because it includes two parts:

        1) Calculating how the diffrent clusters are divided into diffrent micro environments under diffrent thresholds.
           You can choose an appropriate threshold based on the divided result.
           In order to run this part, you need to set the parameter `threshold` to None.
           The output is a dataframe like below:

            .. list-table::
                :header-rows: 1

                * - threshold
                  - subgroup_result
                * - 0.44298617727504136
                  - [{'1'}, {'2'}, {'3'}]
                * - 0.625776310617184
                  - [{'1', '2'}, {'3'}]

           The column `subgroup_result` is a list of sets, each set contains some groups and represents a micro-environment.

        2) Generating the micro environments by setting an appropriate `method` and `threshold` based on the result of first part.
           On this part, all the parameters before `method` are ignored.
           The output is a dataframe like below:

            .. list-table::
                :header-rows: 1

                * - cell_type
                  - microenviroment
                * - NKcells_1	
                  - microenv_0
                * - NKcells_0
                  - microenv_0
                * - Tcells
                  - microenv_1
                * - Myeloid
                  - microenv_2

        :param cluster_res_key: the key which specifies the clustering result in data.tl.result.
        :param n_boot: number of bootstrap samples, default = 100.
        :param boot_prop: proportion of each bootstrap sample, default = 0.8.
        :param dimension: 2 or 3.
        :param fill_rare: bool, whether simulate cells for rare cell type when calculating kde.
        :param min_num: if a cell type has cells < min_num, it is considered rare.
        :param binsize: grid size used for kde, it is used for gridding the space.
                        For example, a sample from square chip is gridded into mesh grids that have 100 intersections(determined by the given binsize),
                        For each cell type, fit the KDE according to the coordinates of all cells of this type and calculate KDE values of the 100 intersections.
                        Then KL divergence between each pair of cell types is calculated based on the calculated KDE values,
                        which is then used to construct the microenvironments.
        :param eps: fill eps to zero kde to avoid inf KL divergence.
        :param show_dividing_by_thresholds: whether to display the result while running the first part of this function.
        :param method: define micro environments using two methods:
                        1) minimum spanning tree, or
                        2) pruning the fully connected tree based on a given threshold of KL, then split the graph into multiple strongly connected component.
        :param threshold: the threshold to divide micro environment.
                        1) set it to None to run the first part of this function.
                        1) set it to an appropriate value to run the second part.
        :param output_path: the directory to save the result, if set it to None, the result is only stored in memory.
        :param res_key: set a key to store the result to data.tl.result, in second part, it must be set the same as first part.
        """  # noqa
        if threshold is None:
            if cluster_res_key not in self.pipeline_res:
                raise PipelineResultInexistent(cluster_res_key)

            if dimension not in [2, 3]:
                raise InvalidMicroEnvInput('Dimension number can only be 2 or 3.')

            if output_path is not None:
                assert os.path.exists(output_path), f"{output_path} is not exists."
                output_path = os.path.join(output_path,
                                           f'micro_envs_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}')
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

            cell_type = self.pipeline_res[cluster_res_key]['group']
            meta = pd.DataFrame({'cell': self.stereo_exp_data.cells.cell_name, 'cell_type': cell_type}).reset_index(
                drop=True)

            coord = pd.DataFrame({
                'cell': self.stereo_exp_data.cells.cell_name,
                'coord_x': self.stereo_exp_data.position[:, 0],
                'coord_y': self.stereo_exp_data.position[:, 1],
            })
            if dimension == 3:
                if self.stereo_exp_data.position_z is None:
                    raise InvalidMicroEnvInput(
                        "The position of cells must have the third dimension while setting `dimension` to 3.")
                coord['coord_z'] = self.stereo_exp_data.position_z

            gme = GetMicroEnvs()
            mst_in_boot, mst_final, pairwise_kl_divergence, split_by_different_threshold = gme.main(
                meta=meta,
                coord=coord,
                n_boot=n_boot,
                boot_prop=boot_prop,
                dimension=dimension,
                fill_rare=fill_rare,
                min_num=min_num,
                binsize=binsize,
                eps=eps,
                output_path=output_path
            )
            self.pipeline_res[res_key] = {
                'output_path': output_path,
                'mst_in_boot': mst_in_boot,
                'mst_final': mst_final,
                'pairwise_kl_divergence': pairwise_kl_divergence,
                'split_by_different_threshold': split_by_different_threshold
            }
            print("Now, you can choose an appropriate threshold based on this function's result.")
            if show_dividing_by_thresholds:
                pn.extension()
                # split_by_different_threshold_copy = split_by_different_threshold.copy()
                # split_by_different_threshold_copy['subgroup_result'] = split_by_different_threshold_copy[
                #     'subgroup_result'].astype('U')
                # return pn.widgets.DataFrame(split_by_different_threshold_copy, disabled=True, show_index=False,
                #                             autosize_mode="fit_viewport", frozen_columns=1)
                return pn.widgets.DataFrame(split_by_different_threshold, disabled=True, show_index=False,
                                            autosize_mode="fit_viewport", frozen_columns=1)
        else:
            if res_key not in self.pipeline_res:
                raise PipelineResultInexistent(res_key)

            if method not in ['mst', 'split']:
                raise ValueError(f"Invalid method({method}), choose it from 'mst' and 'split'")

            if method == 'mst':
                result_df = self.pipeline_res[res_key]['mst_final']
            else:
                result_df = self.pipeline_res[res_key]['pairwise_kl_divergence']
            output_path = self.pipeline_res[res_key]['output_path']
            gme = GetMicroEnvs()
            micro_envs = gme.generate_micro_envs(method, threshold=threshold, result_df=result_df,
                                                 output_path=output_path)
            self.pipeline_res[res_key]['micro_envs'] = micro_envs
