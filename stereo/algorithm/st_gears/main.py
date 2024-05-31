from typing import Optional, Union, List

from stereo.algorithm.ms_algorithm_base import MSDataAlgorithmBase
from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.io.reader import stereo_to_anndata
from stereo.log_manager import LogManager

from .helper import gen_anncell_cid_from_all
from .regis import serial_align
from .recons import stack_slices_pairwise_rigid, stack_slices_pairwise_elas_field

class StGears(MSDataAlgorithmBase):

    def main(
        self,
        cluster_res_key: str = None,
        start_i: int = 0,
        end_i: Optional[int] = None,
        tune_alpha_li: list = [0.2, 0.1, 0.05, 0.025, 0.01, 0.005],
        numItermax: int = 200,
        dissimilarity_val: str = 'kl',
        uniform_weight: bool = False,
        dissimilarity_weight_val: str = 'kl',
        map_method_dis2wei: str = 'logistic',
        filter_by_label: bool = True,
        use_gpu: bool = False,
        verbose: bool = False,
        res_key: str = 'st_gears'
    ):
        """
        ST-GEARS is a strong 3D reconstruction tool for Spatial Transcriptomics, with accurate position alignment plus distortion correction.

        It consists of methods to compute anchors, to rigidly align and to elastically registrate sections:

            This main function computes mappings between adjacent sections in serial,
            using Fused-Gromov Wasserstein Optimal Transport with our innovatie Distributive Constraints.

            `stack_slices_pairwise_rigid` rigidly aligns sections using Procrustes Analysis.

            `stack_slices_pairwise_elas_field` eliminates distorsions through Gaussian Smoothed Elastic Fields.
        
        This algorithm only supports AnnData currently, you can use st.io.stereo_to_anndata to convert your data to AnnData
        and input into st.io.read_h5ad to reload the data.

        :param cluster_res_key: The key to get a cluster result or the column name in obs where annotated cell types are stored, defaults to None
        :param start_i: The index of first sample to calulate, defaults to 0
        :param end_i: The index of last sample to calulate, defaults to None.
                        By default, it is the last of all samples.
        :param tune_alpha_li: List of regularization factor in Fused Gromov Wasserstin (FGW) OT problem formulation, to be
                        automatically tunned. Refer to this paper for the FGW formulation:
                        Optimal transport for structured data with application on graphs. T Vayer, L Chapel, R Flamary,
                        R Tavenard… - arXiv preprint arXiv …, 2018 - arxiv.org
        :param numItermax: Max number of iterations, defaults to 200
        :param dissimilarity_val: Matrix to calculate feature similarity. default to 'kl'.
                        Choose between 'kl' for Kullback-Leibler Divergence, and 'euc'/'euclidean' for euclidean distance.
        :param uniform_weight: Whether to assign same margin weights to every spots, defaults to False.
        :param dissimilarity_weight_val: Matrix to calculate cell types feature similarity when assigning weighted boundary conditions
                        for margin constrains. Refer to our paper for more details. Only assign when uniform_weight is False, defaults to 'kl'
        :param map_method_dis2wei: Methood to map cell types feature similarity to margin weighhts. Choose between linear' and 'logistic'.
                        Only assign when uniform_weight is False, defaults to 'logistic'
        :param filter_by_label: Where to filter out spots not appearing in its registered sample, so it won't interfere with the ot
                        solving process, defaults to True
        :param use_gpu: Whether to use GPU, in the parameter calculation process. OT solving process is only built on CPU, defaults to False
        :param verbose: Whether to print the OT solving process of each iteration, defaults to False
        :param res_key: The key to store calculating result, defaults to 'st_gears'
        """        
        try:
            from tensorflow.python.ops.numpy_ops import np_config
            np_config.enable_numpy_behavior()
        except:
            pass
        # data_list: List[AnnBasedStereoExpData] = self.ms_data.data_list
        assert isinstance(self.ms_data.data_list[0], AnnBasedStereoExpData), """
            ST-Gears only supports AnnData currently, you can use st.io.stereo_to_anndata to convert your data to AnnData
            and input into st.io.read_h5ad to reload the data.
            """

        adata_list = [data.adata for data in self.ms_data.data_list]

        # LogManager.stop_logging()
        # try:
        #     for data in data_list:
        #         if not isinstance(data, AnnBasedStereoExpData):
        #             adata = stereo_to_anndata(data, split_batches=False)
        #             adata_list.append(adata)
        #         else:
        #             adata_list.append(data.adata)
        # except Exception as e:
        #     raise Exception(e)
        # finally:
        #     LogManager.start_logging()

        anncell_cid = gen_anncell_cid_from_all(adata_list, cluster_res_key)

        if start_i is None:
            start_i = 0
        
        if end_i is None:
            end_i = len(adata_list) - 1
        
        pili, tyscoreli, alphali, regis_ilist, ali, bli = serial_align(
            adata_list, anncell_cid, cluster_res_key,
            start_i, end_i, tune_alpha_li, numItermax,
            dissimilarity_val, uniform_weight,
            dissimilarity_weight_val,
            map_method_dis2wei,
            filter_by_label,
            use_gpu, verbose
        )

        self.pipeline_res[res_key] = {
            'pili': pili,
            'tyscoreli': tyscoreli,
            'alphali': alphali,
            'regis_ilist': regis_ilist,
            'ali': ali,
            'bli': bli,
            'anncell_cid': anncell_cid,
            'parametres': {
                'cluster_res_key': cluster_res_key,
                'filter_by_label': filter_by_label,
                'spatial_key': self.ms_data.data_list[0].spatial_key,
            }
        }
        return self
    
    def stack_slices_pairwise_rigid(
        self,
        fil_pc: int = 20,
        set_as_position: bool = False,
        res_key: str = 'st_gears'
    ):
        """
        Rigidly aligning sections using Procrustes Analysis

        :param fil_pc: Percentage of ranked probabilities in transition matrix, after which probabilities will be filtered, defaults to 20
        :param set_as_position: Whether to use the new coordinates calculated by this method as the cells/bins position, defaults to False
        :param res_key: Must be the same as `res_key` in `main` function, defaults to 'st_gears'
        """
        assert res_key in self.pipeline_res, 'Please run ms_data.tl.st_gears first'

        adata_list = [self.ms_data.data_list[i].adata for i in self.pipeline_res[res_key]['regis_ilist']]
        self.pipeline_res[res_key]['adata_list'] = stack_slices_pairwise_rigid(
            slicesl=adata_list,
            pis=self.pipeline_res[res_key]['pili'],
            label_col=self.pipeline_res[res_key]['parametres']['cluster_res_key'],
            fil_pc=fil_pc,
            filter_by_label=self.pipeline_res[res_key]['parametres']['filter_by_label'],
            spatial_key=self.pipeline_res[res_key]['parametres']['spatial_key']
        )
        if set_as_position:
            for i in self.pipeline_res[res_key]['regis_ilist']:
                self.ms_data.data_list[i].spatial_key = 'spatial_rigid'

    def stack_slices_pairwise_elas_field(
        self,
        pixel_size: float,
        fil_pc: float = 20,
        sigma: float=1,
        set_as_position: bool = False,
        res_key: str = 'st_gears'
    ):
        """
        Eliminating distorsions through Gaussian Smoothed Elastic Fields

        :param pixel_size: Edge length of single pixel, when generating elastic field. Input a rough average of spots distance here
        :param fil_pc: Percentage of ranked probabilities in transition matrix, after which probabilities will be filtered, defaults to 20
        :param sigma: sigma value of gaussina kernel, when filtering noises in elastic registration field, with a higher value
            indicating a smoother elastic field. Refer to this `website <http://demofox.org/gauss.html>`_ to decide sigma according to 
            your desired range of convolution, defaults to 1
        :param set_as_position: Whether to use the new coordinates calculated by this method as the cells/bins position, defaults to False
        :param res_key: Must be the same as `res_key` in `main` function, defaults to 'st_gears'
        """
        assert res_key in self.pipeline_res, 'Please run ms_data.tl.st_gears first'

        adata_list = [self.ms_data.data_list[i].adata for i in self.pipeline_res[res_key]['regis_ilist']]
        self.pipeline_res[res_key]['adata_list'] = stack_slices_pairwise_elas_field(
            slicesl=adata_list,
            pis=self.pipeline_res[res_key]['pili'],
            label_col=self.pipeline_res[res_key]['parametres']['cluster_res_key'],
            fil_pc=fil_pc,
            pixel_size=pixel_size,
            sigma=sigma,
            filter_by_label=self.pipeline_res[res_key]['parametres']['filter_by_label'],
            spatial_key=self.pipeline_res[res_key]['parametres']['spatial_key']
        )
        if set_as_position:
            for i in self.pipeline_res[res_key]['regis_ilist']:
                self.ms_data.data_list[i].spatial_key = 'spatial_elas'

