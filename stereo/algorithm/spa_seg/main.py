import os

from typing import Optional, Union, List

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.algorithm.ms_algorithm_base import MSDataAlgorithmBase
from stereo.core.stereo_exp_data import AnnBasedStereoExpData

from .spaseg import SpaSEG
from ._preprocessing import add_spot_pos

class SpaSeg(AlgorithmBase, MSDataAlgorithmBase):
    def __init__(self, stereo_exp_data=None, ms_data=None, pipeline_res=None):
        if stereo_exp_data is not None and ms_data is not None:
            raise ValueError("Only one of stereo_exp_data and ms_data should be provided")
        if stereo_exp_data is not None:
            AlgorithmBase.__init__(self, stereo_exp_data, pipeline_res)
        elif ms_data is not None:
            MSDataAlgorithmBase.__init__(self, ms_data, pipeline_res)
        else:
            raise ValueError("Either stereo_exp_data or ms_data should be provided")

        self._spaseg_model = None

    
    def main(
        self,
        seed: int = 1029,
        n_channel: int = None,
        n_conv: int = 2,
        lr: float = 0.002,
        weight_decay: float = 1e-5,
        pretrain_epochs: int = 400,
        iterations: int = 2100,
        sim_weight: float = 0.4,
        con_weight: float = 0.7,
        min_label: int = 7,
        gpu: Union[int, str] = None,
        result_prefix: str = "SpaSEG"
    ):
        """
        SpaSEG, an unsupervised convolutional neural network-based method towards multiple SRT analysis tasks
        by jointly learning transcriptional similarity between spots and their spatial dependence within tissue.
        SpaSEG adopts edge strength constraint to enable coherent spatial domains, and allows integrative SRT analysis
        by automatically aligning spatial domains across multiple adjacent sections.
        Moreover, SpaSEG can effectively detect spatial domain-specific gene expression patterns(SVG),
        and infer intercellular interactions and co-localizations.

        .. note::

            Currently, this algorithm only supports AnnData, if your data is read from GEF/GEM file,
            you can use `st.io.stereo_to_anndata <stereo.io.stereo_to_anndata.html>`_ to convert your data to AnnData
            and reload the data by `st.io.read_h5ad <stereo.io.read_h5ad.html>`_.

        :param seed: random seed, fixed value to fixed result, defaults to 1029.
        :param n_channel: the input/output channels of the middle convolutional layers, defaults to the dimension of PCA results.
                        In the process of convolution, the input channels of the first convolutional layer
                        is the same as the output channels of the last convolutional layer, which is set as the dimension of PCA results,
                        the output channels of the first convolutional layer, the input channels of the last convolutional layer and 
                        the input/output channels of all the middle convolutional layers are set as `n_channel`.
        :param n_conv: convolution will run `n_conv + 1` times, defaults to 2.
        :param lr: learning rate for Adam algorithm, defaults to 0.002.
        :param weight_decay: weight decay for Adam algorithm, defaults to 1e-5.
        :param pretrain_epochs: pretrain epochs, defaults to 400.
        :param iterations: training iterations, defaults to 2100.
        :param sim_weight: sim weight, defaults to 0.4.
        :param con_weight: con weight, defaults to 0.7.
        :param min_label: the number of labels when the training stops, defaults to 7.
        :param gpu: the GPU to be used, defaults to None to use CPU.
        :param result_prefix: the results will be save into obs whose column names are **'{result_prefix}_discrete_clusters'**
                            and **'{result_prefix}_clusters'**, defaults to **'SpaSEG'**, **'{result_prefix}_discrete_clusters'**
                            is the original cluster label, it may be discrete, **'{result_prefix}_clusters'** is the cluster label
                            after converting to continuous, which is used usually.

        :return: A SpaSeg object
        """
        if self.stereo_exp_data is not None:
            data_list: List[AnnBasedStereoExpData] = [self.stereo_exp_data]
        else:
            data_list: List[AnnBasedStereoExpData] = self.ms_data.data_list

        assert isinstance(data_list[0], AnnBasedStereoExpData), '''
            SpaSEG only supports AnnData currently, you can use st.io.stereo_to_anndata to convert your data to AnnData
            and then reload the data by st.io.read_h5ad.
            '''
        
        # data_list: List[AnnBasedStereoExpData] = self.ms_data.data_list
        adata_list = [add_spot_pos(data.adata, data.bin_type, data.spatial_key) for data in data_list]
            
        if gpu is None:
            use_gpu = False
            device = "cpu"
        else:
            if isinstance(gpu, str):
                gpu = int(gpu)
            if gpu < 0:
                use_gpu = False
                device = "cpu"
            else:
                use_gpu = True
                device = f"cuda:{gpu}"

        input_dim = output_dim = adata_list[0].obsm['X_pca'].shape[1]

        if n_channel is None:
            n_channel = input_dim
        
        spaseg_model = SpaSEG(
            adata=adata_list,
            use_gpu=use_gpu,
            device=device,
            seed=seed,
            input_dim=input_dim,
            nChannel=n_channel,
            output_dim=output_dim,
            nConv=n_conv,
            lr=lr,
            weight_decay=weight_decay,
            pretrain_epochs=pretrain_epochs,
            iterations=iterations,
            sim_weight=sim_weight,
            con_weight=con_weight,
            min_label=min_label,
            # spot_size=spot_size,
            result_prefix=result_prefix
        )

        # prepare image-like tensor data for SpaSEG model input
        input_mxt, H, W = spaseg_model._prepare_data()

        # SpaSEG traning
        cluster_label, embedding = spaseg_model._train(input_mxt)

        # n_batch = self.ms_data.num_slice

        # Add SpaSEG segmentation label for each spot/bin in SRT data
        spaseg_model._add_seg_label(cluster_label, H, W)

        self._spaseg_model = spaseg_model
        return self
    
    def calculate_metrics(
        self,
        true_labels_column: Optional[str] = None,
    ):
        if self._spaseg_model is None:
            raise ValueError("Please run SpaSEG first before calculating metrics.")
        
        self._spaseg_model._cal_metrics(true_labels_column)