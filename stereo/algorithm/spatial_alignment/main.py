from typing import Union, Optional

import pandas as pd

from stereo.io.reader import stereo_to_anndata
from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.algorithm.ms_algorithm_base import MSDataAlgorithmBase
from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.core.ms_data import MSData
from stereo.log_manager import logger
from stereo.preprocess.filter import filter_genes

from .spatialign import Spatialign

class SpatialAlignment(AlgorithmBase):

    def main(
        self,
        use_hvg: bool = False,
        n_neighors: int = 15,
        is_undirected: bool = True,
        spatial_key: str = 'spatial',
        latent_dims: int = 100,
        # seed: int = 42,
        gpu: Optional[Union[int, str]] = None,
        lr: float = 1e-3,
        max_epoch: int = 500,
        alpha: float = 0.5,
        patient: int = 15,
        tau1: float = 0.2,
        tau2: float = 1.0,
        tau3: float = 0.5,
        is_verbose: bool = True,
        inplace: bool = False
    ):
        """
        A method to remove batch effect.

        :param use_hvgs: Whether to use a subset only contains highly variable genes, defaults to False,
                            if True, `data.tl.highly_variable_genes` should be run first
                            and the data will be filtered to only contain highly variable genes.
        :param n_neighors: The number of neighbors selected when constructing a spatial neighbor graph, defaults to 15
        :param is_undirected: Whether the constructed spatial neighbor graph is undirected graph, defaults to True
        :param latent_dims: The number of embedding dimensions, defaults to 100,
                            a reduced dimension matrix whose shape is (n_cells, latent_dims) will be output,
                            you can get it through `data.tl.result['aligned_reduction']`.
        :param seed: Random seed, difference seed will cause difference result, defaults to 42
        :param gpu: Whether to use GPU to train, set the ID of GPU to be used, defaults to None to use CPU
        :param lr: Learning rate, defaults to 1e-3
        :param max_epoch: The number of maximum epochs, defaults to 500
        :param alpha: The momentum parameter, defaults to 0.5
        :param patient: Early stop parameter, defaults to 15
        :param tau1: Instance level and pseudo prototypical cluster level contrastive learning parameters, defaults to 0.2
        :param tau2: Pseudo prototypical cluster entropy parameter, defaults to 1.
        :param tau3: Cross-batch instance self-supervised learning parameter, defaults to 0.5
        :param is_verbose: Whether to print the detail information, defaults to True
        :param inplace: a corrected expression matrix will replace the `data.exp_matrix` if True
                            or will be stored in `data.layers['aligned_matrix']` if False, defaults to False.

        """

        if not isinstance(self.stereo_exp_data, AnnBasedStereoExpData):
            raise TypeError("The input data should be an object of AnnBasedStereoExpData.")

        if use_hvg:
            if 'highly_variable' not in self.stereo_exp_data.genes:
                raise KeyError(f"The data does not contain highly variable genes, run `data.tl.highly_variable_genes` first.")
            logger.info('The data only containing highly variable genes will be used.')
            hvg_flag = self.stereo_exp_data.genes['highly_variable']
            hvg_genes = self.stereo_exp_data.gene_names[hvg_flag]
            filter_genes(self.stereo_exp_data, gene_list=hvg_genes, inplace=True)

        self.model = Spatialign(
            merge_data=self.stereo_exp_data.adata,
            batch_key='batch',
            is_reduce=False,
            # n_pcs=n_pcs,
            n_neigh=n_neighors,
            is_undirected=is_undirected,
            latent_dims=latent_dims,
            tau1=tau1,
            tau2=tau2,
            tau3=tau3,
            is_verbose=is_verbose,
            # seed=42,
            gpu=gpu,
            # save_path=save_path,
            spatial_key=spatial_key
        )
        # return self.model
        self.model.train(lr=lr, max_epoch=max_epoch, alpha=alpha, patient=patient)
        aligned_matrix, aligned_reduction = self.model.alignment()
        
        if inplace:
            self.stereo_exp_data.exp_matrix = aligned_matrix
        else:
            self.stereo_exp_data.layers['aligned_matrix'] = aligned_matrix
        self.stereo_exp_data.cells_matrix['aligned_reduction'] = aligned_reduction