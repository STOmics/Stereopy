import gc
from copy import deepcopy
from typing import Optional
from typing import Union

import numpy as np

import anndata
import pandas as pd

try:
    import scvi
    import mudata
except ImportError:
    errmsg = """
**************************************************
* The scvi-tools and mudata may not be installed *
* Please run the commands:                       *
*     pip install scvi-tools==0.19.0             *
*     pip install mudata==0.1.2                  *
**************************************************
"""
    raise ImportError(errmsg)
from scipy.sparse import issparse

from stereo.algorithm.ms_algorithm_base import MSDataAlgorithmBase
from stereo.core.stereo_exp_data import AnnBasedStereoExpData
from stereo.core.stereo_exp_data import StereoExpData
from stereo.io.reader import stereo_to_anndata
from stereo.log_manager import LogManager


class TotalVi(MSDataAlgorithmBase):
    def __init__(self, *args, **kwargs):
        super(TotalVi, self).__init__(*args, **kwargs)
        self._use_hvg: bool = True
        self._rna_data: StereoExpData = None
        self._protein_data: StereoExpData = None
        self._hvg_data: StereoExpData = None
        self._total_vi_instance: scvi.model.TOTALVI = None

    def main(
            self,
            rna_key: Union[str, int] = None,
            protein_key: Union[str, int] = None,
            hvg_key: Union[str, int] = None,
            res_key: str = 'totalVI',
            rna_use_raw: bool = False,
            protein_use_raw: bool = False,
            # use_gpu: Union[int, str, bool] = None,
            accelerator: Union[str, None] = None,
            devices: Union[str, list[int], int, None] = None,
            num_threads: int = None,
            train_kwargs: Optional[dict] = {},
            **kwags
    ):
        if rna_key is None:
            rna_data = self.ms_data[0]
        else:
            rna_data = self.ms_data[rna_key]

        if protein_key is None:
            protein_data = self.ms_data[1]
        else:
            protein_data = self.ms_data[protein_key]

        if hvg_key is None:
            hvg_data = None
            self._use_hvg = False
        else:
            hvg_data = self.ms_data[hvg_key]
            self._use_hvg = True

        if rna_use_raw and rna_data.raw is None:
            raise Exception(
                "Raw data is not exists in rna data, please run data.tl.raw_checkpoint before normalization.")

        if protein_use_raw and protein_data.raw is None:
            raise Exception(
                "Raw data is not exists in protein data, please run data.tl.raw_checkpoint before normalization.")

        LogManager.stop_logging()
        try:
            if not isinstance(rna_data, AnnBasedStereoExpData):
                rna_adata: anndata.AnnData = stereo_to_anndata(rna_data, split_batches=False)
            else:
                rna_adata: anndata.AnnData = deepcopy(rna_data._ann_data)

            if issparse(rna_adata.X):
                rna_adata.X = rna_adata.X.toarray()

            if rna_use_raw:
                rna_adata.layers['counts'] = deepcopy(
                    rna_adata.raw.X.toarray() if issparse(rna_adata.raw.X) else rna_adata.raw.X)

            if not isinstance(protein_data, AnnBasedStereoExpData):
                protein_adata: anndata.AnnData = stereo_to_anndata(protein_data, split_batches=False)
            else:
                protein_adata: anndata.AnnData = protein_data._ann_data.copy()

            protein_adata = protein_adata[rna_adata.obs_names].copy()

            if issparse(protein_adata.X):
                protein_adata.X = protein_adata.X.toarray()

            if protein_use_raw:
                protein_adata.layers['counts'] = deepcopy(
                    protein_adata.raw.X.toarray() if issparse(protein_adata.raw.X) else protein_adata.raw.X)

            mdata = mudata.MuData({"rna": rna_adata, "protein": protein_adata})

            if self._use_hvg:
                if not isinstance(hvg_data, AnnBasedStereoExpData):
                    hvg_adata: anndata.AnnData = stereo_to_anndata(hvg_data, split_batches=False)
                else:
                    hvg_adata: anndata.AnnData = hvg_data._ann_data.copy()

                if issparse(hvg_adata.X):
                    hvg_adata.X = hvg_adata.X.toarray()

                if rna_use_raw:
                    hvg_adata.layers['counts'] = deepcopy(
                        hvg_adata.raw.X.toarray() if issparse(hvg_adata.raw.X) else hvg_adata.raw.X)

                mdata.mod['multiomics'] = hvg_adata
                mdata.update()
        finally:
            LogManager.start_logging()

        scvi.model.TOTALVI.setup_mudata(
            mdata,
            rna_layer="counts" if rna_use_raw else None,
            protein_layer="counts" if protein_use_raw else None,
            modalities={
                "rna_layer": "multiomics" if self._use_hvg else "rna",
                "protein_layer": "protein",
            })

        total_vi = scvi.model.TOTALVI(mdata, **kwags)
        scvi.settings.dl_num_workers = num_threads
        # total_vi.train(use_gpu=use_gpu, **train_kwargs)
        total_vi.train(accelerator=accelerator, devices=devices, **train_kwargs)

        if not self._use_hvg:
            rna = rna_data
        else:
            rna = hvg_data

        protein = protein_data

        representation = pd.DataFrame(total_vi.get_latent_representation(), index=rna.cell_names)
        rna.tl.result[res_key] = representation
        rna.tl.reset_key_record('totalVI', res_key)
        protein.tl.result[res_key] = representation.loc[protein.cell_names].copy()
        protein.tl.reset_key_record('totalVI', res_key)
        rna.tl.result[res_key].index = pd.Index(range(0, rna.cell_names.size))
        protein.tl.result[res_key].index = pd.Index(range(0, protein.cell_names.size))

        self._rna_data = rna_data
        self._protein_data = protein_data
        self._hvg_data = hvg_data
        self._res_key = res_key
        self._total_vi_instance = total_vi
        return self

    def save_result(
            self,
            use_cluster_res_key: str = None,
            out_dir: str = None,
            diff_exp_file_name: str = None,
            h5mu_file_name: str = None,
            fragment: int = 5,
            batch_size: int = 64
    ):
        import os.path as opth
        if out_dir is None or not opth.exists(out_dir):
            raise FileNotFoundError(f'The directory {out_dir} is not exists.')

        if self._use_hvg:
            clustered_data = self._hvg_data
        else:
            clustered_data = self._rna_data
        for cluster_res_key in clustered_data.tl.key_record['cluster']:
            self._protein_data.cells[cluster_res_key] = clustered_data.cells[cluster_res_key]
            self._protein_data.tl.reset_key_record('cluster', cluster_res_key)

        mdata = self._total_vi_instance.adata

        LogManager.stop_logging()
        try:
            rna_adata: anndata.AnnData = stereo_to_anndata(self._rna_data, base_adata=mdata.mod['rna'],
                                                           split_batches=False)
            protein_adata: anndata.AnnData = stereo_to_anndata(self._protein_data, base_adata=mdata.mod['protein'],
                                                               split_batches=False)
            # protein_adata.var['protein_names'] = protein_adata.var_names
            if self._use_hvg:
                mdata.mod['multiomics'].uns['omics'] = [['Transcriptomics'], ['Proteomics']]
                mdata.mod['multiomics'].uns['leiden_resolution'] = 1
                hvg_adata: anndata.AnnData = stereo_to_anndata(self._hvg_data, base_adata=mdata.mod['multiomics'],
                                                               split_batches=False)
            mdata.update()
        finally:
            LogManager.start_logging()

        mod_key = 'multiomics' if self._use_hvg else 'rna'
        rna = hvg_adata if self._use_hvg else rna_adata

        de_df = self._total_vi_instance.differential_expression(groupby=f"{mod_key}:{use_cluster_res_key}", delta=0.5)
        if diff_exp_file_name is None:
            diff_exp_file_name = f'{self._rna_data.sn}_{self._rna_data.bin_size}_differential_expression.csv'
        de_df.to_csv(f'{out_dir}/{diff_exp_file_name}')

        # Divide the 4000 gene list into 5 slices, After sharding,
        # the output arrays need to be merged and the shapes need to be consistent.
        frequency = len(rna.var_names) // fragment
        assert (frequency != 0), 'The number of slices of genes is wrong, causing the array to go out of bounds'
        denoised_rna_list = []
        denoised_protein_list = []
        rna_end = rna_start = 0
        for i in range(fragment):
            if i < fragment - 1:
                rna_end += frequency
                gene_list = rna.var_names[rna_start:rna_end]
                rna_start = rna_end
            else:
                gene_list = rna.var_names[rna_start:]
            denoised_rna_, denoised_protein_ = \
                self._total_vi_instance.get_normalized_expression(n_samples=25,
                                                                  batch_size=batch_size,
                                                                  gene_list=gene_list,
                                                                  return_mean=True)
            denoised_rna_list.append(denoised_rna_)
            denoised_protein_list.append(denoised_protein_)
            del denoised_rna_, denoised_protein_
            gc.collect()

        # Merging results after sharded model inference
        denoised_rna = pd.concat(denoised_rna_list, axis=1)
        denoised_protein = pd.concat(denoised_protein_list, axis=0)
        # he protein uses the average of five results. Can the median be used
        denoised_protein = denoised_protein.groupby(denoised_protein.index).sum()
        denoised_protein = denoised_protein.div(fragment, fill_value=np.NaN)

        denoised_protein = denoised_protein.loc[rna_adata.obs_names]

        protein_foreground_prob = self._total_vi_instance.get_protein_foreground_probability(
            n_samples=25, return_mean=True).loc[rna_adata.obs_names]

        rna.layers['denoised_rna'] = denoised_rna
        protein_adata.layers['denoised_protein'] = denoised_protein
        protein_adata.layers['protein_foreground_prob'] = protein_foreground_prob

        mdata.update()

        if h5mu_file_name is None:
            h5mu_file_name = f'{self._rna_data.sn}_{self._rna_data.bin_size}.h5mu'
        mudata.write_h5mu(f"{out_dir}/{h5mu_file_name}", mdata, compression='gzip')

        self._differential_expression = de_df
        self._use_cluster_res_key = use_cluster_res_key

    def filter_from_diff_exp(
            self,
            public_thresholds: dict = None,
            rna_thresholds: dict = None,
            protein_thresholds: dict = None,
            protein_list: list = None
    ):
        if self._use_hvg:
            rna_data = self._hvg_data
        else:
            rna_data = self._rna_data

        filtered_pro = []
        filtered_rna = []
        cats = rna_data.tl.result[self._use_cluster_res_key]['group'].cat.categories
        for i, c in enumerate(cats):
            cid = f"{c} vs Rest"
            cell_type_df = self._differential_expression.loc[self._differential_expression['comparison'] == cid]
            cell_type_df = cell_type_df.sort_values("lfc_median", ascending=False)

            # cell_type_df = cell_type_df[cell_type_df.lfc_median > lfc_median_thresh]
            if public_thresholds is not None:
                for column, threshold in public_thresholds.items():
                    cell_type_df = cell_type_df[cell_type_df[column] > threshold]

            # pro_rows = cell_type_df.index.str.contains("_")
            # data_pro = cell_type_df.iloc[pro_rows]
            pro_rows = np.intersect1d(cell_type_df.index, protein_list)
            data_pro = cell_type_df[cell_type_df.index.isin(pro_rows)]
            if protein_thresholds is not None:
                for column, threshold in protein_thresholds.items():
                    data_pro = data_pro[data_pro[column] > threshold]

            # data_rna = cell_type_df.iloc[~pro_rows]
            data_rna = cell_type_df[~cell_type_df.index.isin(pro_rows)]
            if rna_thresholds is not None:
                for column, threshold in rna_thresholds.items():
                    data_rna = data_rna[data_rna[column] > threshold]

            # filtered_pro[c] = data_pro.index.tolist()[:3]
            # filtered_rna[c] = data_rna.index.tolist()[:3]
            filtered_pro += data_pro.index.tolist()[:3]
            filtered_rna += data_rna.index.tolist()[:3]
        # filtered_rna = [x.split("_")[0] for v in filtered_rna.values() for x in v]
        # filtered_rna = list(set(filtered_rna))

        return filtered_rna, filtered_pro

    @property
    def totalvi(self) -> scvi.model.TOTALVI:
        if not hasattr(self, '_total_vi_instance'):
            raise AttributeError("'TotalVi' object has no attribute 'totalvi'")
        return self._total_vi_instance

    @property
    def differential_expression(self) -> pd.DataFrame:
        if not hasattr(self, '_differential_expression'):
            raise AttributeError("'TotalVi' object has no attribute 'differential_expression'")
        return self._differential_expression

    # def __getattr__(self, __name: str) -> Any:
    #     # dict_attr = self.__dict__.get(__name, None)
    #     # if dict_attr:
    #     #     return dict_attr

    #     if hasattr(self._total_vi_instance, __name):
    #         def run_function(data: StereoExpData, *args, **kwargs):
    #             func = getattr(self._total_vi_instance, __name, None)
    #             if func is not None:
    #                 res_key = __name.replace('get_', '')
    #                 data.tl.result[self._res_key][res_key] = func(*args, **kwargs)
    #                 return data.tl.result[self._res_key][res_key]
    #             else:
    #                 raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{__name}'")
    #         return run_function

    #     raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{__name}'")
