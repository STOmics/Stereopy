from typing import Any, Union, Sequence, Optional
from copy import deepcopy
import anndata
import numpy as np
import pandas as pd
try:
    import scvi
    import mudata
except ImportError:
    errmsg="""
**************************************************
* The scvi-tools and mudata may not be installed *
* Please run the commands:                       *
*     pip install scvi-tools==0.19.0             *
*     pip install mudata==0.1.2                  *
**************************************************
"""
    raise ImportError(errmsg)
from scipy.sparse import issparse

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.algorithm.ms_algorithm_base import MSDataAlgorithmBase
from stereo.core.stereo_exp_data import StereoExpData, AnnBasedStereoExpData
from stereo.io.reader import stereo_to_anndata
from stereo.log_manager import LogManager, logger
from stereo.preprocess.filter import filter_cells


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
        res_key: str = 'res_totalVI',
        mtx_res_key: str = 'totalVI',
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
            # if len(self.ms_data) >= 3:
            #     hvg_data = self.ms_data[2]
            # else:
            #     self._use_hvg = False
            self._use_hvg = False
        else:
            hvg_data = self.ms_data[hvg_key]
            self._use_hvg = True
        
        if rna_data.raw is None:
            raise Exception("Raw data is not exists, please run data.tl.raw_checkpoint before normalization.")

        # logger.debug(f'Before filtering, rna_data.shape: {rna_data.shape}, protein_data.shape: {protein_data.shape}')
        # if rna_data.shape != protein_data.shape:
        #     filter_cells(rna_data, cell_list=protein_data.cell_names)
        #     filter_cells(protein_data, cell_list=rna_data.cell_names)
        #     filter_cells(rna_data.raw, cell_list=rna_data.cell_names)
        #     if protein_data.raw:
        #         filter_cells(protein_data.raw, cell_list=protein_data.cell_names)
        # logger.debug(f'After filtering, rna_data.shape: {rna_data.shape}, protein_data.shape: {protein_data.shape}')

        if not isinstance(rna_data, AnnBasedStereoExpData):
            LogManager.stop_logging()
            rna_adata: anndata.AnnData = stereo_to_anndata(rna_data, split_batches=False)
            LogManager.start_logging()
        else:
            rna_adata: anndata.AnnData = deepcopy(rna_data._ann_data)
        
        if issparse(rna_adata.X):
            rna_adata.X = rna_adata.X.toarray()

        rna_adata.layers['counts'] = deepcopy(rna_adata.raw.X.toarray() if issparse(rna_adata.raw.X) else rna_adata.raw.X)
        
        if not isinstance(protein_data, AnnBasedStereoExpData):
            LogManager.stop_logging()
            protein_adata: anndata.AnnData = stereo_to_anndata(protein_data, split_batches=False)
            LogManager.start_logging()
        else:
            protein_adata: anndata.AnnData = protein_data._ann_data.copy()
        
        protein_adata = protein_adata[rna_adata.obs_names].copy()
        
        if issparse(protein_adata.X):
            protein_adata.X = protein_adata.X.toarray()
        
        mdata = mudata.MuData({"rna": rna_adata, "protein": protein_adata})
        
        if self._use_hvg:
            if not isinstance(hvg_data, AnnBasedStereoExpData):
                LogManager.stop_logging()
                hvg_adata: anndata.AnnData = stereo_to_anndata(hvg_data, split_batches=False)
                LogManager.start_logging()
            else:
                hvg_adata: anndata.AnnData = hvg_data._ann_data.copy()
            
            if issparse(hvg_adata.X):
                hvg_adata.X = hvg_adata.X.toarray()
            
            hvg_adata.layers['counts'] = deepcopy(hvg_adata.raw.X.toarray() if issparse(hvg_adata.raw.X) else hvg_adata.raw.X)
        
            mdata.mod['multiomics'] = hvg_adata
            mdata.update()
        
        scvi.model.TOTALVI.setup_mudata(
            mdata,
            rna_layer="counts",
            protein_layer=None,
            modalities={
                "rna_layer": "multiomics" if self._use_hvg else "rna",
                "protein_layer": "protein",
        })

        total_vi = scvi.model.TOTALVI(mdata, **kwags)
        total_vi.train()
        
        if not self._use_hvg:
            rna = rna_data
        else:
            rna = hvg_data
        
        protein = protein_data

        representation = pd.DataFrame(total_vi.get_latent_representation(), index=rna.cell_names)

        rna.tl.result[mtx_res_key] = representation
        rna.tl.reset_key_record('totalVI', mtx_res_key)

        protein.tl.result[mtx_res_key] = representation.loc[protein.cell_names].copy()
        protein.tl.reset_key_record('totalVI', mtx_res_key)

        rna.tl.result[mtx_res_key].index = pd.Index(range(0, rna.cell_names.size))
        protein.tl.result[mtx_res_key].index = pd.Index(range(0, protein.cell_names.size))


        denoised_rna, denoised_protein = total_vi.get_normalized_expression(n_samples=25, return_mean=True)

        rna.tl.result[res_key] = {
            'denoised_rna': denoised_rna
        }
        rna.tl.reset_key_record('res_totalVI', res_key)

        protein.tl.result[res_key] = {
            'denoised_protein': denoised_protein.loc[protein.cell_names],
            'protein_foreground_prob': total_vi.get_protein_foreground_probability(n_samples=25, return_mean=True).loc[protein.cell_names]
        }
        protein.tl.reset_key_record('res_totalVI', res_key)

        self._rna_data = rna_data
        self._protein_data = protein_data
        self._hvg_data = hvg_data
        self._res_key = res_key
        self._mtx_res_key = mtx_res_key
        self._total_vi_instance = total_vi
        # return self.save_result, total_vi
        return self


    def save_result(
        self,
        use_cluster_res_key: str = None,
        out_dir: str = None,
        diff_exp_file_name: str = None,
        h5mu_file_name: str = None
    ):
        import os.path as opth
        if out_dir is None or not opth.exists(out_dir):
            raise FileNotFoundError(f'The directory {out_dir} is not exists.')
        
        mdata = self._total_vi_instance.adata
        
        if self._use_hvg:
            rna_data = self._hvg_data
            mod_key = 'multiomics'
        else:
            rna_data = self._rna_data
            mod_key = 'rna'
        
        cluster_res = rna_data.tl.result[use_cluster_res_key].set_index('bins')
        mdata.mod[mod_key].obs[use_cluster_res_key] = cluster_res['group']
        mdata.update()
        de_df = self._total_vi_instance.differential_expression(groupby=f"{mod_key}:{use_cluster_res_key}", delta=0.5)
        if diff_exp_file_name is None:
            diff_exp_file_name = f'{rna_data.sn}_{rna_data.bin_size}_differential_expression.csv'
        de_df.to_csv(f'{out_dir}/{diff_exp_file_name}')
        
        for cluster_res_key in rna_data.tl.key_record['cluster']:
            # self._protein_data.tl.result[cluster_res_key] = rna_data.tl.result[cluster_res_key]
            self._protein_data.cells[cluster_res_key] = rna_data.cells[cluster_res_key]
        
        if not isinstance(self._rna_data, AnnBasedStereoExpData):
            LogManager.stop_logging()
            rna_adata: anndata.AnnData = stereo_to_anndata(self._rna_data, split_batches=False)
            LogManager.start_logging()
        else:
            rna_adata: anndata.AnnData = self._rna_data._ann_data
        
        if not isinstance(self._protein_data, AnnBasedStereoExpData):
            LogManager.stop_logging()
            protein_adata: anndata.AnnData = stereo_to_anndata(self._protein_data, split_batches=False)
            protein_adata.var['protein_names'] = protein_adata.var_names
            LogManager.start_logging()
        else:
            protein_adata: anndata.AnnData = self._protein_data._ann_data
        
        mdata = mudata.MuData({
            'rna': rna_adata,
            'protein': protein_adata
        })

        if self._use_hvg:
            if not isinstance(self._hvg_data, AnnBasedStereoExpData):
                LogManager.stop_logging()
                hvg_adata: anndata.AnnData = stereo_to_anndata(self._hvg_data, split_batches=False)
                LogManager.start_logging()
            else:
                hvg_adata: anndata.AnnData = self._hvg_data._ann_data
            mdata.mod['multiomics'] = hvg_adata
            mdata.update()
        if h5mu_file_name is None:
            h5mu_file_name = f'{rna_data.sn}_{rna_data.bin_size}.h5mu'
        mudata.write(f"{out_dir}/{h5mu_file_name}", mdata)

        # rna_data.tl.result[self._res_key]['differential_expression'] = de_df
        self._differential_expression = de_df
        self._use_cluster_res_key = use_cluster_res_key
    
    def filter_from_diff_exp(
        self,
        public_thresholds: dict = None,
        rna_thresholds: dict = None,
        protein_thresholds: dict = None,
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

            pro_rows = cell_type_df.index.str.contains("_")
            data_pro = cell_type_df.iloc[pro_rows]
            if protein_thresholds is not None:
                for column, threshold in protein_thresholds.items():
                    data_pro = data_pro[data_pro[column] > threshold]

            data_rna = cell_type_df.iloc[~pro_rows]
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