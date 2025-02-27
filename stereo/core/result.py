from warnings import warn
from typing import Union
from copy import deepcopy

import pandas as pd
import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

from stereo.core.stereo_exp_data import StereoExpData, AnnBasedStereoExpData

class _BaseResult(object):
    CLUSTER_NAMES = {
        'leiden', 'louvain', 'phenograph', 'annotation', 'leiden_from_bins', 'louvain_from_bins',
        'phenograph_from_bins', 'annotation_from_bins', 'celltype', 'cell_type'
    }
    PREFIX_FOR_NON_CLUSTER = {
        'gene_exp', 'silhouette_score', 'adjusted_rand_score'
    }
    CONNECTIVITY_NAMES = {'neighbors'}
    REDUCE_NAMES = {'umap', 'pca', 'tsne', 'correct'}
    HVG_NAMES = {'highly_variable_genes', 'hvg', 'highly_variable'}
    MARKER_GENES_NAMES = {
        'marker_genes', 'marker_genes_filtered',
        'rank_genes_groups', 'rank_genes_groups_filtered'
    }
    SCT_NAMES = {'sctransform', 'sct'}

    RENAME_DICT = {
        'highly_variable_genes': 'hvg',
        'marker_genes': 'rank_genes_groups',
        'marker_genes_filtered': 'rank_genes_groups_filtered'
    }

    CLUSTER, CONNECTIVITY, REDUCE, HVG, MARKER_GENES, SCT = 0, 1, 2, 3, 4, 5
    TYPE_NAMES_DICT = {
        CLUSTER: CLUSTER_NAMES,
        CONNECTIVITY: CONNECTIVITY_NAMES,
        REDUCE: REDUCE_NAMES,
        HVG: HVG_NAMES,
        MARKER_GENES: MARKER_GENES_NAMES,
        SCT: SCT_NAMES
    }

    def __init__(self):
        self.set_result_key_method = None

    def __setitem__(self, key, _):
        if self.set_result_key_method:
            self.set_result_key_method(key)
    

class Result(_BaseResult, dict):

    def __init__(
        self,
        stereo_exp_data: StereoExpData,
        *args,
        **kwargs
    ):
        # super().__init__()
        if not isinstance(stereo_exp_data, StereoExpData):
            raise TypeError("stereo_exp_data must be an object of StereoExpData.")
        
        _BaseResult.__init__(self)
        dict.__init__(self, *args, **kwargs)
        self.__stereo_exp_data = stereo_exp_data
        self.set_item_callback = None
        self.get_item_method = None
        self.contain_method = None

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        if id(self) in memo:
            new_result = memo[id(self)]
        else:
            new_result = Result(self.__stereo_exp_data)
            memo[id(self)] = new_result
            if id(self.__stereo_exp_data) in memo:
                data = memo[id(self.__stereo_exp_data)]
            else:
                data = deepcopy(self.__stereo_exp_data, memo)
            
            new_attrs = {
                deepcopy(k, memo): deepcopy(v, memo) for k, v in self.__dict__.items() if k != '_Result__stereo_exp_data'
            }
            new_attrs['_Result__stereo_exp_data'] = data
            new_result.__dict__.update(new_attrs)
            for k, v in self.items():
                dict.__setitem__(new_result, deepcopy(k, memo), deepcopy(v, memo))
        return new_result

    def __contains__(self, item):
        if self.contain_method:
            if self.contain_method(item):
                return True
            # TODO: when get item in ms_data[some_idx].tl.result, if name match the ms_data rule, it is very confused
        if item in self.__stereo_exp_data.genes:
            return True
        elif item in self.__stereo_exp_data.genes_matrix:
            return True
        elif item in self.__stereo_exp_data.genes_pairwise:
            return True
        elif item in self.__stereo_exp_data.cells:
            return True
        elif item in self.__stereo_exp_data.cells_matrix:
            return True
        elif item in self.__stereo_exp_data.cells_pairwise:
            return True
        is_contained = dict.__contains__(self, item)
        if is_contained:
            res = dict.__getitem__(self, item)
            if type(res) is dict and 'connectivities_key' in res and 'distances_key' in res:
                if res['connectivities_key'] in self.__stereo_exp_data.cells_pairwise and \
                        res['distances_key'] in self.__stereo_exp_data.cells_pairwise:
                    return True
                else:
                    return False
            if item in self.HVG_NAMES or any([n in item for n in self.HVG_NAMES]):
                if self._get_hvg_res(item) is not None:
                    return True
                else:
                    return False
        return is_contained

    def __getitem__(self, name):
        if self.get_item_method:
            item = self.get_item_method(name)
            if item is not None:
                return item
            # TODO: when get item in ms_data[some_idx].tl.result, if name match the ms_data rule, it is very confused

        genes = self.__stereo_exp_data.genes
        cells = self.__stereo_exp_data.cells
        if name in genes._var:
            # warn(
            #     f'{name} will be moved from `StereoExpData.tl.result` to `StereoExpData.genes` in the '
            #     f'future, make sure your code access the property correctly.',
            #     category=FutureWarning
            # )
            if name in self.HVG_NAMES or any([n in name for n in self.HVG_NAMES]):
                return self._get_hvg_res(name)
            else:
                return genes._var[name]
        elif name in genes._matrix:
            # warn(
            #     f'{name} will be moved from `StereoExpData.tl.result` to `StereoExpData.genes_matrix` in the '
            #     f'future, make sure your code access the property correctly.',
            #     category=FutureWarning
            # )
            return genes._matrix[name]
        elif name in genes._pairwise:
            # warn(
            #     f'{name} will be moved from `StereoExpData.tl.result` to `StereoExpData.genes_pairwise` in the '
            #     f'future, make sure your code access the property correctly.',
            #     category=FutureWarning
            # )
            return genes._pairwise[name]
        elif name in cells._obs.columns:
            # warn(
            #     f'{name} will be moved from `StereoExpData.tl.result` to `StereoExpData.cells` in the '
            #     f'future, make sure your code access the property correctly. ',
            #     category=FutureWarning
            # )
            return pd.DataFrame(
                {
                    'bins': cells.cell_name,
                    'group': cells._obs[name].values
                }
            )
        elif name in cells._matrix:
            # warn(
            #     f'{name} will be moved from `StereoExpData.tl.result` to `StereoExpData.cells_matrix` in the '
            #     f'future, make sure your code access the property correctly. ',
            #     category=FutureWarning
            # )
            return cells._matrix[name]
        elif name in cells._pairwise:
            # warn(
            #     f'{name} will be moved from `StereoExpData.tl.result` to `StereoExpData.cells_pairwise` in the '
            #     f'future, make sure your code access the property correctly. ',
            #     category=FutureWarning
            # )
            return cells._pairwise[name]
        elif name in self.HVG_NAMES or any([n in name for n in self.HVG_NAMES]):
            if dict.__contains__(self, name):
                return self._get_hvg_res(name)
        res = dict.__getitem__(self, name)
        if type(res) is dict and 'connectivities_key' in res and 'distances_key' in res:
            return {
                'neighbor': None,
                'connectivities': cells._pairwise[res['connectivities_key']],
                'nn_dist': cells._pairwise[res['distances_key']],
                'params': res['params'] if 'params' in res else {}
            }
        return res
    
    def _get_hvg_res(self, name):
        hvg_colunms = []
        if dict.__contains__(self, name):
            all_columns = dict.__getitem__(self, name)
        else:
            all_columns = ["means", "mean_bin", "dispersions", "dispersions_norm", name]
        for c in all_columns:
            if c in self.__stereo_exp_data.genes._var.columns:
                hvg_colunms.append(c)
        if len(hvg_colunms) > 0:
            return self.__stereo_exp_data.genes._var.loc[:, hvg_colunms]

    def _real_set_item(self, type, key, value):
        if type == Result.CLUSTER:
            for prefix in Result.PREFIX_FOR_NON_CLUSTER:
                if key.startswith(prefix):
                    return False
            self._set_cluster_res(key, value)
        elif type == Result.CONNECTIVITY:
            self._set_connectivities_res(key, value)
        elif type == Result.REDUCE and 'variance_ratio' not in key:
            return self._set_reduce_res(key, value)
        elif type == Result.HVG:
            self._set_hvg_res(key, value)
        elif type == Result.MARKER_GENES:
            self._set_marker_genes_res(key, value)
        else:
            return False
        return True

    def __setitem__(self, key, value):
        _BaseResult.__setitem__(self, key, value)
        
        if self.set_item_callback:
            self.set_item_callback(key, value)
            return
        for name_type, name_dict in Result.TYPE_NAMES_DICT.items():
            if key in name_dict and self._real_set_item(name_type, key, value):
                return
        for name_type, name_dict in Result.TYPE_NAMES_DICT.items():
            for like_name in name_dict:
                # if not key.startswith('gene_exp_') and like_name in key and self._real_set_item(name_type, key, value):
                if like_name in key and self._real_set_item(name_type, key, value):
                    return
        if type(value) is pd.DataFrame:
            if 'bins' in value.columns.values and 'group' in value.columns.values:
                self._set_cluster_res(key, value)
                return
            elif not {"means", "dispersions", "dispersions_norm", "highly_variable"} - set(value.columns.values):
                self._set_hvg_res(key, value)
                return
            # elif key.startswith('gene_exp_'):
            #     dict.__setitem__(self, key, value)
            #     return
            # elif len(value.shape) == 2 and value.shape[0] > 399 and value.shape[1] > 399:
            # elif len(value.shape) == 2 and \
            #     value.shape[0] == self.__stereo_exp_data.shape[0] and value.shape[1] <= self.__stereo_exp_data.shape[1]:
            #     # TODO this is hard-code method to guess it's a reduce ndarray
            #     self._set_reduce_res(key, value)
            #     return
            elif len(value.shape) == 2:
                if value.shape == (self.__stereo_exp_data.n_cells, self.__stereo_exp_data.n_cells):
                    self.__stereo_exp_data.cells._pairwise[key] = value
                    return
                elif value.shape == (self.__stereo_exp_data.n_genes, self.__stereo_exp_data.n_genes):
                    self.__stereo_exp_data.genes._pairwise[key] = value
                    # self._set_reduce_res(key, value)
                    return
                elif value.shape[0] == self.__stereo_exp_data.n_cells and value.shape[1] < self.__stereo_exp_data.n_genes:
                    # TODO this is hard-code method to guess it's a reduce ndarray
                    self._set_reduce_res(key, value)
                    return
                elif value.shape[0] == self.__stereo_exp_data.n_genes and value.shape[1] < self.__stereo_exp_data.n_cells:
                    # TODO this is hard-code method to guess it's a reduce ndarray
                    self._set_reduce_res(key, value)
                    return
        elif type(value) is dict:
            if not {'connectivities', 'nn_dist'} - set(value.keys()):
                self._set_connectivities_res(key, value)
                return
        elif (type(value) is np.ndarray or issparse(value)) and value.ndim >= 2:
            if value.shape == (self.__stereo_exp_data.n_cells, self.__stereo_exp_data.n_cells):
                self.__stereo_exp_data.cells._pairwise[key] = value
                return
            elif value.shape == (self.__stereo_exp_data.n_genes, self.__stereo_exp_data.n_genes):
                self.__stereo_exp_data.genes._pairwise[key] = value
                return
            elif value.shape[0] == self.__stereo_exp_data.n_cells and value.shape[1] < self.__stereo_exp_data.n_genes:
                # TODO this is hard-code method to guess it's a reduce ndarray
                self._set_reduce_res(key, value)
                return
            elif value.shape[0] == self.__stereo_exp_data.n_genes and value.shape[1] < self.__stereo_exp_data.n_cells:
                # TODO this is hard-code method to guess it's a reduce ndarray
                self._set_reduce_res(key, value)
                return
        dict.__setitem__(self, key, value)

    def _set_cluster_res(self, key, value):
        assert type(value) is pd.DataFrame and 'group' in value.columns.values, "this is not cluster res"
        # warn(
        #     f'FutureWarning: {key} will be moved from `StereoExpData.tl.result` to `StereoExpData.cells` in the '
        #     'future, make sure your code set the property correctly. ',
        #     category=FutureWarning
        # )
        self.__stereo_exp_data.cells._obs[key] = value['group'].values
        self.CLUSTER_NAMES.add(key)

    def _set_connectivities_res(self, key, value):
        assert type(value) is dict and not {'connectivities', 'nn_dist'} - set(value.keys()), \
            'not enough key to set connectivities'
        # warn(
        #     f'FutureWarning: {key} will be moved from `StereoExpData.tl.result` to `StereoExpData.cells_pairwise` in '
        #     f'the future, make sure your code set the property correctly. ',
        #     category=FutureWarning
        # )
        params = {}
        if key == 'neighbors':
            connectivities_key = 'connectivities'
            distances_key = 'distances'
        else:
            connectivities_key = f'{key}_connectivities'
            distances_key = f'{key}_distances'
        params['connectivities_key'] = connectivities_key
        params['distances_key'] = distances_key
        if 'params' in value:
            params['params'] = value['params']

        # self.__stereo_exp_data.cells._pairwise[key] = value
        self.__stereo_exp_data.cells._pairwise[connectivities_key] = value['connectivities']
        self.__stereo_exp_data.cells._pairwise[distances_key] = value['nn_dist']
        self.CONNECTIVITY_NAMES.add(key)
        dict.__setitem__(self, key, params)

    def _set_reduce_res(self, key, value):
        # assert type(value) is pd.DataFrame, 'reduce result must be pandas.DataFrame'
        if not isinstance(value, (pd.DataFrame, np.ndarray)):
            # raise TypeError('reduce result must be pandas.DataFrame or numpy.ndarray')
            return False
        # warn(
        #     f'{key} will be moved from `StereoExpData.tl.result` to `StereoExpData.cells_matrix` in the '
        #     f'future, make sure your code set the property correctly. ',
        #     category=FutureWarning
        # )
        if value.shape[0] == self.__stereo_exp_data.n_cells and \
                value.shape[1] < self.__stereo_exp_data.n_genes:
            self.__stereo_exp_data.cells._matrix[key] = value
            self.REDUCE_NAMES.add(key)
            return True
        elif value.shape[0] == self.__stereo_exp_data.n_genes and \
                value.shape[1] < self.__stereo_exp_data.n_cells:
            self.__stereo_exp_data.genes._matrix[key] = value
            self.REDUCE_NAMES.add(key)
            return True
        else:
            return False

    def _set_hvg_res(self, key, value):
        assert type(value) is pd.DataFrame, 'hvg result must be pandas.DataFrame'
        # self.__stereo_exp_data.genes._var = pd.concat([self.__stereo_exp_data.genes._var, value], axis=1)
        for c in value.columns:
            self.__stereo_exp_data.genes[c] = value[c]
        self.HVG_NAMES.add(key)
        columns = value.columns.to_list()    
        dict.__setitem__(self, key, columns)

    def _set_marker_genes_res(self, key, value):
        dict.__setitem__(self, key, value)

    def set_value(self, key, value):
        if hasattr(value, 'shape'):
            if len(value.shape) >= 2:
                if value.shape[0:2] == (self.__stereo_exp_data.n_cells, self.__stereo_exp_data.n_cells):
                    self.__stereo_exp_data.cells_pairwise[key] = value
                elif value.shape[0:2] == (self.__stereo_exp_data.n_genes, self.__stereo_exp_data.n_genes):
                    self.__stereo_exp_data.genes_pairwise[key] = value
                elif value.shape[0] == self.__stereo_exp_data.n_cells:
                    self.__stereo_exp_data.cells_matrix[key] = value
                elif value.shape[0] == self.__stereo_exp_data.n_genes:
                    self.__stereo_exp_data.genes_matrix[key] = value
                else:
                    dict.__setitem__(self, key, value)
            else:
                dict.__setitem__(self, key, value)
        else:
            dict.__setitem__(self, key, value)


class AnnBasedResult(_BaseResult, object):

    def __init__(self, data: AnnBasedStereoExpData):
        super().__init__()
        # self.__stereo_exp_data = data
        self.__based_ann_data = data.adata
        self.__result_keys = []
    
    # @property
    # def adata(self):
    #     return self.__based_ann_data

    def __contains__(self, item):
        if item in AnnBasedResult.CLUSTER_NAMES:
            return item in self.__based_ann_data.obs
        elif item in AnnBasedResult.CONNECTIVITY_NAMES:
            if item in self.__based_ann_data.uns and type(self.__based_ann_data.uns[item]) is dict:
                if 'connectivities_key' in self.__based_ann_data.uns[item] and \
                        'distances_key' in self.__based_ann_data.uns[item]:
                        if self.__based_ann_data.uns[item]['connectivities_key'] in self.__based_ann_data.obsp and \
                            self.__based_ann_data.uns[item]['distances_key'] in self.__based_ann_data.obsp:
                            return True
                        else:
                            return False
            return item in self.__based_ann_data.uns
        elif item in AnnBasedResult.REDUCE_NAMES:
            return f'X_{item}' in self.__based_ann_data.obsm
        elif item in AnnBasedResult.HVG_NAMES:
            if item in self.__based_ann_data.uns:
                return True
            elif AnnBasedResult.RENAME_DICT.get(item, None) in self.__based_ann_data.uns:
                return True
        elif item in AnnBasedResult.MARKER_GENES_NAMES:
            if item in self.__based_ann_data.uns:
                return True
            elif AnnBasedResult.RENAME_DICT.get(item, None) in self.__based_ann_data.uns:
                return True
        elif item in AnnBasedResult.SCT_NAMES:
            if item in self.__based_ann_data.uns and type(self.__based_ann_data.uns[item]) is dict:
                sct_key = self.__based_ann_data.uns[item].get('sct_key', None)
                sct_id = self.__based_ann_data.uns[item].get('id', None)
                if sct_key is not None and sct_id is not None:
                    sct_key_1 = f"{sct_key}_{sct_id}_1"
                    sct_key_2 = f"{sct_key}_{sct_id}_2"
                    if sct_key_1 in self.__based_ann_data.uns and sct_key_2 in self.__based_ann_data.uns:
                            return True
        # elif item.startswith('gene_exp_'):
        #     if item in self.__based_ann_data.uns:
        #         return True
        elif item.startswith('paga'):
            if item in self.__based_ann_data.uns:
                return True
        elif item.startswith('regulatory_network_inference'):
            if f'{item}_regulons' in self.__based_ann_data.uns:
                return True
            elif f'{item}_auc_matrix' in self.__based_ann_data.uns:
                return True
            elif f'{item}_adjacencies' in self.__based_ann_data.uns:
                return True

        obsm_obj = self.__based_ann_data.obsm.get(f'X_{item}', None)
        if obsm_obj is not None:
            return True
        obsm_obj = self.__based_ann_data.obsm.get(f'{item}', None)
        if obsm_obj is not None:
            return True
        obs_obj = self.__based_ann_data.obs.get(item, None)
        if obs_obj is not None:
            return True
        obsp_obj = self.__based_ann_data.obsp.get(item, None)
        if obsp_obj is not None:
            return True
        varm_obj = self.__based_ann_data.varm.get(item, None)
        if varm_obj is not None:
            return True
        var_obj = self.__based_ann_data.var.get(item, None)
        if var_obj is not None:
            return True
        varp_obj = self.__based_ann_data.varp.get(item, None)
        if varp_obj is not None:
            return True
        uns_obj = self.__based_ann_data.uns.get(item, None)
        if uns_obj is not None:
            if type(uns_obj) is dict and 'connectivities_key' in uns_obj and 'distances_key' in uns_obj:
                if uns_obj['connectivities_key'] in self.__based_ann_data.obsp and \
                        uns_obj['distances_key'] in self.__based_ann_data.obsp:
                    return True
                else:
                    return False
            return True
        return False

    def __getitem__(self, name):
        if name in AnnBasedResult.CLUSTER_NAMES:
            return pd.DataFrame({
                'bins': self.__based_ann_data.obs_names,
                'group': self.__based_ann_data.obs[name].values
            })
        elif name in AnnBasedResult.CONNECTIVITY_NAMES:
            neighbors_res = {
                'neighbor': None,  # TODO really needed?
                'connectivities': self.__based_ann_data.obsp['connectivities'],
                'nn_dist': self.__based_ann_data.obsp['distances'],
            }
            if 'params' in self.__based_ann_data.uns[name]:
                neighbors_res['params'] = self.__based_ann_data.uns[name]['params']
            else:
                neighbors_res['params'] = {}
            return neighbors_res
        elif name in AnnBasedResult.REDUCE_NAMES:
            return pd.DataFrame(self.__based_ann_data.obsm[f'X_{name}'], copy=False)
        elif name in AnnBasedResult.HVG_NAMES:
            # TODO ignore `mean_bin`, really need?
            hvg_colunms = []
            for c in ["means", "mean_bin", "dispersions", "dispersions_norm", "highly_variable"]:
                if c in self.__based_ann_data.var.columns:
                    hvg_colunms.append(c)
            if len(hvg_colunms) > 0:
                return self.__based_ann_data.var.loc[:, hvg_colunms]
        elif name in AnnBasedResult.MARKER_GENES_NAMES or \
            any([n in name for n in AnnBasedResult.MARKER_GENES_NAMES]):
            return self._get_marker_genes_res(name)
        elif name in AnnBasedResult.SCT_NAMES or any([n in name for n in AnnBasedResult.SCT_NAMES]):
            if name in self.__based_ann_data.uns and type(self.__based_ann_data.uns[name]) is dict and \
                'sct_key' in self.__based_ann_data.uns[name] and 'id' in self.__based_ann_data.uns[name]:
                # sct_key = self.__based_ann_data.uns[name]['sct_key']
                # sct_id = self.__based_ann_data.uns[name]['id'] 
                # sct_key_1 = f"{sct_key}_{sct_id}_1"
                # sct_key_2 = f"{sct_key}_{sct_id}_2"
                # return self.__based_ann_data.uns[sct_key_1], self.__based_ann_data.uns[sct_key_2]
                return self._get_sctransform_res(name)
        # elif name.startswith('gene_exp_'):
        #     return self.__based_ann_data.uns[name]
        # elif name.startswith('regulatory_network_inference'):
        #     return self.__based_ann_data.uns[name]

        obsm_obj = self.__based_ann_data.obsm.get(f'X_{name}', None)
        if obsm_obj is not None:
            return pd.DataFrame(obsm_obj)
        obsm_obj = self.__based_ann_data.obsm.get(f'{name}', None)
        if obsm_obj is not None:
            return pd.DataFrame(obsm_obj)
        obs_obj = self.__based_ann_data.obs.get(name, None)
        if obs_obj is not None:
            return pd.DataFrame({
                'bins': self.__based_ann_data.obs_names,
                'group': self.__based_ann_data.obs[name].values
            })
        obsp_obj = self.__based_ann_data.obsp.get(name, None)
        if obsp_obj is not None:
            return obsp_obj
        varm_obj = self.__based_ann_data.varm.get(name, None)
        if varm_obj is not None:
            return varm_obj
        varp_obj = self.__based_ann_data.varp.get(name, None)
        if varp_obj is not None:
            return varp_obj
        uns_obj = self.__based_ann_data.uns.get(name, None)
        if uns_obj is not None and type(uns_obj) is dict and 'params' in uns_obj and \
                'connectivities_key' in uns_obj and 'distances_key' in uns_obj:
            neighbors_res = {
                'neighbor': None,  # TODO really needed?
                'connectivities': self.__based_ann_data.obsp[uns_obj['connectivities_key']],
                'nn_dist': self.__based_ann_data.obsp[uns_obj['distances_key']],
            }
            if 'params' in uns_obj:
                neighbors_res['params'] = uns_obj['params']
            else:
                neighbors_res['params'] = {}
            return neighbors_res
        elif uns_obj is not None and type(uns_obj) is dict and 'sct_key' in uns_obj and 'id' in uns_obj:
            # sct_key_1 = f'{uns_obj["sct_key"]}_{uns_obj["id"]}_1'
            # sct_key_2 = f'{uns_obj["sct_key"]}_{uns_obj["id"]}_2'
            # return self.__based_ann_data.uns[sct_key_1], self.__based_ann_data.uns[sct_key_2]
            return self._get_sctransform_res(name)
        elif uns_obj is not None:
            return uns_obj
        raise KeyError(name)

    def _real_set_item(self, type, key, value):
        if type == AnnBasedResult.CLUSTER:
            for prefix in AnnBasedResult.PREFIX_FOR_NON_CLUSTER:
                if key.startswith(prefix):
                    return False
            self._set_cluster_res(key, value)
        elif type == AnnBasedResult.CONNECTIVITY:
            self._set_connectivities_res(key, value)
        elif type == AnnBasedResult.REDUCE and 'variance_ratio' not in key:
            return self._set_reduce_res(key, value)
        elif type == AnnBasedResult.HVG_NAMES:
            self._set_hvg_res(key, value)
        elif type == AnnBasedResult.MARKER_GENES:
            self._set_marker_genes_res(key, value)
        elif type == AnnBasedResult.SCT:
            self._set_sctransform_res(key, value)
        else:
            return False
        return True

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

        self.__result_keys.append(key)

        for name_type, name_dict in AnnBasedResult.TYPE_NAMES_DICT.items():
            if key in name_dict and self._real_set_item(name_type, key, value):
                return

        for name_type, name_dict in AnnBasedResult.TYPE_NAMES_DICT.items():
            for like_name in name_dict:
                if like_name in key and self._real_set_item(name_type, key, value):
                    return

        if type(value) is pd.DataFrame:
            if 'bins' in value.columns.values and 'group' in value.columns.values:
                self._set_cluster_res(key, value)
                return
            elif not {"means", "dispersions", "dispersions_norm", "highly_variable"} - set(value.columns.values):
                self._set_hvg_res(key, value)
                return
            elif key.startswith('gene_exp_'):
                self.__based_ann_data.varm[key] = value
                return
            # elif len(value.shape) == 2 and \
            #         value.shape[0] == self.__based_ann_data.shape[0] and value.shape[1] <= self.__based_ann_data.shape[1]:
            #     # TODO this is hard-code method to guess it's a reduce ndarray
            #     self._set_reduce_res(key, value)
            #     return
            elif len(value.shape) == 2:
                if value.shape == (self.__based_ann_data.n_obs, self.__based_ann_data.n_obs):
                    self.__based_ann_data.obsp[key] = value
                    return
                elif value.shape == (self.__based_ann_data.n_vars, self.__based_ann_data.n_vars):
                    self.__based_ann_data.varp[key] = value
                    return
                elif value.shape[0] == self.__based_ann_data.n_obs and value.shape[1] < self.__based_ann_data.n_vars:
                    # TODO this is hard-code method to guess it's a reduce ndarray
                    self._set_reduce_res(key, value, start_with_X=False)
                    return
                elif value.shape[0] == self.__based_ann_data.n_vars and value.shape[1] < self.__based_ann_data.n_obs:
                    # TODO this is hard-code method to guess it's a reduce ndarray
                    self._set_reduce_res(key, value)
        elif type(value) is dict:
            if not {'connectivities', 'nn_dist'} - set(value.keys()):
                self._set_connectivities_res(key, value)
                return
        elif (type(value) is np.ndarray or issparse(value)) and value.ndim >= 2:
            if value.shape == (self.__based_ann_data.n_obs, self.__based_ann_data.n_obs):
                self.__based_ann_data.obsp[key] = value
                return
            elif value.shape == (self.__based_ann_data.n_vars, self.__based_ann_data.n_vars):
                    self.__based_ann_data.varp[key] = value
                    return
            elif value.shape[0] == self.__based_ann_data.n_obs and value.shape[1] < self.__based_ann_data.n_vars:
                # TODO this is hard-code method to guess it's a reduce ndarray
                # self.__based_ann_data.obsm[key] = value
                self._set_reduce_res(key, value, start_with_X=False)
                return
            elif value.shape[0] == self.__based_ann_data.n_vars and value.shape[1] < self.__based_ann_data.n_obs:
                # TODO this is hard-code method to guess it's a reduce ndarray
                # self.__based_ann_data.varm[key] = value
                self._set_reduce_res(key, value)
                return
        elif type(value) is tuple:
            if len(value) == 2 and type(value[0]) is dict and \
                'counts' in value[0] and 'data' in value[0] and 'scale.data' in value[0]:
                self._set_sctransform_res(key, value)
                return

        self.__based_ann_data.uns[key] = value

    def _set_cluster_res(self, key, value):
        assert type(value) is pd.DataFrame and 'group' in value.columns.values, "this is not cluster res"
        # FIXME ignore set params to uns, this may cause dirty data in uns, if it exist at the first time
        self.__based_ann_data.uns[key] = {'params': {}, 'source': 'stereopy', 'method': key}
        self.__based_ann_data.obs[key] = value['group'].values

    def _set_connectivities_res(self, key, value):
        assert type(value) is dict and not {'connectivities', 'nn_dist'} - set(value.keys()), \
            'not enough key to set connectivities'
        self.__based_ann_data.uns[key] = {
            'params': {},
            'source': 'stereopy',
            'method': 'neighbors'
        }
        if 'params' in value:
            self.__based_ann_data.uns[key]['params'] = value['params']
        if key == 'neighbors':
            self.__based_ann_data.uns[key]['connectivities_key'] = 'connectivities'
            self.__based_ann_data.uns[key]['distances_key'] = 'distances'
            self.__based_ann_data.obsp['connectivities'] = value['connectivities']
            self.__based_ann_data.obsp['distances'] = value['nn_dist']
        else:
            self.__based_ann_data.uns[key]['connectivities_key'] = f'{key}_connectivities'
            self.__based_ann_data.uns[key]['distances_key'] = f'{key}_distances'
            self.__based_ann_data.obsp[f'{key}_connectivities'] = value['connectivities']
            self.__based_ann_data.obsp[f'{key}_distances'] = value['nn_dist']

    def _set_reduce_res(self, key, value, start_with_X=True):
        # assert type(value) is pd.DataFrame, 'reduce result must be pandas.DataFrame'
        if not isinstance(value, (pd.DataFrame, np.ndarray)):
            # raise TypeError('reduce result must be pandas.DataFrame or numpy.ndarray')
            return False
        if isinstance(value, pd.DataFrame):
            value = value.to_numpy(copy=True)
        if value.shape[0] == self.__based_ann_data.n_obs and value.shape[1] < self.__based_ann_data.n_vars:
            if start_with_X:
                self.__based_ann_data.uns[key] = {'params': {}, 'source': 'stereopy', 'method': key}
                self.__based_ann_data.obsm[f'X_{key}'] = value
            else:
                self.__based_ann_data.obsm[key] = value
            return True
        elif value.shape[0] == self.__based_ann_data.n_vars and value.shape[1] < self.__based_ann_data.n_obs:
            self.__based_ann_data.varm[key] = value
            return True
        else:
            return False

    def _set_hvg_res(self, key, value):
        self.__based_ann_data.uns[key] = {'params': {}, 'source': 'stereopy', 'method': key}
        self.__based_ann_data.var.loc[:, ["means", "dispersions", "dispersions_norm", "highly_variable"]] = \
            value.loc[:, ["means", "dispersions", "dispersions_norm", "highly_variable"]].to_numpy()
    
    def _get_marker_genes_res(self, name):
        if name in self.__based_ann_data.uns:
            marker_genes_result = self.__based_ann_data.uns[name]
        else:
            renamed = AnnBasedResult.RENAME_DICT.get(name, None)
            if renamed is None:
                return self.__based_ann_data.uns[name] # just for throwing an error.
            else:
                marker_genes_result = self.__based_ann_data.uns[renamed]
        marker_genes_result_reconstructed = {}
        if marker_genes_result['params']['method'] == 't-test':
            method = 't_test'
        elif marker_genes_result['params']['method'] == 'wilcoxon':
            method = 'wilcoxon_test'
        else:
            method = marker_genes_result['params']['method']
        marker_genes_result_reconstructed['parameters'] = {
            'cluster_res_key': marker_genes_result['params']['groupby'],
            'method': method,
            'control_groups': marker_genes_result['params']['reference'],
            'corr_method': marker_genes_result['params']['corr_method'],
            'use_raw': marker_genes_result['params']['use_raw'],
            'layer': marker_genes_result['params']['layer'] if 'layer' in marker_genes_result['params'] else None,
        }
        if 'marker_genes_res_key' in marker_genes_result['params']:
            marker_genes_result_reconstructed['parameters']['marker_genes_res_key'] = \
                                                                                marker_genes_result['params']['marker_genes_res_key']
        if 'pts' in marker_genes_result:
            marker_genes_result_reconstructed['pct'] = marker_genes_result['pts'].reset_index(names='genes')
            marker_genes_result_reconstructed['pct_rest'] = marker_genes_result['pts_rest'].reset_index(names='genes')
        if 'mean_count' in marker_genes_result:
            marker_genes_result_reconstructed['mean_count'] = marker_genes_result['mean_count']
        # clusters = self.__based_ann_data.obs[marker_genes_result['params']['groupby']].cat.categories
        clusters = marker_genes_result['names'].dtype.names
        key_map = {
            'scores': 'scores', 
            'pvalues': 'pvals',
            'pvalues_adj': 'pvals_adj',
            'log2fc': 'logfoldchanges',
            'genes': 'names',
        }
        control_groups = marker_genes_result_reconstructed['parameters']['control_groups']
        for c in clusters:
            df_data = {k1: marker_genes_result[k2][c] for k1, k2 in key_map.items()}
            df = pd.DataFrame(df_data)
            if 'real_gene_name' in self.__based_ann_data.var.columns:
                df['gene_name'] = self.__based_ann_data.var['real_gene_name'].loc[df['genes']].to_numpy()
            if 'pts' in marker_genes_result:
                df['pct'] = marker_genes_result['pts'][c].loc[df['genes']].to_numpy()
                df['pct_rest'] = marker_genes_result['pts_rest'][c].loc[df['genes']].to_numpy()
            if 'mean_count' in marker_genes_result:
                df['mean_count'] = marker_genes_result['mean_count'][c].loc[df['genes']].to_numpy()
            if isinstance(control_groups, (list, np.ndarray)):
                control_str = '-'.join([cg for cg in control_groups if cg != c])
            else:
                control_str = control_groups
            marker_genes_result_reconstructed[f'{c}.vs.{control_str}'] = df
        return marker_genes_result_reconstructed

    def _set_marker_genes_res(self, key, value):
        # self.__based_ann_data.uns[key] = value
        from stereo.io.utils import transform_marker_genes_to_anndata
        key = AnnBasedResult.RENAME_DICT.get(key, key)
        self.__based_ann_data.uns[key] = transform_marker_genes_to_anndata(value)
    
    def _get_sctransform_res(self, key):
        sct_key = self.__based_ann_data.uns[key]['sct_key']
        sct_id = self.__based_ann_data.uns[key]['id']
        sct_key_1 = f'{sct_key}_{sct_id}_1'
        sct_key_2 = f'{sct_key}_{sct_id}_2'
        res_1 = self.__based_ann_data.uns[sct_key_1]
        if isinstance(res_1['scale.data'], dict):
            res_1['scale.data'] = pd.DataFrame(
                res_1['scale.data']['data'], index=res_1['scale.data']['index'], columns=res_1['scale.data']['columns']
            )
        res_2 = self.__based_ann_data.uns[sct_key_2]
        return res_1, res_2
    
    def _set_sctransform_res(self, key, value):
        if isinstance(value, tuple) and len(value) == 2:
            sct_key_1 = f'{key}_{id(value)}_1'
            sct_key_2 = f'{key}_{id(value)}_2'
            self.__based_ann_data.uns[key] = {'sct_key': key, 'id': id(value)}
            self.__based_ann_data.uns[sct_key_1] = value[0]
            self.__based_ann_data.uns[sct_key_2] = value[1]
        else:
            raise ValueError('sctransform result must be a tuple with 2 elements')

    def set_value(self, key, value):
        if hasattr(value, 'shape'):
            # if (len(value.shape) >= 1) and (value.shape[0] == self.__based_ann_data.shape[0]):
            #     self.__based_ann_data.obsm[key] = value
            # elif (len(value.shape) >= 2) and (value.shape[1] == self.__based_ann_data.shape[1]):
            #     self.__based_ann_data.varm[key] = value
            if len(value.shape) >= 2:
                if value.shape[0:2] == (self.__based_ann_data.n_obs, self.__based_ann_data.n_obs):
                    self.__based_ann_data.obsp[key] = value
                elif value.shape[0:2] == (self.__based_ann_data.n_vars, self.__based_ann_data.n_vars):
                    self.__based_ann_data.varp[key] = value
                elif value.shape[0] == self.__based_ann_data.n_obs:
                    self.__based_ann_data.obsm[key] = value
                elif value.shape[0] == self.__based_ann_data.n_vars:
                    self.__based_ann_data.varm[key] = value
                else:
                    self.__based_ann_data.uns[key] = value
            else:
                self.__based_ann_data.uns[key] = value
        else:
            self.__based_ann_data.uns[key] = value
    
    def keys(self):
        return self.__result_keys


class MSDataPipeLineResult(dict):
    def __init__(self, _ms_data):
        self._ms_data = _ms_data
    
    def __getitem__(self, key):
        scope_key = self._ms_data.generate_scope_key(key)
        if scope_key in self._ms_data.scopes_data:
            return self._ms_data.scopes_data[scope_key].tl.result
        else:
            current_scope_key = self._ms_data.generate_scope_key(self._ms_data.tl.scope)
            if current_scope_key in self._ms_data.scopes_data:
                if key in self._ms_data.scopes_data[current_scope_key].tl.result:
                    return self._ms_data.scopes_data[current_scope_key].tl.result[key]
        return dict.__getitem__(self, key)
    
    def __contains__(self, key: object) -> bool:
        scope_key = self._ms_data.generate_scope_key(key)
        if scope_key in self._ms_data.tl.result_keys:
            return True
        else:
            return dict.__contains__(self, key)
    
    def __iter__(self):
        return iter(self.keys())
    
    def keys(self):
        super_keys = dict.keys(self)
        tmp = {}
        for key in super_keys:
            tmp[key] = None
        
        for key in self._ms_data.tl.result_keys.keys():
            tmp[key] = None
        
        return tmp.keys()