from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Union, Literal, Optional
from copy import deepcopy

import numpy as np
import pandas as pd

from . import StPipeline, StereoExpData
from .ms_pipeline import MSDataPipeLine
from ..plots.plot_collection import PlotCollection


def _default_idx() -> int:
    # return -1 function `__get_auto_key` will start with 0 instead of 1
    return -1


@dataclass
class _MSDataView(object):
    _msdata: MSData = None
    _names: List[str] = field(default_factory=list)
    _data_list: List[StereoExpData] = field(default_factory=list)
    _name_dict: Dict[str, StereoExpData] = field(default_factory=dict)
    _merged_data: StereoExpData = None
    _tl = None
    _plt = None

    def __post_init__(self):
        for name, data in zip(self._names, self._data_list):
            self._name_dict[name] = data 

    def __get_data_list(self, key_idx_list):
        data_list = []
        names = []
        for ki in key_idx_list:
            if isinstance(ki, (str, np.str_)):
                data_list.append(self._name_dict[ki])
                names.append(ki)
            elif isinstance(ki, (int, np.integer)):
                ki = int(ki)
                data_list.append(self._data_list[ki])
                names.append(self._names[ki])
            elif isinstance(ki, (list, tuple, np.ndarray, pd.Index)):
                temp_data_list, temp_names = self.__get_data_list(ki)
                data_list.extend(temp_data_list)
                names.extend(temp_names)
            else:
                raise KeyError(ki)
        return data_list, names
    
    def __check_slice(self, slice_obj: slice):
        if not isinstance(slice_obj, slice):
            raise TypeError(f'{slice_obj} should be slice')
        if slice_obj.start is not None and isinstance(slice_obj.start, (str, np.str_)):
            if slice_obj.start in self._name_dict:
                new_start = self._names.index(slice_obj.start)
            else:
                new_start = None
        else:
            new_start = slice_obj.start
        if slice_obj.stop is not None and isinstance(slice_obj.stop, (str, np.str_)):
            if slice_obj.stop in self._name_dict:
                new_stop = self._names.index(slice_obj.stop)
            else:
                new_stop = None
        else:
            new_stop = slice_obj.stop

        if slice_obj.step is not None and not isinstance(slice_obj.step, (int, np.integer)):
            raise TypeError(f'slice.step should be int')
        
        return slice(new_start, new_stop, slice_obj.step)
    

    def __getitem__(self, key: Union[str, int, list, tuple, np.ndarray, pd.Index, slice]) -> Union[StereoExpData, _MSDataView]:
        if isinstance(key, (str, np.str_)):
            return self._name_dict[key]
        elif isinstance(key, (int, np.integer)):
            return self._data_list[key]
        elif isinstance(key, (list, tuple, np.ndarray, pd.Index)):
            data_list, names = self.__get_data_list(key)
            return _MSDataView(_msdata=self._msdata, _data_list=data_list, _names=names)
        elif isinstance(key, slice):
            key = self.__check_slice(key)
            data_list = self._data_list[key]
            names = self._names[key]
            return _MSDataView(_msdata=self._msdata, _data_list=data_list, _names=names)
        else:
            raise KeyError(key)

    @property
    def tl(self):
        if self._tl is None:
            self._tl = TL(self)
        return self._tl

    @property
    def plt(self):
        if self._plt is None:
            self._plt = PLT(self)
        return self._plt

    @property
    def data_list(self):
        return self._data_list
    
    @property
    def names(self):
        return self._names
    
    @property
    def num_slice(self):
        return len(self._data_list)

    def __str__(self):
        return f'''data_list: {len(self._data_list)}'''
    
    def __len__(self):
        return len(self._data_list)

    @property
    def merged_data(self):
        if self._merged_data is None:
            self._merged_data = self._msdata.integrate(scope=self._names)
        return self._merged_data
    
    @merged_data.setter
    def merged_data(self, merged_data):
        self._merged_data = merged_data

    
    def to_msdata(self) -> MSData:
        return MSData(
            _data_list=deepcopy(self._data_list),
            _merged_data=deepcopy(self._merged_data),
            _names=deepcopy(self._names),
            _var_type=self._msdata._var_type,
            _relationship=self._msdata._relationship,
            _relationship_info=deepcopy(self._msdata._relationship_info)
        )


_NON_EDITABLE_ATTRS = {'data_list', 'names', '_obs', '_var', '_relationship', '_relationship_info'}
_RELATIONSHIP_ENUM = {'continuous', 'time_series', 'other'}


@dataclass
class _MSDataStruct(object):
    """
    `MSData` is a composite structure of several `StereoExpData` organized by some relationship.

    Parameters
    ----------
    data_list: List[StereoExpData] `stereo_exp_data` array
        An array of `stereo_exp_data` organized by some relationship defined by `_relationship` and `_relationship_info`

    merged_data: `stereo_exp_data` object
        An `stereo_exp_data` merged with `data_list` used batches integrate.

    names: List[str] `stereo_exp_data` array's names
        An array of `stereo_exp_data`s' unique names.

    obs: pandas.DataFrame = None
        `pandas.DataFrame` describes all the cells or bins observed, indexes mean cell names or bin names, columns mean
        some math statistic or types produced by bio-information algorithm.

    var: pd.DataFrame = None
        `pandas.DataFrame` describes genes, similar to `_obs`.

    _var_type: str = 'intersect'
        Which claims that `_var` is intersected by lots of `genes` from different samples.

    relationship: str = 'other'
        Relationship about samples in `_data_list`.

    _relationship_info: object
        Relationship extra info.

    tl: object
        `MSData` algorithms collections, include all tools methods inherited from `stereo_exp_data` and multi-samples
         methods. Methods from `stereo_exp_data` will organized with mutilthreads while running.

    plt: object
        `MSData` plot methods collections, same as `tl`.

    Examples
    --------
    Constructing MSData from two `stereo_exp_data`s.

    >>> from stereo.io.reader import read_gef
    >>> from stereo.core.ms_data import MSData
    >>> data1 = read_gef("../demo_data/SS200000135TL_D1/SS200000135TL_D1.gef")
    >>> data2 = read_gef("../demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gef")
    >>> ms_data = MSData(_data_list=[data1, data2], _names=['raw', 'tissue'], _relationship='other', _var_type='intersect') # noqa
    >>> ms_data

    ms_data: {'raw': (9004, 25523), 'tissue': (9111, 20816)}
    num_slice: 2
    names: ['raw', 'tissue']
    obs: ['test_obs_1']
    var: ['test_var_1']
    relationship: other
    var_type: intersect to 20760
    tl.result: defaultdict(<class 'list'>, {})

    Constructing MSData one by one using add method.

    >>> from stereo.core.ms_data import MSData
    >>> from stereo.io.reader import read_gef
    >>> ms_data = MSData()
    >>> ms_data += read_gef("../demo_data/SS200000135TL_D1/SS200000135TL_D1.gef")
    >>> ms_data += read_gef("../demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gef")
    >>> ms_data
    ms_data: {'0': (9004, 25523), '1': (9111, 20816)}
    num_slice: 2
    names: ['0', '1']
    obs: ['test_obs_1']
    var: ['test_var_1']
    relationship: other
    var_type: intersect to 20760
    tl.result: defaultdict(<class 'list'>, {})

    Slice features like python list.

    >>> from stereo.core.ms_data import MSData
    >>> from stereo.io.reader import read_gef
    >>> ms_data = MSData()
    >>> ms_data += read_gef("../demo_data/SS200000135TL_D1/SS200000135TL_D1.gef")
    >>> ms_data += read_gef("../demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gef")
    >>> ms_data += read_gef("../demo_data/SS200000135TL_D1/SS200000135TL_D1.gef")
    >>> ms_data += read_gef("../demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gef")
    >>> ms_data += read_gef("../demo_data/SS200000135TL_D1/SS200000135TL_D1.gef")
    >>> ms_data += read_gef("../demo_data/SS200000135TL_D1/SS200000135TL_D1.tissue.gef")
    >>> ms_data[3:]
    _MSDataView(_names=['3', '4', '5'], _data_list=[StereoExpData object with n_cells X n_genes = 9111 X 20816
    bin_type: bins
    bin_size: 100
    offset_x = 0
    offset_y = 2
    cells: ['cell_name']
    genes: ['gene_name'], ...)

    Slice features like DataFrame.

    >>> ms_data[('1', '4'):]
    _MSDataView(_names=['1', '4'], _data_list=[StereoExpData object with n_cells X n_genes = 9111 X 20816
    bin_type: bins
    bin_size: 100
    offset_x = 0
    offset_y = 2
    cells: ['cell_name']
    genes: ['gene_name'], ...)
    """

    # base attributes
    # TODO temporarily length 10
    _data_list: List[StereoExpData] = field(default_factory=list)
    _merged_data: StereoExpData = None
    _names: List[str] = field(default_factory=list)
    _obs: pd.DataFrame = None
    _var: pd.DataFrame = None
    _var_type: str = 'intersect'
    _relationship: str = 'other'
    # TODO not define yet
    _relationship_info: dict = field(default_factory=dict)

    # code-supported attributes
    _name_dict: Dict[str, StereoExpData] = field(default_factory=dict)
    _data_dict: Dict[int, str] = field(default_factory=dict)
    __idx_generator: int = _default_idx()

    def __post_init__(self) -> object:
        while len(self._data_list) > len(self._names):
            self._names.append(self.__get_auto_key())
        if not self._name_dict or not self._data_dict:
            self.reset_name(default_key=False)
        return self

    @property
    def data_list(self):
        return self._data_list

    @property
    def merged_data(self):
        return self._merged_data

    @merged_data.setter
    def merged_data(self, value: StereoExpData):
        self._merged_data = value

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value: List[str]):
        if len(value) != len(self._data_list):
            raise Exception('new names\' length should be same as data_list')
        self._names = list(value)
        self.reset_name(default_key=False)
    
    @property
    def var_type(self):
        return self._var_type
    
    @var_type.setter
    def var_type(self, value: str):
        if value not in {'intersect', 'union'}:
            raise Exception(f'new var_type must be in {"intersect", "union"}')
        self._var_type = value

    @property
    def relationship(self):
        return self._relationship

    @relationship.setter
    def relationship(self, value: str):
        if value not in _RELATIONSHIP_ENUM:
            raise Exception(f'new relationship must be in {_RELATIONSHIP_ENUM}')
        self._relationship = value

    @property
    def relationship_info(self):
        return self._relationship_info
    
    @relationship_info.setter
    def relationship_info(self, value: dict):
        self._relationship_info = value

    def reset_position(self, mode='integrate'):
        if mode == 'integrate' and self.merged_data:
            self.merged_data.reset_position()
        else:
            for data in self.data_list:
                data.reset_position()

    def __len__(self):
        return len(self._data_list)

    def __copy__(self) -> object:
        # TODO temporarily can not copy、deepcopy, return self
        return self

    def __deepcopy__(self, _) -> object:
        return self

    def get_data_list(self, key_idx_list):
        data_list = []
        names = []
        for ki in key_idx_list:
            if isinstance(ki, (str, np.str_)):
                data_list.append(self._name_dict[ki])
                names.append(ki)
            elif isinstance(ki, (int, np.integer)):
                ki = int(ki)
                data_list.append(self._data_list[ki])
                names.append(self._names[ki])
            elif isinstance(ki, (list, tuple, np.ndarray, pd.Index)):
                temp_data_list, temp_names = self.get_data_list(ki)
                data_list.extend(temp_data_list)
                names.extend(temp_names)
            else:
                raise KeyError(ki)
        return data_list, names
    
    def __check_slice(self, slice_obj: slice):
        if not isinstance(slice_obj, slice):
            raise TypeError(f'{slice_obj} should be slice')
        if slice_obj.start is not None and isinstance(slice_obj.start, (str, np.str_)):
            if slice_obj.start in self._name_dict:
                new_start = self._names.index(slice_obj.start)
            else:
                new_start = None
        else:
            new_start = slice_obj.start
        if slice_obj.stop is not None and isinstance(slice_obj.stop, (str, np.str_)):
            if slice_obj.stop in self._name_dict:
                new_stop = self._names.index(slice_obj.stop)
            else:
                new_stop = None
        else:
            new_stop = slice_obj.stop

        if slice_obj.step is not None and not isinstance(slice_obj.step, (int, np.integer)):
            raise TypeError(f'slice.step should be int')
        
        return slice(new_start, new_stop, slice_obj.step)

    def __getitem__(self, key: Union[str, int, list, tuple, np.ndarray, pd.Index, slice]) -> Union[StereoExpData, _MSDataView]:
        if isinstance(key, (str, np.str_)):
            return self._name_dict[key]
        elif isinstance(key, (int, np.integer)):
            return self._data_list[key]
        elif isinstance(key, (list, tuple, np.ndarray, pd.Index)):
            data_list, names = self.get_data_list(key)
            return _MSDataView(_msdata=self, _data_list=data_list, _names=names)
        elif isinstance(key, slice):
            key = self.__check_slice(key)
            data_list = self._data_list[key]
            names = self._names[key]
            return _MSDataView(_msdata=self, _data_list=data_list, _names=names)
        else:
            raise KeyError(key)
        

    def __setitem__(self, key, value):
        assert isinstance(key, (int, np.integer, str, np.str_))
        assert isinstance(value, StereoExpData)

        if key in self._name_dict:
            key = self._names.index(key)
        if isinstance(key, (int, np.integer)):
            if key >= self.num_slice:
                raise IndexError("list index out of range")
            old_obj = self._data_list[key]
            name = self._names[key]
            self._data_list[key] = value
            self._name_dict[name] = value
            self._data_dict.pop(id(old_obj))
            self._data_dict[id(value)] = name
        else:
            self.__real_add(value, key)

    def __add__(self, other):
        assert isinstance(other, StereoExpData)
        self.__real_add(other)
        return self

    def __contains__(self, item) -> bool:
        if type(item) is str:
            return item in self._name_dict
        elif isinstance(item, StereoExpData):
            return id(item) in self._data_dict
        else:
            raise TypeError('In-Expression: only supports `name` or `StereoExpData-object`')

    def add_data(self, data=None, names=None, **kwargs) -> object:
        if not data:
            raise Exception
        if isinstance(data, StereoExpData):
            return self.__add_data_objs([data], [names] if names else None)
        elif type(data) is str:
            return self.__add_data_paths([data], [names] if names else None, **kwargs)
        elif type(data) is list:
            if len(data) != len(names):
                raise Exception('length of objs and length of keys must equal')
            if isinstance(data[0], StereoExpData):
                return self.__add_data_objs(data, names)
            elif type(data[0]) is str:
                return self.__add_data_paths(data, names, **kwargs)
        raise TypeError

    def del_data(self, name):
        obj = self._name_dict.pop(name)
        self._data_list.index(obj)
        self._names.remove(name)
        self._data_dict.pop(id(obj))
        self._data_list.remove(obj)

    def __delitem__(self, key):
        self.del_data(key)

    def __add_data_objs(self, data_list: List[StereoExpData], keys: List[str] = None) -> object:
        if keys:
            for key in keys:
                if key in self._names:
                    raise KeyError(f'key={key} already exists')
        for data_obj in data_list:
            if data_obj in self:
                raise Exception
        for idx, data_obj in enumerate(data_list):
            self.__real_add(data_obj, keys[idx] if keys and idx < len(keys) else None)
        return self

    def __add_data_paths(self, file_path_list: List[str], keys: List[str] = None, **kwargs) -> object:
        from stereo.io.reader import read_gef, read_gem, read_ann_h5ad
        data_list = []
        # TODO mixed file format, how to handle arguments
        bin_sizes = kwargs.get('bin_size', None)
        bin_types = kwargs.get('bin_type', None)
        spatial_keys = kwargs.get('spatial_key', None)
        if bin_sizes is not None or bin_types is not None:
            assert len(file_path_list) == len(bin_sizes) == len(bin_types)
        for idx, file_path in enumerate(file_path_list):
            if file_path.endswith('.gef'):
                data_list.append(read_gef(
                    file_path,
                    bin_size=bin_sizes[idx] if bin_sizes is not None else 100,
                    bin_type=bin_types[idx] if bin_types is not None else 'bins',
                ))
            elif file_path.endswith('.gem') or file_path.endswith('.gem.gz'):
                data_list.append(read_gem(
                    file_path,
                    bin_size=bin_sizes[idx] if bin_sizes is not None else 100,
                    bin_type=bin_types[idx] if bin_types is not None else 'bins',
                ))
            elif file_path.endswith('.h5ad'):
                data_list.append(read_ann_h5ad(
                    file_path,
                    spatial_key=spatial_keys[idx],
                    bin_size=bin_sizes[idx],
                    bin_type=bin_types[idx],
                ))
            else:
                raise Exception(f'file format({file_path}) not support')
        return self.__add_data_objs(data_list, keys)

    def __get_auto_key(self) -> str:
        self.__idx_generator += 1
        return str(f'{self.__idx_generator}')

    def __real_add(self, obj: StereoExpData, key: Union[str, None] = None) -> object:
        if not key:
            key = self.__get_auto_key()
            while key in self._name_dict:
                key = self.__get_auto_key()
        self._name_dict[key] = obj
        self._data_dict[id(obj)] = key
        self._names.append(key)
        self._data_list.append(obj)
        return self

    @property
    def obs(self) -> pd.DataFrame:
        if self._merged_data:
            return self._merged_data.cells.to_df()
        return pd.DataFrame()

    @property
    def var(self) -> pd.DataFrame:
        if self.merged_data:
            return self._merged_data.genes.to_df()
        return pd.DataFrame()

    def var_percent(self):
        percent_list = []
        var_set = set(self.var.index)
        for data_obj in self._data_list:
            count = 0
            for gene in data_obj.gene_names:
                if gene in var_set:
                    count += 1
            percent_list.append(count / len(data_obj.gene_names))
        return percent_list

    @property
    def shape(self) -> dict:
        return dict(zip(self._names, [data_obj.shape for data_obj in self._data_list]))

    @property
    def num_slice(self):
        return len(self)

    def rename(self, mapper: Dict[str, str]) -> object:
        # if len(rename_keys) is m, and len(self._data_dict) is n, method time complexity:
        # O(2 * (n * m + n + m))
        if not mapper:
            raise Exception('`rename_keys` is empty or None')
        elif not self._name_dict:
            raise Exception('`ms_data` is empty')
        mapper_values = mapper.values()
        set_of_src = set(mapper_values)
        if len(set_of_src) != len(mapper_values):
            raise Exception('`rename_keys` with values target at same obj')
        # avoid circle-renaming, we only support rename to a new key
        intersection_src_keys = mapper_values & self._name_dict.keys()
        if intersection_src_keys:
            raise Exception(f'{intersection_src_keys} already exists, can not rename to!')
        # allow intersection_src_keys being empty
        intersection_dst_keys = mapper.keys() & self._name_dict.keys()
        if len(intersection_dst_keys) != len(mapper.keys()):
            raise Exception(f'some keys in {mapper.keys()} not exist in ms_data')
        for src in intersection_dst_keys:
            dst = mapper[src]
            src_obj = self._name_dict.pop(src)
            self._name_dict[dst] = src_obj
            self._data_dict[id(src_obj)] = dst
        self._names = []
        for obj in self._data_list:
            self._names.append(self._data_dict[id(obj)])
        return self

    def reset_name(self, start_idx=None, default_key=True) -> object:
        # if self.data_list is n, O(3n)
        self.__idx_generator = _default_idx() if start_idx is None else start_idx
        self._name_dict, self._data_dict = dict(), dict()
        for idx, obj in enumerate(self._data_list):
            name = self.__get_auto_key() if default_key else self._names[idx]
            self._name_dict[name] = obj
            self._data_dict[id(obj)] = name
            self._names[idx] = name
        if len(self._data_list) < len(self._names):
            self._names = self._names[0:len(self._data_list)]
        return self
    
class ScopesData(dict):
    def __init__(self, ms_data: MSData, *args, **kwargs):
        self._ms_data = ms_data
        super().__init__(*args, **kwargs)
    
    def __setitem__(self, key, value):
        if not isinstance(value, StereoExpData):
            raise TypeError(f'value must be a StereoExpData object')
        
        def set_result_key_method(result_key):
            self._ms_data.tl.result_keys.setdefault(key, [])
            if result_key in self._ms_data.tl.result_keys[key]:
                self._ms_data.tl.result_keys[key].remove(result_key)
            self._ms_data.tl.result_keys[key].append(result_key)
        value.tl.result.set_result_key_method = set_result_key_method

        return super().__setitem__(key, value)


@dataclass
class MSData(_MSDataStruct):
    __doc__ = _MSDataStruct.__doc__

    _tl = None
    _plt = None
    _scopes_data: Dict[str, StereoExpData] = None

    def __post_init__(self) -> object:
        if self._scopes_data is None:
            self._scopes_data = ScopesData(self)
        else:
            self._scopes_data = self.__reset_scopes_data(self._scopes_data)
        super().__post_init__()
        return self

    def __reset_scopes_data(self, value):
        if not isinstance(value, dict):
            raise TypeError(f'value must be a dict object')
        if not isinstance(value, ScopesData):
            scopes_data = ScopesData(self)
            for scope_key, scope_data in value.items():
                scopes_data[scope_key] = scope_data
        else:
            scopes_data = value
        return scopes_data

    @property
    def tl(self):
        if self._tl is None:
            self._tl = TL(self)
        return self._tl

    @property
    def plt(self):
        if self._plt is None:
            self._plt = PLT(self)
        return self._plt
    
    @property
    def scopes_data(self):
        return self._scopes_data
    
    @scopes_data.setter
    def scopes_data(self, value):
        self._scopes_data = self.__reset_scopes_data(value)

    @property
    def mss(self):
        return self.tl.result

    def generate_scope_key(self, scope=None):
        if scope is None:
            scope = slice(None)
        scope_key = scope
        try:
            if isinstance(scope, (int, np.integer)):
                scope_key = f"scope_[{scope}]"
            elif isinstance(scope, (str, np.str_)):
                if scope in self._name_dict:
                    scope_key = f"scope_[{self._names.index(scope)}]"
                else:
                    scope_key = scope
            elif isinstance(scope, slice):
                names = self[scope]._names
                scope_key = f"scope_[{','.join([str(self._names.index(name)) for name in names])}]"  # noqa
            elif isinstance(scope, (list, tuple, np.ndarray, pd.Index)):
                _, names = self.get_data_list(scope)
                scope_key = f"scope_[{','.join([str(self._names.index(name)) for name in names])}]"  # noqa
        except:
            scope_key = scope
        finally:
            return scope_key
    
    def remove_scopes_data(self, scope):
        scope_key = self.generate_scope_key(scope)
        if scope_key in self._scopes_data:
            del self._scopes_data[scope_key]
        if scope_key in self.tl.result_keys:
            del self.tl.result_keys[scope_key]


    def integrate(self, scope=None, remove_existed=False, **kwargs):
        """
        Integrate some single-samples specified by `scope` to a merged one.
        
        :param scope: Which scope of samples to be integrated, defaults to None.
                        Each integrate sample is saved in memory, performing this function
                        by passing duplicate `scope` will return the saved one.
        :param remove_existed: Whether to remove the saved integrate sample when passing a duplicate `scope`, defaults to False.

        """
        from stereo.utils.data_helper import merge
        if self._var_type not in {"union", "intersect"}:
            raise Exception("Please specify the operation on samples with the parameter '_var_type'")
        
        if 'var_type' in kwargs:
            del kwargs['var_type']
        if 'batch_tags' in kwargs:
            del kwargs['batch_tags']
        
        if remove_existed:
            self.remove_scopes_data(scope)
        scope_key = self.generate_scope_key(scope)
        if scope_key in self._scopes_data:
            return self._scopes_data[scope_key]
        
        if scope == None:
            data_list = self.data_list
        else:
            data_list = self[scope].data_list
        if len(data_list) > 1:
            if scope is None:
                batch_tags = None
            else:
                batch_tags = [self._names.index(name) for name in self[scope].names]
            merged_data = merge(*data_list, var_type=self._var_type, batch_tags=batch_tags)
        else:
            merged_data = deepcopy(data_list[0])
            batch = self._names.index(self[scope].names[0])
            merged_data.cells.cell_name = np.char.add(merged_data.cells.cell_name, f'-{batch}')
            merged_data.cells.batch = batch
        
        obs_columns = merged_data.cells._obs.columns.drop('batch')
        if len(obs_columns) > 0:
            merged_data.cells._obs.drop(columns=obs_columns, inplace=True)
        var_columns = merged_data.genes._var.columns
        if 'real_gene_name' in var_columns:
            var_columns = var_columns.drop('real_gene_name')
        if len(var_columns) > 0:
            merged_data.genes._var.drop(columns=var_columns, inplace=True)
        
        # def set_result_key_method(key):
        #     self.tl.result_keys.setdefault(scope_key, [])
        #     if key in self.tl.result_keys[scope_key]:
        #         self.tl.result_keys[scope_key].remove(key)
        #     self.tl.result_keys[scope_key].append(key)
        
        # merged_data.tl.result.set_result_key_method = set_result_key_method
        
        scope_key = self.generate_scope_key(scope)
        self._scopes_data[scope_key] = merged_data

        if scope == None or scope == slice(None):
            self._merged_data = merged_data

        return merged_data


    def split_after_batching_integrate(self):
        if self._var_type == "union":
            raise NotImplementedError("Split a union data not implemented yet")
        from stereo.utils.data_helper import split
        self._data_list = split(self.merged_data)
        self.reset_name(default_key=False)
        self.merged_data = None

    def to_integrate(
            self,
            scope: slice,
            res_key: str,
            _from: slice,
            type: Literal['obs', 'var'] = 'obs',
            item: Optional[Union[list, np.ndarray, str]] = None,
            fill=np.NaN,
            cluster: bool = True
    ):
        """
        Integrate an obs column or a var column from some single-samples spcified by `_from` to the merged sample. 

        :param scope: Which integrate mss group to save result.
        :param res_key: New column name in merged sample obs or var.
        :param _from: Where to get the single-sample target infomation.
        :param type: obs or var level, defaults to 'obs'.
        :param item: The column names in single-sample obs or var, defaults to the value of `res_key`.
        :param fill: Default value when the merged sample has no conrresponding item, defaults to np.NaN.
        :param cluster: Whether it is a clustering result, defaults to True.

        .. note::

            The length of `scope` must be equal to `_from`.

            The `type` just only supports 'obs' currently.
        
        Examples
        --------
        Constructing MSData from 5 single-samples.

        >>> import stereo as st
        >>> data1 = st.io.read_h5ad('../data/10.h5ad')
        >>> data2 = st.io.read_h5ad('../data/11.h5ad')
        >>> data3 = st.io.read_h5ad('../data/12.h5ad')
        >>> data4 = st.io.read_h5ad('../data/13.h5ad')
        >>> data5 = st.io.read_h5ad('../data/14.h5ad')
        >>> ms_data = data1 + data2 + data3 + data4 + data5
        >>> ms_data
        ms_data: {'0': (493, 30254), '1': (285, 30254), '2': (753, 30254), '3': (731, 30254), '4': (412, 30254)}
        num_slice: 5
        names: ['0', '1', '2', '3', '4']
        obs: []
        var: []
        relationship: other
        var_type: intersect to 0
        mss: []
        
        Integrating all samples to a merged one.

        >>> ms_data.integrate()
        
        Integrating an obs column named as 'celltype' from first three samples to the merged sample, to name as 'celltype'

        >>> from stereo.core.ms_pipeline import slice_generator
        >>> ms_data.to_integrate(res_key='celltype', scope=slice_generator[0:3], _from=slice_generator[0:3], type='obs', item=['celltype'] * 3)

        Integrating an obs column named as 'celltype' from all samples to the merged sample, to name as 'celltype'

        >>> from stereo.core.ms_pipeline import slice_generator
        >>> ms_data.to_integrate(res_key='celltype', scope=slice_generator[:], _from=slice_generator[:], type='obs', item=['celltype'] * ms_data.num_slice)

        """
        assert self[scope]._names == self[_from]._names, f"`scope`: {scope} should equal with _from: {_from}"
        assert isinstance(item, str) or len(item) == len(self[_from]._names), "`item`'s length not equal to _from"
        scope_names = self[scope]._names
        scope_key = self.generate_scope_key(scope_names)
        assert scope_key in self._scopes_data or self._merged_data, f"`to_integrate` need running function `integrate` first"
        if type == 'obs':
            if scope_key in self._scopes_data:
                self._scopes_data[scope_key].cells[res_key] = fill
            
            if self._merged_data is not None:
                self._merged_data.cells[res_key] = fill
        elif type == 'var':
            raise NotImplementedError
        else:
            raise Exception(f"`type`: {type} not in ['obs', 'var'], this should not happens!")
        
        data_list = self[scope]._data_list
        if item is None:
            item = res_key
        if isinstance(item, str):
            item = [item] * len(data_list)
        for idx, stereo_exp_data in enumerate(data_list):
            if type == 'obs':
                res: pd.Series = stereo_exp_data.cells[item[idx]]
                sample_idx = self._names.index(scope_names[idx])
                new_index: pd.Series = res.index.astype('str') + f'-{sample_idx}'
                # res.index = new_index
                if scope_key in self._scopes_data:
                    index_intersect = np.intersect1d(new_index, self._scopes_data[scope_key].cell_names)
                    # isin = np.isin(new_index, index_intersect)
                    isin = new_index.isin(index_intersect)
                    _res = res[isin].to_numpy()
                    _index = new_index[isin]
                    self._scopes_data[scope_key].cells.loc[_index, res_key] = _res
                if self._scopes_data[scope_key] is self._merged_data:
                    continue
                if self._merged_data is not None:
                    index_intersect = np.intersect1d(new_index, self._merged_data.cell_names)
                    # isin = np.isin(new_index, index_intersect)
                    isin = new_index.isin(index_intersect)
                    _res = res[isin].to_numpy()
                    _index = new_index[isin]
                    self._merged_data.cells.loc[_index, res_key] = _res
            elif type == 'var':
                raise NotImplementedError
            else:
                raise Exception(f"`type`: {type} not in ['obs', 'var'], this should not happens!")
        if type == 'obs':
            if cluster:
                if scope_key in self._scopes_data:
                    self._scopes_data[scope_key].tl.reset_key_record('cluster', res_key)
                    self._scopes_data[scope_key].tl.result.set_result_key_method(res_key)
                    self._scopes_data[scope_key].cells[res_key] = self._scopes_data[scope_key].cells[res_key].astype('category')

                if self._merged_data is not None and self._merged_data is not self._scopes_data[scope_key]:
                    self._merged_data.tl.reset_key_record('cluster', res_key)
                    self._merged_data.tl.result.set_result_key_method(res_key)
                    self._merged_data.cells[res_key] = self._merged_data.cells[res_key].astype('category')
        elif type == 'var':
            raise NotImplementedError
        else:
            raise Exception(f"`type`: {type} not in ['obs', 'var'], this should not happens!")

    def to_isolated(
            self,
            scope: slice,
            res_key: str,
            to: slice,
            type: Literal['obs', 'var'] = 'obs',
            item: Optional[Union[list, np.ndarray, str]] = None,
            fill=np.NaN
    ):
        """
        Copy a result from mss group specfied by scope to some single-samples specfied by `to`.

        :param scope: Which integrate mss group to get result.
        :param res_key: the key to get result from mms group.
        :param to: which single-samples are the result copy to.
        :param type: obs or var level, defaults to 'obs'
        :param item: New column name in obs of single-sample, defaults to the value of `res_key`.
        :param fill: Default value when the single-sample has no conrresponding item, defaults to np.NaN

        .. note::

            The length of `scope` must be equal to `to`.
            
            Only supports clustering result when `type` is 'obs' and hvg result when `type` is 'var'.

            Parameter `item` only available for obs type.
        
        Examples
        --------
        Constructing MSData from 5 single-samples.

        >>> import stereo as st
        >>> data1 = st.io.read_h5ad('../data/10.h5ad')
        >>> data2 = st.io.read_h5ad('../data/11.h5ad')
        >>> data3 = st.io.read_h5ad('../data/12.h5ad')
        >>> data4 = st.io.read_h5ad('../data/13.h5ad')
        >>> data5 = st.io.read_h5ad('../data/14.h5ad')
        >>> ms_data = data1 + data2 + data3 + data4 + data5
        >>> ms_data
        ms_data: {'0': (493, 30254), '1': (285, 30254), '2': (753, 30254), '3': (731, 30254), '4': (412, 30254)}
        num_slice: 5
        names: ['0', '1', '2', '3', '4']
        obs: []
        var: []
        relationship: other
        var_type: intersect to 0
        mss: []
        
        Integrating all samples to a merged one.

        >>> ms_data.integrate()

        ... did a clustering, the key of result is 'leiden' ...

        Copy the 'leiden' result to first three samples, to name as 'leiden'.

        >>> from stereo.core.ms_pipeline import slice_generator
        >>> ms_data.to_isolated(scope=slice_generator[0:3], res_key='leiden', to=slice_generator[0:3], type='obs', item=['leiden'] * 3)


        Copy the 'leiden' result to all samples, to name as 'leiden'.

        >>> from stereo.core.ms_pipeline import slice_generator
        >>> ms_data.to_isolated(scope=slice_generator[:], res_key='leiden', to=slice_generator[:], type='obs', item=['leiden'] * 3)
        
        
        """
        assert self[scope]._names == self[to]._names, f"`scope`: {scope} should equal with to: {to}"
        assert isinstance(item, str) or len(item) == len(self[to]._names), "`item`'s length not equal to `to`"

        scope_names = self[scope]._names
        scope_key = self.generate_scope_key(scope_names)
        merged_res: pd.DataFrame = self.tl.result[scope_key][res_key].copy(deep=True)
        if type == "obs":
            # TODO: only support cluster data
            if "bins" not in merged_res.columns or "group" not in merged_res.columns:
                raise Exception("Only soupport cluster result currently.")
            merged_res.set_index('bins', inplace=True)
        elif type == "var":
            # TODO: only support hvg data
            merged_res.index = self._scopes_data[scope_key].genes.gene_name

        data_list = self[scope]._data_list
        if item is None:
            item = res_key
        if isinstance(item, str):
            item = [item] * len(data_list)
        for idx, stereo_exp_data in enumerate(data_list):
            if type == 'obs':
                column_name = item[idx]
                original_index = stereo_exp_data.cells._obs.index
                stereo_exp_data.cells._obs.index = np.char.add(
                    np.char.add(stereo_exp_data.cells._obs.index.to_numpy().astype('U'), '-'),
                    stereo_exp_data.cells['batch']
                )
                stereo_exp_data.cells._obs[column_name] = merged_res['group']
                
                if fill is not np.NaN:
                    if stereo_exp_data.cells._obs[column_name].dtype.name == 'category':
                        stereo_exp_data.cells._obs[column_name].cat.add_categories(fill, inplace=True)
                    stereo_exp_data.cells._obs[column_name].fillna(fill, inplace=True)
                if stereo_exp_data.cells._obs[column_name].dtype.name == 'category':
                    stereo_exp_data.cells._obs[column_name].cat.remove_unused_categories(inplace=True)
                stereo_exp_data.cells._obs.index = original_index
            elif type == 'var':
                intersect = np.intersect1d(stereo_exp_data.genes.gene_name, merged_res.index)
                result_df = pd.DataFrame(
                    fill, index=stereo_exp_data.genes.gene_name, columns=merged_res.columns
                )
                for column in merged_res.columns:
                    if merged_res[column].dtype is np.dtype(bool):
                        result_df[column] = False
                    result_df.loc[intersect, column] = merged_res.loc[intersect, column]
                stereo_exp_data.tl.result[item[idx]] = result_df
                stereo_exp_data.tl.reset_key_record('hvg', item[idx])
            else:
                raise Exception(f"`type`: {type} not in ['obs', 'var'], this should not happens!")
            
    @staticmethod
    def to_msdata(
        data: StereoExpData,
        batch_key: str = 'batch',
        relationship: Optional[str] = 'other',
        var_type: Optional[str] = 'intersect'
    ):
        if batch_key not in data.cells:
            raise KeyError(f"The batch key '{batch_key}' is not in cells or obs.")
        
        from stereo.preprocess.filter import filter_by_clusters
        batch_data = pd.DataFrame({
            'bins': data.cells.cell_name,
            'group': data.cells._obs[batch_key].astype('category')
        })

        sub_data_list = []
        sub_data_names = []
        for batch_code in batch_data['group'].cat.categories:
            sub_data, _ = filter_by_clusters(data, batch_data, groups=batch_code, inplace=False)
            sub_data_list.append(sub_data)
            sub_data_names.append(batch_code)
        
        return MSData(_data_list=sub_data_list, _names=sub_data_names, _relationship=relationship, _var_type=var_type)

    def __str__(self):
        return f'''ms_data: {self.shape}
num_slice: {self.num_slice}
names: {self.names}
merged_data: {None if self._merged_data is None else f"id({id(self._merged_data)})"}
obs: {self.obs.columns.to_list()}
var: {self.var.columns.to_list()}
relationship: {self.relationship}
var_type: {self._var_type} to {len(self.var.index)}
current_mode: {self.tl.mode}
current_scope: {self.generate_scope_key(self.tl.scope)}
scopes_data: {[key + ":" + f"id({id(value)})" for key, value in self._scopes_data.items()]}
mss: {[key + ":" + str(value) for key, value in self.tl.result_keys.items()]}
'''

    def __repr__(self):
        return self.__str__()
    
    def write(self, filename, to_mudata=False):
        if not to_mudata:
            from stereo.io.writer import write_h5ms
            write_h5ms(self, filename)
        else:
            from stereo.io.writer import write_h5mu
            return write_h5mu(self, filename)


TL = type('TL', (MSDataPipeLine,), {'ATTR_NAME': 'tl', "BASE_CLASS": StPipeline})
PLT = type('PLT', (MSDataPipeLine,), {'ATTR_NAME': 'plt', "BASE_CLASS": PlotCollection})
