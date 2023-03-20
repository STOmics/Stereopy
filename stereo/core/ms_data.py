from collections import defaultdict
from typing import List, Dict, Union
from dataclasses import dataclass, field

import pandas as pd
from joblib import Parallel, cpu_count, delayed

from ..log_manager import logger
from . import StPipeline, StereoExpData
from ..plots.plot_collection import PlotCollection


def _default_idx() -> int:
    # return -1 function `__get_auto_key` will start with 0 instead of 1
    return -1


@dataclass
class _MSDataView:
    _data_list: List[StereoExpData] = field(default_factory=list)
    _tl = None
    _plt = None

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

    def __str__(self):
        return f'''data_list: {len(self._data_list)}'''


@dataclass
class _MSDataStruct(object):
    # 最重要的思想：
    # 1 `_MSDataStruct` 只索引、组织使用逻辑，不缓存样本数据，真正的数据在StereoExpData
    # 2 内存：懒加载; 运行时间：代码质量、numba、GPU; 

    # TODO 单片增删改细胞、基因，会影响obs、var
    # 1 参考AnnData
    # 2 filter_genes\filter_cells

    # base attributes
    # TODO: temporarily length 10
    _data_list: List[StereoExpData] = field(default_factory=list)
    _names: List[str] = field(default_factory=list)
    _obs: pd.DataFrame = None
    _var: pd.DataFrame = None
    _relationship: str = 'other'
    _relationship_info: object = None  # TODO not define yet

    # code-supported attributes
    _name_dict: Dict[str, StereoExpData] = field(default_factory=dict)
    _data_dict: Dict[int, str] = field(default_factory=dict)
    __idx_generator: int = _default_idx()
    __reconstruct: set = field(default_factory=set)

    # class attr
    _NON_EDITABLE_ATTRS = {'data_list', 'names', '_obs', '_var', '_relationship', '_relationship_info'}
    RELATIONSHIP_ENUM = {'continuous', 'time_series', 'other'}

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
    def names(self):
        return self._names

    @names.setter
    def names(self, value: List[str]):
        if len(value) != len(self._data_list):
            raise Exception(f'new names\' length should be same as data_list')
        self._names = value
        self.reset_name(default_key=False)

    @property
    def relationship(self):
        return self._relationship

    @relationship.setter
    def relationship(self, value: str):
        if value not in MSData.RELATIONSHIP_ENUM:
            raise Exception(f'new relationship must be in {MSData.RELATIONSHIP_ENUM}')
        self._relationship = value

    def __len__(self):
        return len(self._data_list)

    def __copy__(self) -> object:
        # TODO: temporarily can not copy、deepcopy, return self
        return self

    def __deepcopy__(self, _) -> object:
        return self

    def __getitem__(self, key: Union[str, int, slice]) -> Union[StereoExpData, _MSDataView]:
        if type(key) is int:
            idx = key
            return self._data_list[idx]
        elif type(key) is str:
            return self._name_dict[key]
        elif type(key) is slice:
            data_list = []
            if type(key.start) is tuple or type(key.start) is list:
                for obj_key in key.start:
                    data_list.append(self._name_dict[obj_key])
            elif type(key.start) is int:
                data_list = self._data_list[key]
            return _MSDataView(_data_list=data_list)
        raise TypeError(f'{key} is not one of Union[str, int]')

    def __contains__(self, item) -> bool:
        if type(item) is str:
            return item in self._name_dict
        elif isinstance(item, StereoExpData):
            return id(item) in self._data_dict
        else:
            raise TypeError('In-Expression: only supports `name` or `StereoExpData-object`')

    def add_data(self, objs, keys=None, **kwargs) -> object:
        if not objs:
            raise Exception
        if isinstance(objs, StereoExpData):
            return self.__add_data_objs([objs], [keys] if keys else None)
        elif type(objs) is str:
            return self.__add_data_paths([objs], [keys] if keys else None, **kwargs)
        elif type(objs) is list:
            if len(objs) != len(keys):
                raise Exception('length of objs and length of keys must equal')
            if isinstance(objs[0], StereoExpData):
                return self.__add_data_objs(objs, keys)
            elif type(objs[0]) is str:
                return self.__add_data_paths(objs, keys, **kwargs)
        raise TypeError

    def del_data(self, name):
        obj = self._name_dict.pop(name)
        self._data_list.index(obj)
        self._names.remove(name)
        self._data_dict.pop(id(obj))
        self._data_list.remove(obj)
        self.__reconstruct.add('var')
        self.__reconstruct.add('obs')

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
        for idx, file_path in enumerate(file_path_list):
            if file_path.endswith('.gef'):
                data_list.append(read_gef(
                    file_path,
                    bin_size=bin_sizes[idx] if bin_sizes else 100,
                    bin_type=bin_types[idx] if bin_types else 'bins',
                ))
            elif file_path.endswith('.gem'):
                data_list.append(read_gem(
                    file_path,
                    bin_size=bin_sizes[idx] if bin_sizes else 100,
                    bin_type=bin_types[idx] if bin_types else 'bins',
                ))
            elif file_path.endswith('.h5ad'):
                data_list.append(read_ann_h5ad(file_path))
            else:
                raise Exception(f'file format({file_path}) not support')
        return self.__add_data_objs(data_list, keys)

    def __get_auto_key(self) -> str:
        self.__idx_generator += 1
        return str(f'{self.__idx_generator}')

    def __real_add(self, obj: StereoExpData, key: Union[str, None] = None) -> object:
        if not key:
            key = self.__get_auto_key()
        self._name_dict[key] = obj
        self._data_dict[id(obj)] = key
        self._names.append(key)
        self._data_list.append(obj)
        self.__reconstruct.add('var')
        self.__reconstruct.add('obs')
        return self

    def __obs_indexes(self) -> pd.Index:
        indexes = '0_' + pd.Index(self._data_list[0].cell_names).astype('str')
        for idx in range(1, len(self._data_list)):
            indexes = indexes.append(
                f'{idx}_' + pd.Index(self._data_list[idx].cell_names).astype('str')
            )
        return indexes

    @property
    def obs(self) -> pd.DataFrame:
        if not self._data_list:
            raise Exception('`MSData` object with no data')
        if self._obs is None or 'obs' in self.__reconstruct:
            self._obs = pd.DataFrame(index=self.__obs_indexes(), columns=['test_obs_1'])
            if 'obs' in self.__reconstruct:
                self.__reconstruct.remove('obs')
        return self._obs

    def __var_indexes(self) -> set:
        res = set(self._data_list[0].gene_names)
        for obj in self._data_list[1:]:
            res = res & set(obj.gene_names)
        return res

    @property
    def var(self) -> pd.DataFrame:
        if not self._data_list:
            raise Exception('`MSData` object with no data')
        if self._var is None or 'var' in self.__reconstruct:
            self._var = pd.DataFrame(index=list(self.__var_indexes()), columns=['test_var_1'])
            if 'var' in self.__reconstruct:
                self.__reconstruct.remove('var')
        return self._var

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
            self._name_dict[self.__get_auto_key() if default_key else self._names[idx]] = obj
        for name, obj in self._name_dict.items():
            self._data_dict[id(obj)] = name
        self._names = []
        for obj in self._data_list:
            self._names.append(self._data_dict[id(obj)])
        return self


class MsDataResult(object):

    def __init__(self):
        self.result = dict()
        self._key_records = defaultdict(list)

    def set_result(self, res_key, res, type_key: str = None):
        self.result[res_key] = res
        if type_key:
            self._key_records[type_key].append(res_key)


class MSDataPipeLine(object):
    ATTR_NAME = 'tl'
    BASE_CLASS = StPipeline

    def __init__(self, _ms_data):
        self._ms_data = _ms_data
        self._result = MsDataResult()

    def __getattr__(self, item):
        dict_attr = self.__dict__.get(item, None)
        if dict_attr:
            return dict_attr

        # start with __ may not be our algorithm function, and will cause import problem
        if item.startswith('__'):
            raise AttributeError

        new_attr = MSDataPipeLine.BASE_CLASS.__dict__.get(item)
        if new_attr:
            def log_delayed_task(idx, *arg, **kwargs):
                logger.info(f'data_obj(idx={idx}) in ms_data start to run {item}')
                new_attr(*arg, **kwargs)

            def temp(*args, **kwargs):
                Parallel(n_jobs=min(len(self._ms_data._data_list), cpu_count()), backend='threading', verbose=100)(
                    delayed(log_delayed_task)(idx, obj.__getattribute__(MSDataPipeLine.ATTR_NAME), *args, **kwargs)
                    for idx, obj in enumerate(self._ms_data._data_list)
                )

            return temp

        from ..algorithm.algorithm_base import AlgorithmBase
        delayed_list = []
        for exp_obj in self._ms_data._data_list:
            obj_method = AlgorithmBase.get_attribute_helper(item, exp_obj.tl.data, exp_obj.tl.result)
            if obj_method:
                def log_delayed_task(idx, *arg, **kwargs):
                    logger.info(f'index-{idx} in ms_data start to run {item}')
                    obj_method(*arg, **kwargs)

                delayed_list.append(log_delayed_task)

        if delayed_list:
            def temp(*args, **kwargs):
                # TODO 这块有木有需求？有可能有多进程？
                Parallel(n_jobs=min(len(self._ms_data._data_list), cpu_count()), backend='threading', verbose=100)(
                    delayed(one_job)(idx, *args, **kwargs)
                    for idx, one_job in enumerate(delayed_list)
                )

            return temp

        from ..algorithm.ms_algorithm_base import MSDataAlgorithmBase
        ms_data_method = MSDataAlgorithmBase.get_attribute_helper(item, self._ms_data, self._result)
        if ms_data_method:
            return ms_data_method

        raise AttributeError


TL = type('TL', (MSDataPipeLine,), {'ATTR_NAME': 'tl', "BASE_CLASS": StPipeline})
PLT = type('PLT', (MSDataPipeLine,), {'ATTR_NAME': 'plt', "BASE_CLASS": PlotCollection})


@dataclass
class MSData(_MSDataStruct):
    _tl = None
    _plt = None

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

    def __str__(self):
        return f'''ms_data: {self._names}
num_slice: {self.num_slice}
names: {self.names}
obs: {self.obs.columns}
var: {self.var.columns}
'''

    def __repr__(self):
        return self.__str__()
