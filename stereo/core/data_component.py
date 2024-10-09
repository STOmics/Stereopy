from typing import Union
from copy import deepcopy

from anndata import AnnData
import numpy as np
from scipy.sparse import spmatrix
import pandas as pd

from . import stereo_exp_data

class Layers(dict):
    def __init__(
        self,
        data: 'stereo_exp_data.StereoExpData',
        *args,
        **kwargs
    ):
        super(Layers, self).__init__(*args, **kwargs)
        self.__stereo_exp_data = data

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        if id(self) in memo:
            new_layers = memo[id(self)]
        else:
            new_layers = Layers(self.__stereo_exp_data)
            memo[id(self)] = new_layers
            if id(self.__stereo_exp_data) in memo:
                data = memo[id(self.__stereo_exp_data)]
            else:
                data = deepcopy(self.__stereo_exp_data, memo)
            
            new_attrs = {
                deepcopy(k, memo): deepcopy(v, memo) for k, v in self.__dict__.items() if k != '_Layers__stereo_exp_data'
            }
            new_attrs['_Layers__stereo_exp_data'] = data
            new_layers.__dict__.update(new_attrs)
            for k, v in self.items():
                dict.__setitem__(new_layers, deepcopy(k, memo), deepcopy(v, memo))
        return new_layers
    
    def __setitem__(self, key, value):
        if not isinstance(value, (np.ndarray, spmatrix, pd.DataFrame)):
            raise ValueError("layer must be np.ndarray, spmatrix or pd.DataFrame.")
        if value.shape != self.__stereo_exp_data.shape:
            raise ValueError(f"in layer '{key}', expected shape {self.__stereo_exp_data.shape}, but got {value.shape}.")
        if isinstance(value, pd.DataFrame):
            value = value.to_numpy(copy=True)
        super().__setitem__(key, value)
    
    def __str__(self) -> str:
        info = f"layers with keys {list(self.keys())}."
        return info

    def __repr__(self) -> str:
        return self.__str__()
