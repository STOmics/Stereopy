from copy import deepcopy
from typing import Union, List, Optional
import numpy as np
from stereo.core.stereo_exp_data import StereoExpData
from stereo.algorithm.ms_algorithm_base import MSDataAlgorithmBase
from .methods import pairwise_align, center_align
from .helper import stack_slices_pairwise, stack_slices_center

class SpatialAlignment(MSDataAlgorithmBase):
    def main(
        self,
        method: str = 'pairwise',
        initial_slice: Optional[Union[str, int]] = None,
        slices: Optional[List[Union[str, int]]] = None,
        *args,
        **kwargs
    ):
        """
        Calculates and returns optimal alignment of two slices or computes center alignment of slices.

        :param method: pairwise or center, defaults to 'pairwise'
        :param initial_slice: slice to use as the initialization for center alignment, defaults to None
        :param slices: list of slices to align, defaults to None
        """
        if method not in ('pairwise', 'center'):
            raise ValueError(f'Error method({method}), it must be one of pairwise and center')
        
        if method == 'pairwise':
            if slices is None:
                slices = self.ms_data.data_list
            else:
                slices = [self.ms_data[s] for s in slices]
            assert len(slices) >= 2, "You should have at least 2 slices on 'pairwise_align' method"
                
            scount = len(slices)
            pi_pairs = []
            for i in range(scount - 1):
                slice_a, slice_b = slices[i], slices[i + 1]
                pi_pairs.append(pairwise_align(slice_a, slice_b, *args, **kwargs))
            stack_slices_pairwise(slices, pi_pairs)
        else:
            if slices is None:
                slices = self.ms_data.data_list
            else:
                slices = [self.ms_data[s] for s in slices]
            assert len(slices) >= 2, "You should have at least 2 slices on 'center_align' method"
            if initial_slice is None:
                initial_slice = deepcopy(slices[0])
            else:
                initial_slice = deepcopy(self.ms_data[initial_slice])
            center_slice, pis = center_align(initial_slice, slices, *args, **kwargs)
            stack_slices_center(center_slice, slices, pis)
            self.ms_data.center_slice = center_slice
        
        return self.ms_data
