from copy import deepcopy
from typing import (
    Union,
    List,
    Optional
)

from stereo.algorithm.ms_algorithm_base import MSDataAlgorithmBase
from stereo.log_manager import logger
from .helper import stack_slices_center
from .helper import stack_slices_pairwise
from .methods import center_align
from .methods import pairwise_align


class Paste(MSDataAlgorithmBase):
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

        Other parameters refer to `algorithm.paste.pairwise_align` and `algorithm.paste.center_align`
        """
        if method not in ('pairwise', 'center'):
            raise ValueError(f'Error method({method}), it must be one of pairwise and center')
        
        try:
            from tensorflow.python.ops.numpy_ops import np_config
            np_config.enable_numpy_behavior()
        except:
            pass

        logger.info(f'Using method {method}')
        if method == 'pairwise':
            if slices is None:
                slice_names = self.ms_data.names
                slices = self.ms_data.data_list
            else:
                slice_names = [s if isinstance(s, str) else self.ms_data.names[s] for s in slices]
                slices = [self.ms_data[s] for s in slices]
            assert len(slices) >= 2, "You should have at least 2 slices on 'pairwise_align' method"

            scount = len(slices)
            pi_pairs = []
            for i in range(scount - 1):
                slice_a, slice_b = slices[i], slices[i + 1]
                logger.info(f'Processing slice {slice_names[i]} and {slice_names[i + 1]}')
                pi_pairs.append(pairwise_align(slice_a, slice_b, *args, **kwargs))
            stack_slices_pairwise(slices, pi_pairs)
        else:
            if slices is None:
                slice_names = self.ms_data.names
                slices = self.ms_data.data_list
            else:
                slice_names = [s if isinstance(s, str) else self.ms_data.names[s] for s in slices]
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
