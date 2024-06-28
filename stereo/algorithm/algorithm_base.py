import inspect
import re
import sys
import time
from abc import ABCMeta
from dataclasses import dataclass
from typing import final

from .algorithm_err_code import ErrorCode
from ..core.stereo_exp_data import StereoExpData
from ..log_manager import logger


@dataclass
class AlgorithmBase(metaclass=ABCMeta):
    """
    `AlgorithmBase` designed for auto-loading algorithms into `stereo.core.pipeline.StPipeline`.

    Example Usage:
        >>> from stereo.core import StereoExpData
        >>> st_data = StereoExpData()
        >>> # algorithm1 auto-loaded as a function of `StPipeline` with a snake case function name
        >>> st_data.tl.algorithm1()
    """

    # common object variable
    stereo_exp_data: StereoExpData = None
    pipeline_res: dict = None

    _steps_order_by_name = list()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        # sorted by the starting word, just like 'step1' ... 'step{N}' , before first '_' char
        cls._steps_order_by_name = sorted(
            [
                (f_name, f, inspect.getfullargspec(f).args)
                for f_name, f in cls.__dict__.items() if f_name.startswith('step')
            ],
            key=lambda x: x[0]
        )

    def main(self, **kwargs):
        if not self._steps_order_by_name:
            raise NotImplementedError(
                f'algorithm {self.__class__.__name__} have no steps, overwrite function `main` or you can define your '
                f'`step` function flow the doc:\n{STEP_FUNC_DESIGN_DOC}'
            )
        the_very_beginning_time = time.time()
        for f_name, f, f_args_name in self._steps_order_by_name:
            step_start_time = time.time()
            logger.debug(f'start to run {f_name}.')
            ret_code = f(self, **{key: value for key, value in kwargs.items() if key in f_args_name})
            if type(ret_code) is int and ret_code != ErrorCode.Success:
                logger.warning(f'{f_name} failed with ret_code: {ret_code}')
                return ErrorCode.Failed
            logger.debug(f'{f_name} end, cost: {round(time.time() - step_start_time, 4)} seconds')
        logger.info(f'{self.__class__.__name__} end, cost: {round(time.time() - the_very_beginning_time, 4)} seconds')
        return ErrorCode.Success

    @final
    def iter(self, *args, **kwargs):
        for f_name, f, _ in self._steps_order_by_name:
            ret_code = f(self, *args, **kwargs)
            if type(ret_code) is int and ret_code != ErrorCode.Success:
                logger.warning(f'{f_name} failed with ret_code: {ret_code}')
                return ErrorCode.Failed
            yield ErrorCode.Success

    @final
    def memory_profile(self, stream=sys.stdout, **kwargs):
        try:
            from memory_profiler import profile
        except ImportError:
            raise ImportError
        # these cost some `io` and `cpu` to profile memory
        if self._steps_order_by_name:
            for f_name, f, f_args_name in self._steps_order_by_name:
                main_mem_check_func = profile(f, stream=stream)
                main_mem_check_func(self, **{key: value for key, value in kwargs.items() if key in f_args_name})
            return ErrorCode.Success
        main_mem_check_func = profile(self.main, stream=stream)
        main_mem_check_func(**kwargs)
        return ErrorCode.Success

    @staticmethod
    def get_attribute_helper(item, stereo_exp_data: StereoExpData, res: dict):
        try:
            __import__(f"stereo.algorithm.{item}")
        except Exception:
            raise AttributeError(f"No attribute named 'StPipeline.{item}'")

        # TODO: this may be not the best way to get sub-class
        # num of subclasses may be like 100-200 at most
        for sub_cls in AlgorithmBase.__subclasses__():
            sub_cls_name = _camel_to_snake(sub_cls.__name__.split(".")[-1])
            if sub_cls_name == 'ms_data_algorithm_base':
                continue
            if sub_cls_name == item:
                # snake_cls_name as method name in pipeline
                sub_obj = sub_cls(stereo_exp_data=stereo_exp_data, pipeline_res=res)
                return sub_obj.main
        return None


__CAMEL_TO_SNAKE = re.compile(r"((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")


def _camel_to_snake(value):
    return __CAMEL_TO_SNAKE.sub(r"_\1", value).lower()


STEP_FUNC_DESIGN_DOC = '''
    The `step` functions defined in subclass of `AlgorithmBase`, whose names started with `step1_`, `step2_` ...
    `step{N}_`, will execute in ascending order by number `N` when someone run a not-overwrite `main` function.

    Example:
        class AlgorithmDemo(AlgorithmBase):
            ...
            def step{N-1}_your_step_name(self):
                pass

            def step{N}_your_step_name(self):
                pass

    More examples are in `algorithm_example.py`.
    ''' # noqa
