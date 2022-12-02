import re
import sys
import time
import inspect

from stereo.log_manager import logger

__CAMEL_TO_SNAKE = re.compile(r"((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")


def camel_to_snake(value):
    return __CAMEL_TO_SNAKE.sub(r"_\1", value).lower()


class EnumErrorCode:
    Success, SuccessMsg = 0, 'All steps done, algorithm run perfectly! ($_$)'
    Failed, FailedMsg = -1, 'Failed, examine the log and do not be depressed! (^_^)'


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
    '''


class AlgorithmBase(object):
    """
    `AlgorithmBase` designed for auto-loading algorithms into `stereo.core.pipeline.StPipeline`.

    Example Usage:
        >>> # first, register the algorithms you want to use
        >>> from stereo.algorithm.algorithm_example import Algorithm1
        >>> # Output:
        >>> #     algorithm name: Algorithm1 is register to class AlgorithmBase
        >>> from stereo.core import StereoExpData
        >>> st_data = StereoExpData()
        >>> # algorithm1 auto-loaded as a function of `StPipeline` with a snake case function name
        >>> st_data.tl.algorithm1()
    """

    __SUB_CLASSES = dict()
    __STEPS_ORDER_BY_NAME = list()

    @staticmethod
    def __register_subclass_to_base(sub_cls):
        # convert camel case as snake case, will be as function name in pipeline
        camel_sub_cls_name = camel_to_snake(sub_cls.__name__)
        assert camel_sub_cls_name not in AlgorithmBase.__SUB_CLASSES, f'{camel_sub_cls_name} already registered'
        AlgorithmBase.__SUB_CLASSES[camel_sub_cls_name] = sub_cls
        logger.info(f'algorithm name: {sub_cls.__name__} is registered to `AlgorithmBase`')

    @staticmethod
    def _get_all_subclass():
        return AlgorithmBase.__SUB_CLASSES.items()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        cls.__SUB_CLASSES = dict()
        # sorted by the starting word, just like 'step1' ... 'step{N}' , before first '_' char
        cls.__STEPS_ORDER_BY_NAME = sorted(
            [(f_name, f, inspect.getfullargspec(f).args) for f_name, f in cls.__dict__.items() if
             f_name.startswith('step')],
            key=lambda x: x[0].split('_')[0]
        )
        AlgorithmBase.__register_subclass_to_base(cls)

    def __init__(self, stereo_exp_data):
        # all algorithm are design for `StereoExpData` analysis
        self.stereo_exp_data = stereo_exp_data

    def main(self, **kwargs):
        if not self.__STEPS_ORDER_BY_NAME:
            raise NotImplementedError(
                f'algorithm {self.__class__.__name__} have no steps, overwrite function `main` or you can define your '
                f'`step` function flow the doc:\n{STEP_FUNC_DESIGN_DOC}'
            )
        the_very_beginning_time = time.time()
        for f_name, f, f_args_name in self.__STEPS_ORDER_BY_NAME:
            step_start_time = time.time()
            logger.debug(f'start to run {f_name}.')
            ret_code = f(self, **{key: value for key, value in kwargs.items() if key in f_args_name})
            if type(ret_code) is int and ret_code != EnumErrorCode.Success:
                logger.warn(f'{f_name} failed with ret_code: {ret_code}')
                return EnumErrorCode.Failed
            logger.debug(f'{f_name} end, cost: {round(time.time() - step_start_time, 4)} seconds')
        logger.info(f'{self.__class__.__name__} end, cost: {round(time.time() - the_very_beginning_time, 4)} seconds')
        return EnumErrorCode.Success

    def iter(self, *args, **kwargs):
        for f_name, f, _ in self.__STEPS_ORDER_BY_NAME:
            ret_code = f(self, *args, **kwargs)
            if type(ret_code) is int and ret_code != EnumErrorCode.Success:
                logger.warn(f'{f_name} failed with ret_code: {ret_code}')
                return EnumErrorCode.Failed
            yield EnumErrorCode.Success

    def test_memory_profile_step_by_step(self, stream=sys.stdout, **kwargs):
        from memory_profiler import profile
        # these cost a little bit `io` and `cpu` to profile memory usage
        if self.__STEPS_ORDER_BY_NAME:
            for f_name, f, f_args_name in self.__STEPS_ORDER_BY_NAME:
                main_mem_check_func = profile(f, stream=stream)
                main_mem_check_func(self, **{key: value for key, value in kwargs.items() if key in f_args_name})
            return EnumErrorCode.Success
        main_mem_check_func = profile(self.main, stream=stream)
        main_mem_check_func(**kwargs)
        return EnumErrorCode.Success
