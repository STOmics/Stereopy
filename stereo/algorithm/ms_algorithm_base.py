from dataclasses import dataclass
from abc import ABCMeta

from stereo.algorithm.algorithm_base import AlgorithmBase, _camel_to_snake
from stereo.core.ms_data import MSData


@dataclass
class MSDataAlgorithmBase(metaclass=ABCMeta):
    ms_data: MSData = None
    pipeline_res: dict = None

    @staticmethod
    def get_attribute_helper(item, ms_data: MSData, res: dict):
        try:
            __import__(f"stereo.algorithm.{item}")
        except Exception:
            # raise AttributeError(f"No attribute named 'StPipeline.{item}'")
            return None

        # TODO: this may be not the best way to get sub-class
        # num of subclasses may be like 100-200 at most
        for sub_cls in MSDataAlgorithmBase.__subclasses__():
            sub_cls_name = _camel_to_snake(sub_cls.__name__.split(".")[-1])
            if sub_cls_name == item:
                # snake_cls_name as method name in pipeline
                sub_obj = sub_cls(ms_data=ms_data, pipeline_res=res)
                return sub_obj.main
        return None
