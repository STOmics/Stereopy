from abc import ABCMeta
from typing import Union

from ..core.stereo_exp_data import AnnBasedStereoExpData
from ..core.stereo_exp_data import StereoExpData


class PlotBase(metaclass=ABCMeta):
    PLOT_NAME_TO_NAMES = {}

    def __init__(self, stereo_exp_data: Union[StereoExpData, AnnBasedStereoExpData], pipeline_res: dict = None):
        # common object variable
        self.stereo_exp_data = stereo_exp_data
        self.pipeline_res = pipeline_res

    @classmethod
    def __init_subclass__(cls, **kwargs):
        for attr_name, attr in cls.__dict__.items():
            if callable(attr):
                PlotBase.PLOT_NAME_TO_NAMES[attr_name] = (cls.__module__, cls.__name__)

    @staticmethod
    def get_attribute_helper(item, data, result):
        names = PlotBase.PLOT_NAME_TO_NAMES.get(item, None)
        if not names:
            return None

        try:
            __import__(f"{names[0]}")
        except Exception:
            raise AttributeError(f"No module named '{names[0]}'")

        for sub_cls in PlotBase.__subclasses__():
            sub_cls_name = sub_cls.__name__.split(".")[-1]
            if sub_cls_name == names[1]:
                sub_obj = sub_cls(data, result)
                return sub_obj.__getattribute__(item)
        return None
