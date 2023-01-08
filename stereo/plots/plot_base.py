from abc import ABCMeta
from dataclasses import dataclass

from ..core import StereoExpData


@dataclass
class PlotBase(metaclass=ABCMeta):
    PLOT_NAME_TO_NAMES = {}

    # common object variable
    stereo_exp_data: StereoExpData
    pipeline_res: dict = None

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
        except:
            raise AttributeError(f"No module named '{names[0]}'")

        for sub_cls in PlotBase.__subclasses__():
            sub_cls_name = sub_cls.__name__.split(".")[-1]
            if sub_cls_name == names[1]:
                sub_obj = sub_cls(data, result)
                return sub_obj.__getattribute__(item)
        return None
