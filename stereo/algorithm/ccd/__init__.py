# flake8: noqa
from .constants import *
from .html_report import generate_report
from .metrics import *
from .sliding_window import SlidingWindow
from .sliding_window import SlidingWindowMultipleSizes
from .utils import (
    timeit,
    plot_spatial,
    set_figure_params,
    reset_figure_params
)
