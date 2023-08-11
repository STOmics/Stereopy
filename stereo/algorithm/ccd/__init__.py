from .html_report import generate_report
from .metrics import calculate_spatial_metrics
from .utils import timeit, plot_spatial, set_figure_params
from .constants import *
from .metrics import *

from stereo.log_manager import logger

try:
    from .sliding_window import SlidingWindow
except ImportError:
    logger.warning("Module SlidingWindow is not present.")


try:
    from .sliding_window import SlidingWindowMultipleSizes
except ImportError:
    logger.warning("Module SlidingWindowMultipleSizes is not present.")
