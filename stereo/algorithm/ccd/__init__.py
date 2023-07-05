import logging

from .community_clustering_algorithm import CommunityClusteringAlgo
from .html_report import generate_report
from .metrics import calculate_spatial_metrics
from .utils import timeit, plot_spatial
from .constants import *
from .metrics import *

try:
    from .sliding_window import SlidingWindow
except ImportError:
    logging.warn("Module SlidingWindow is not present.")


try:
    from .sliding_window import SlidingWindowMultipleSizes
except ImportError:
    logging.warn("Module SlidingWindowMultipleSizes is not present.")
