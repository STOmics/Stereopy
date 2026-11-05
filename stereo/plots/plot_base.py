import importlib
import sys
from abc import ABCMeta
from pathlib import Path
from typing import Union

from ..core.stereo_exp_data import AnnBasedStereoExpData
from ..core.stereo_exp_data import StereoExpData

# Plot classes register themselves when their modules are imported.
# ``plot_paga`` is first so ``data.plt.paga_plot`` does not depend on later,
# optional plot modules importing successfully.
_PLOT_MODULE_NAMES = (
    'stereo.plots.plot_paga',
    'stereo.plots.plot_cluster_traj',
    'stereo.plots.plot_cluster_traj_3d',
    'stereo.plots.plot_vec',
    'stereo.plots.plot_vec_3d',
    'stereo.plots.vt3d_browser.example',
    'stereo.plots.plot_ccd',
    'stereo.plots.plot_coo',
    'stereo.plots.plot_dendrogram',
    'stereo.plots.plot_clusters_scatter',
    'stereo.plots.plot_clusters_heatmap',
    'stereo.plots.plot_time_series',
    'stereo.plots.plot_elbow',
    'stereo.plots.plot_genes_in_pseudotime',
    'stereo.plots.plot_st_gears',
    'stereo.plots.plot_ccc',
    'stereo.plots.plot_grn',
    'stereo.plots.plot_batch_qc',
    'stereo.plots.plot_ms_spatial_scatter',
    'stereo.plots.plot_spa_track',
)

_ATTEMPTED_PLOT_MODULES = set()
_LAST_PLOT_IMPORT_ERROR = None


def _logger():
    """Return the Stereopy logger without importing it at module import time.

    Falls back to the standard library logger if Stereopy's logger cannot be
    imported, so a logging failure does not hide the original plot import error.
    """
    try:
        from stereo.log_manager import logger
        return logger
    except Exception:
        import logging
        return logging.getLogger('Stereo')


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
        """Resolve a plot method registered by a :class:`PlotBase` subclass.

        Parameters
        ----------
        item : str
            Plot method name, for example ``paga_plot``.
        data : StereoExpData or AnnBasedStereoExpData
            Dataset passed to the plot class constructor.
        result : dict
            Pipeline result mapping passed to the plot class constructor.

        Returns
        -------
        callable or None
            Bound plot method, or ``None`` when no plot class provides ``item``.
        """
        global _LAST_PLOT_IMPORT_ERROR
        _LAST_PLOT_IMPORT_ERROR = None
        _ensure_plot_classes_registered(item)
        names = PlotBase.PLOT_NAME_TO_NAMES.get(item, None)
        if not names:
            if _LAST_PLOT_IMPORT_ERROR:
                _logger().warning(
                    "Plot function '%s' is unavailable because a plot module failed to import: %s",
                    item,
                    _LAST_PLOT_IMPORT_ERROR,
                )
            return None

        module = _import_plot_module(names[0])
        cls = None
        if module is not None:
            cls = getattr(module, names[1], None)
        if cls is None:
            for sub_cls in PlotBase.__subclasses__():
                sub_cls_name = sub_cls.__name__.split(".")[-1]
                if sub_cls_name == names[1] and (module is None or sub_cls.__module__ == names[0]):
                    cls = sub_cls
                    break
        if cls is None:
            return None
        sub_obj = cls(data, result)
        return getattr(sub_obj, item, None)


def get_last_plot_import_error():
    """Return the most recent plot-module import error, if any.

    Returns
    -------
    str or None
        Exception text captured while importing a plot module.
    """
    return _LAST_PLOT_IMPORT_ERROR


def _ensure_plot_classes_registered(item):
    """Import plot modules until ``item`` is registered or every module was tried.

    ``stereo.plots`` imports every plot class from package ``__init__``. If that
    import stops before ``PlotPaga`` is defined, ``plot_collection`` can still be
    cached in ``sys.modules`` and ``data.plt.paga_plot`` looks up a missing name.
    Importing the plot module here registers the class.
    """
    if PlotBase.PLOT_NAME_TO_NAMES.get(item, None):
        return
    for module_name in _PLOT_MODULE_NAMES:
        if PlotBase.PLOT_NAME_TO_NAMES.get(item, None):
            return
        if module_name in _ATTEMPTED_PLOT_MODULES:
            continue
        _ATTEMPTED_PLOT_MODULES.add(module_name)
        _import_plot_module(module_name)


def _import_plot_module(module_name):
    """Import ``module_name`` and return the module, or ``None`` on failure.

    A failed ``stereo.plots`` package init removes the package from
    ``sys.modules`` but leaves submodules that were already imported. Importing
    ``stereo.plots.plot_paga`` then re-runs ``__init__.py`` and fails at the
    same earlier import, so the file is loaded directly in that case.
    """
    global _LAST_PLOT_IMPORT_ERROR
    if not module_name:
        return None
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        _LAST_PLOT_IMPORT_ERROR = str(exc)
        _logger().warning("Importing %s failed (%s); trying to load the module file.", module_name, exc)
    if module_name.startswith('stereo.plots.'):
        return _load_plot_module_from_file(module_name)
    return None


def _ensure_plots_package():
    """Make sure ``stereo.plots`` exists in ``sys.modules`` with a filesystem path."""
    import types

    pkg_name = 'stereo.plots'
    plots_dir = str(Path(__file__).resolve().parent)
    pkg = sys.modules.get(pkg_name)
    if pkg is None:
        pkg = types.ModuleType(pkg_name)
        sys.modules[pkg_name] = pkg
    if not getattr(pkg, '__path__', None):
        pkg.__path__ = [plots_dir]
    pkg.__package__ = pkg_name
    parent = sys.modules.get('stereo')
    if parent is not None and not hasattr(parent, 'plots'):
        parent.plots = pkg
    return pkg


def _load_plot_module_from_file(module_name):
    """Load a ``stereo.plots`` submodule from its file after package init failed."""
    global _LAST_PLOT_IMPORT_ERROR
    import importlib.util

    rel_name = module_name[len('stereo.plots.'):]
    rel_path = Path(rel_name.replace('.', '/'))
    base = Path(__file__).resolve().parent
    path = None
    for candidate in (base / rel_path.with_suffix('.py'), base / rel_path / '__init__.py'):
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        return None

    _ensure_plots_package()
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = module_name.rpartition('.')[0]
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        _LAST_PLOT_IMPORT_ERROR = str(exc)
        _logger().warning("Failed to load plot module %s from %s: %s", module_name, path, exc)
        return None
    return module
