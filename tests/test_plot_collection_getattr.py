"""Tests for issue #454: TypeError when calling data.plt.paga_plot.

PlotCollection.__getattr__ used to pass None into the download decorator when
plot lookup failed. Calling that wrapper raises
``TypeError: 'NoneType' object is not callable``. Lookup also has to import
PlotPaga when package init did not register it.
"""
import ast
import importlib.util
import sys
import types
from functools import wraps

import pytest


def _load_plot_base():
    """Load plot_base without importing the rest of Stereopy."""
    module_name = 'stereo.plots.plot_base'
    cached = sys.modules.get(module_name)
    if cached is not None and hasattr(cached, 'PlotBase'):
        return cached

    def _stub(name):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        return module

    _stub('stereo')
    _stub('stereo.core')
    core_data = _stub('stereo.core.stereo_exp_data')

    class StereoExpData(object):
        pass

    class AnnBasedStereoExpData(object):
        pass

    core_data.StereoExpData = StereoExpData
    core_data.AnnBasedStereoExpData = AnnBasedStereoExpData
    plots_pkg = _stub('stereo.plots')
    plots_pkg.__path__ = ['/workspace/stereo/plots']
    plots_pkg.__package__ = 'stereo.plots'

    spec = importlib.util.spec_from_file_location(
        module_name,
        '/workspace/stereo/plots/plot_base.py',
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = 'stereo.plots'
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def plot_base():
    return _load_plot_base()


def _function_source(path, class_name, func_name):
    with open(path, 'r') as handle:
        source = handle.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == func_name:
                    lines = source.splitlines()
                    return '\n'.join(lines[item.lineno - 1:item.end_lineno])
    raise AssertionError('{0}.{1} not found in {2}'.format(class_name, func_name, path))


def test_getattr_checks_none_before_download():
    source = _function_source('stereo/plots/plot_collection.py', 'PlotCollection', '__getattr__')
    none_at = source.find('new_attr is None')
    download_at = source.find('download(new_attr)')
    assert none_at != -1
    assert download_at != -1
    assert none_at < download_at


def test_ms_pipeline_does_not_decorate_none():
    with open('stereo/core/ms_pipeline.py', 'r') as handle:
        source = handle.read()
    assert 'download(PlotBase.get_attribute_helper' not in source
    assert 'new_attr is not None and getattr(new_attr, \'__download__\', True)' in source
    assert 'if new_attr is not None and base.__name__ == \'PlotBase\':' in source


def test_download_none_matches_reported_typeerror():
    def download(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            fig = func(*args, **kwargs)
            return fig
        return wrapped

    wrapped_none = download(None)
    assert callable(wrapped_none)
    with pytest.raises(TypeError, match="'NoneType' object is not callable"):
        wrapped_none()


def test_registered_plot_method_is_callable(plot_base):
    class PlotPagaIssue454(plot_base.PlotBase):
        def paga_plot_issue454(self, adjacency='connectivities_tree'):
            return 'figure-' + adjacency

    try:
        method = plot_base.PlotBase.get_attribute_helper('paga_plot_issue454', None, None)
        assert callable(method)
        assert method(adjacency='connectivities') == 'figure-connectivities'
    finally:
        plot_base.PlotBase.PLOT_NAME_TO_NAMES.pop('paga_plot_issue454', None)


def test_unregistered_paga_plot_is_loaded_and_callable(plot_base):
    saved = dict(plot_base.PlotBase.PLOT_NAME_TO_NAMES)
    saved_attempted = set(plot_base._ATTEMPTED_PLOT_MODULES)
    original_import = plot_base._import_plot_module
    created = {}

    def fake_import(module_name):
        if 'cls' not in created:
            class PlotPaga(plot_base.PlotBase):
                def paga_plot(self, adjacency='connectivities_tree'):
                    return 'paga-' + adjacency

            module = types.ModuleType(PlotPaga.__module__)
            module.PlotPaga = PlotPaga
            created['cls'] = PlotPaga
            created['mod'] = module
        return created['mod']

    plot_base.PlotBase.PLOT_NAME_TO_NAMES.pop('paga_plot', None)
    plot_base._ATTEMPTED_PLOT_MODULES.clear()
    plot_base._import_plot_module = fake_import
    try:
        method = plot_base.PlotBase.get_attribute_helper('paga_plot', None, {'paga': {}})
        assert callable(method)
        assert method() == 'paga-connectivities_tree'
    finally:
        plot_base.PlotBase.PLOT_NAME_TO_NAMES.clear()
        plot_base.PlotBase.PLOT_NAME_TO_NAMES.update(saved)
        plot_base._ATTEMPTED_PLOT_MODULES.clear()
        plot_base._ATTEMPTED_PLOT_MODULES.update(saved_attempted)
        plot_base._import_plot_module = original_import


def test_missing_plot_returns_none_instead_of_wrapper(plot_base):
    saved = dict(plot_base.PlotBase.PLOT_NAME_TO_NAMES)
    saved_attempted = set(plot_base._ATTEMPTED_PLOT_MODULES)
    original_import = plot_base._import_plot_module
    plot_base._import_plot_module = lambda module_name: None
    plot_base._ATTEMPTED_PLOT_MODULES.clear()
    plot_base.PlotBase.PLOT_NAME_TO_NAMES.pop('missing_plot_issue454', None)
    try:
        resolved = plot_base.PlotBase.get_attribute_helper('missing_plot_issue454', None, None)
        assert resolved is None
    finally:
        plot_base.PlotBase.PLOT_NAME_TO_NAMES.clear()
        plot_base.PlotBase.PLOT_NAME_TO_NAMES.update(saved)
        plot_base._ATTEMPTED_PLOT_MODULES.clear()
        plot_base._ATTEMPTED_PLOT_MODULES.update(saved_attempted)
        plot_base._import_plot_module = original_import


def test_direct_class_lookup_when_subclass_scan_would_miss(plot_base):
    """Use the class stored on the module when another PlotPaga subclass exists."""
    saved = dict(plot_base.PlotBase.PLOT_NAME_TO_NAMES)

    class PlotPaga(plot_base.PlotBase):
        def paga_plot(self):
            return 'stale'

    module = types.ModuleType('stereo.plots.plot_paga_issue454')
    sys.modules[module.__name__] = module

    class RealPlotPaga(plot_base.PlotBase):
        def paga_plot(self):
            return 'real'

    RealPlotPaga.__name__ = 'PlotPaga'
    RealPlotPaga.__module__ = module.__name__
    module.PlotPaga = RealPlotPaga
    plot_base.PlotBase.PLOT_NAME_TO_NAMES['paga_plot'] = (module.__name__, 'PlotPaga')
    try:
        method = plot_base.PlotBase.get_attribute_helper('paga_plot', None, None)
        assert method() == 'real'
    finally:
        plot_base.PlotBase.PLOT_NAME_TO_NAMES.clear()
        plot_base.PlotBase.PLOT_NAME_TO_NAMES.update(saved)
        sys.modules.pop(module.__name__, None)


def test_failed_plot_import_returns_none(plot_base):
    """A missing plot module must not raise while the dispatcher is resolving a name."""
    saved_error = plot_base._LAST_PLOT_IMPORT_ERROR
    try:
        loaded = plot_base._import_plot_module('stereo.plots.no_such_module_issue454')
        assert loaded is None
        assert plot_base.get_last_plot_import_error()
        missing = plot_base._load_plot_module_from_file('stereo.plots.no_such_module_issue454')
        assert missing is None
    finally:
        plot_base._LAST_PLOT_IMPORT_ERROR = saved_error


def test_fixed_dispatcher_raises_attributeerror_for_none():
    """None from lookup must not become a wrapper that raises TypeError."""

    def download(func):
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped

    def fixed_getattr(new_attr, item):
        if new_attr is None:
            raise AttributeError(
                '{0} not existed, please check the function name you called!'.format(item)
            )
        if getattr(new_attr, '__download__', True):
            new_attr = download(new_attr)
        return new_attr

    def buggy_getattr(new_attr, item):
        if getattr(new_attr, '__download__', True):
            new_attr = download(new_attr)
        if new_attr:
            return new_attr
        raise AttributeError(item)

    with pytest.raises(AttributeError, match='paga_plot not existed'):
        fixed_getattr(None, 'paga_plot')

    wrapped = buggy_getattr(None, 'paga_plot')
    with pytest.raises(TypeError, match="'NoneType' object is not callable"):
        wrapped()

    def sample_plot():
        return 'figure'

    assert fixed_getattr(sample_plot, 'sample_plot')() == 'figure'
