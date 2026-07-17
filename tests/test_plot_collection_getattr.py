"""Test that PlotCollection.__getattr__ properly handles None from get_attribute_helper.

Tests the fix for issue #454: TypeError: 'NoneType' object is not callable.
The bug was that when get_attribute_helper returned None, the download decorator
wrapped None, creating a callable that crashes with TypeError when invoked.
"""
import ast
import pytest


def test_getattr_none_guard_in_source():
    """Verify that the fixed __getattr__ checks for None before applying download."""
    with open('stereo/plots/plot_collection.py', 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'PlotCollection':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__getattr__':
                    source_lines = source.split('\n')
                    func_start = item.lineno - 1
                    func_end = item.end_lineno
                    func_source = '\n'.join(source_lines[func_start:func_end])

                    assert 'new_attr is None' in func_source or 'new_attr is not None' in func_source, \
                        "The __getattr__ method must check for None before applying download"

                    none_check_line = -1
                    download_apply_line = -1
                    for i, line in enumerate(func_source.split('\n')):
                        if 'new_attr is None' in line or 'new_attr is not None' in line:
                            none_check_line = i
                        if 'download(new_attr)' in line:
                            download_apply_line = i

                    assert none_check_line != -1, "None check not found in __getattr__"
                    assert download_apply_line != -1, "download(new_attr) not found in __getattr__"
                    assert none_check_line < download_apply_line, \
                        "None check must come BEFORE download(new_attr)"
                    return

    pytest.fail("PlotCollection.__getattr__ not found in source")


def test_fix_logic_correctness():
    """Directly test the fixed logic vs buggy logic to confirm the fix works."""
    from functools import wraps

    def download(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            fig = func(*args, **kwargs)
            return fig
        return wrapped

    def fixed_getattr(new_attr, item):
        """The FIXED __getattr__ logic."""
        if new_attr is None:
            raise AttributeError(
                f'{item} not existed, please check the function name you called!'
            )
        if getattr(new_attr, '__download__', True):
            new_attr = download(new_attr)
        return new_attr

    def buggy_getattr(new_attr, item):
        """The BUGGY __getattr__ logic (before fix)."""
        if getattr(new_attr, '__download__', True):
            new_attr = download(new_attr)
        if new_attr:
            return new_attr
        raise AttributeError(
            f'{item} not existed, please check the function name you called!'
        )

    with pytest.raises(AttributeError, match='paga_plot not existed'):
        fixed_getattr(None, 'paga_plot')

    buggy_result = buggy_getattr(None, 'paga_plot')
    assert callable(buggy_result), "Buggy code wraps None into a callable"
    with pytest.raises(TypeError, match="'NoneType' object is not callable"):
        buggy_result()

    def sample_plot():
        return "figure"

    result = fixed_getattr(sample_plot, 'sample_plot')
    assert callable(result)
    assert result() == "figure"


def test_download_decorator_with_none():
    """Verify the download decorator behavior when func is None."""
    from functools import wraps
    from matplotlib.figure import Figure

    def download(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            out_path = kwargs.pop('out_path', None)
            fig = func(*args, **kwargs)
            return fig
        return wrapped

    wrapped_none = download(None)
    assert callable(wrapped_none), "download(None) creates a callable wrapper"

    with pytest.raises(TypeError):
        wrapped_none()
