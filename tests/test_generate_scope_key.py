"""Test that generate_scope_key always returns a hashable string.

This test extracts the generate_scope_key logic directly from ms_data.py
to avoid importing the full stereo dependency chain.
"""
import ast
import textwrap
import unittest

import numpy as np
import pandas as pd


def _extract_generate_scope_key():
    """Parse generate_scope_key from source and build a callable stub class."""
    with open('stereo/core/ms_data.py') as f:
        source = f.read()

    tree = ast.parse(source)

    func_source = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'generate_scope_key':
            lines = source.splitlines()
            func_lines = lines[node.lineno - 1:node.end_lineno]
            func_source = '\n'.join(func_lines)
            break

    assert func_source is not None, 'Could not find generate_scope_key in ms_data.py'
    return func_source


class _StubMSData:
    """Minimal MSData stub that has just enough for generate_scope_key to work."""

    def __init__(self, names):
        self._names = list(names)
        self._name_dict = {n: object() for n in names}

    def __getitem__(self, key):
        if isinstance(key, slice):
            view = type(self)(self._names[key])
            return view
        elif isinstance(key, (list, tuple)):
            resolved = []
            for k in key:
                if isinstance(k, str) and k in self._name_dict:
                    resolved.append(k)
                elif isinstance(k, int) and k < len(self._names):
                    resolved.append(self._names[k])
                else:
                    raise KeyError(k)
            return type(self)(resolved)
        elif isinstance(key, str):
            return self._name_dict[key]
        elif isinstance(key, int):
            return self._names[key]
        raise KeyError(key)

    def get_data_list(self, key_idx_list):
        data_list = []
        names = []
        for ki in key_idx_list:
            if isinstance(ki, (str, np.str_)):
                if ki not in self._name_dict:
                    raise KeyError(ki)
                data_list.append(self._name_dict[ki])
                names.append(ki)
            elif isinstance(ki, (int, np.integer)):
                ki = int(ki)
                data_list.append(object())
                names.append(self._names[ki])
            else:
                raise KeyError(ki)
        return data_list, names


exec(compile(
    "import numpy as np\nimport pandas as pd\n" + textwrap.dedent(_extract_generate_scope_key()),
    '<generate_scope_key>',
    'exec'
))
_StubMSData.generate_scope_key = locals()['generate_scope_key']


class TestGenerateScopeKey(unittest.TestCase):
    """Unit tests for generate_scope_key ensuring it always returns hashable strings."""

    def _make(self, names):
        return _StubMSData(names)

    def test_scope_none_returns_string(self):
        stub = self._make(['s0', 's1', 's2'])
        key = stub.generate_scope_key(None)
        self.assertIsInstance(key, str)

    def test_scope_int_returns_string(self):
        stub = self._make(['s0', 's1'])
        key = stub.generate_scope_key(0)
        self.assertEqual(key, 'scope_[0]')

    def test_scope_str_known_name(self):
        stub = self._make(['s0', 's1'])
        key = stub.generate_scope_key('s1')
        self.assertEqual(key, 'scope_[1]')

    def test_scope_str_unknown(self):
        stub = self._make(['s0', 's1'])
        key = stub.generate_scope_key('unknown')
        self.assertEqual(key, 'unknown')

    def test_scope_list_returns_hashable_string(self):
        stub = self._make(['s0', 's1', 's2'])
        key = stub.generate_scope_key(['s0', 's1', 's2'])
        self.assertIsInstance(key, str)
        self.assertTrue(key.startswith('scope_['))
        {key: True}

    def test_scope_list_with_bad_names_fallback_still_hashable(self):
        """This is the core regression test for issue #386."""
        stub = self._make(['s0', 's1'])
        key = stub.generate_scope_key(['bad_name1', 'bad_name2'])
        self.assertIsInstance(key, str)
        {key: True}

    def test_scope_tuple_returns_hashable(self):
        stub = self._make(['s0', 's1'])
        key = stub.generate_scope_key(('s0', 's1'))
        self.assertIsInstance(key, str)
        {key: True}

    def test_scope_slice_returns_hashable(self):
        stub = self._make(['s0', 's1', 's2'])
        key = stub.generate_scope_key(slice(None))
        self.assertIsInstance(key, str)
        {key: True}

    def test_scope_ndarray_returns_hashable(self):
        stub = self._make(['s0', 's1', 's2'])
        key = stub.generate_scope_key(np.array(['s0', 's1']))
        self.assertIsInstance(key, str)
        {key: True}

    def test_scope_pd_index_returns_hashable(self):
        stub = self._make(['s0', 's1', 's2'])
        key = stub.generate_scope_key(pd.Index(['s0', 's1']))
        self.assertIsInstance(key, str)
        {key: True}

    def test_key_can_always_be_used_as_dict_key(self):
        """Regression test: all scope types must produce valid dict keys."""
        stub = self._make(['s0', 's1', 's2'])
        scopes = [
            None,
            0,
            's0',
            ['s0', 's1'],
            ('s0',),
            slice(None),
            np.array(['s0', 's1']),
            pd.Index(['s0']),
            ['nonexistent1', 'nonexistent2'],
        ]
        result_dict = {}
        for scope in scopes:
            key = stub.generate_scope_key(scope)
            self.assertIsInstance(key, str, f'scope={scope} produced non-string key: {key!r}')
            result_dict[key] = scope


if __name__ == '__main__':
    unittest.main()
