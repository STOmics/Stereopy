# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Stereopy is a Python 3.8 spatial transcriptomics library. It is a pure Python scientific package (no web services, databases, or microservices). The core data structure is `StereoExpData` backed by `AnnData`.

### Python version

The project strictly requires Python 3.8 (`pyproject.toml`: `>=3.8, <3.9`). The VM has Python 3.8 installed via the deadsnakes PPA, with a virtualenv at `/workspace/.venv`.

### Activating the environment

```bash
source /workspace/.venv/bin/activate
export PYTHONPATH=/workspace:$PYTHONPATH
```

The `PYTHONPATH` export is necessary because `pip install -e .` does not work (the `pyproject.toml` lacks `[tool.setuptools.packages]` config causing setuptools auto-discovery to fail). Using `PYTHONPATH` is the simplest workaround.

### Lint

```bash
flake8 stereo/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

There are pre-existing `F821` errors in the codebase (undefined names in `stereo/algorithm/st_gears/recons.py` and `stereo/utils/data_helper.py`).

### Tests

```bash
cd /workspace/tests
python -m pytest -m "not gpu and not heavy and not cell_cut_env" --durations=0
```

Tests require downloading data from `pan.genomics.cn` (external Chinese genomics data server). If this server is unreachable, all tests that depend on downloaded data will fail with `UnboundLocalError` in `stereo/utils/_download.py`. This is expected behavior in network-restricted environments. The test framework (82 tests collected, 9 deselected) itself works correctly.

### Import verification

```python
import stereo as st
print(st.__version__)  # should print 1.6.2
```

### Key gotchas

- `exp_matrix` can be `np.ndarray` or `scipy.sparse` — several algorithms handle these differently and some paths only work with dense arrays.
- The `sub_by_index` method on `StereoExpData` mutates the object's view; subsetting before running the analysis pipeline can cause dimension mismatches in PCA/neighbors.
- The `umap()` pipeline method does NOT accept `n_pcs` as a keyword argument (unlike what the commented-out old signature suggests). Use `pca_res_key` and `neighbors_res_key` instead.
