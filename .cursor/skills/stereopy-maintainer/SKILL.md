---
name: stereopy-maintainer
description: >-
  Stereopy project maintenance guide for code review, bug fixing, and feature development.
  Use when working on stereo/ source code, fixing GitHub issues, reviewing PRs,
  adding tools or algorithms, modifying I/O formats, or debugging data pipeline errors.
---

# Stereopy Maintainer Skill

## Architecture at a Glance

```
StereoExpData (stereo/core/stereo_exp_data.py)
│  Core container: exp_matrix, cells, genes, position
│  exp_matrix: np.ndarray | scipy.sparse.spmatrix
│  bin_type: 'bins' | 'cell_bins'
│
├── .tl → StPipeline (stereo/core/st_pipeline.py)
│   │  Runs tools, stores results
│   └── .result → Result (stereo/core/result.py)
│       Dict-like, keys categorized by type:
│       CLUSTER: leiden, louvain, phenograph, annotation
│       REDUCE: umap, pca, tsne
│       CONNECTIVITY: neighbors
│       HVG: highly_variable_genes → renamed 'hvg'
│       MARKER_GENES: marker_genes → renamed 'rank_genes_groups'
│       SCT: sctransform
│
├── AnnBasedStereoExpData (AnnData-backed variant)
│   └── .tl.result → AnnBasedResult
│
└── MSData (stereo/core/ms_data.py)
    │  Multi-sample container, holds multiple StereoExpData
    └── .tl → MSDataPipeLine (stereo/core/ms_pipeline.py)
```

## Critical Type Guards

Always check these before operating — runtime types vary:

| Variable | Possible Types | Guard |
|----------|---------------|-------|
| `exp_matrix` | `np.ndarray`, `scipy.sparse.*` | `issparse(m)` |
| `data.tl.result[key]` | `dict`, `pd.DataFrame` | `isinstance(v, dict)` |
| `Cell/Gene .to_df()` columns | `str`, `object` | `.astype(str)` |
| H5AD format | standard AnnData, Stereopy-extended | check for `@` keys |

## Module Map

| Module | Path | Responsibility |
|--------|------|---------------|
| Data model | `stereo/core/stereo_exp_data.py` | `StereoExpData`, `AnnBasedStereoExpData` |
| Cell/Gene | `stereo/core/cell.py`, `gene.py` | Cell and Gene metadata containers |
| Pipeline | `stereo/core/st_pipeline.py` | Tool execution, `@logit` decorator |
| Multi-sample | `stereo/core/ms_data.py`, `ms_pipeline.py` | `MSData`, scope management |
| Results | `stereo/core/result.py` | `Result`, `AnnBasedResult`, key routing |
| Reader | `stereo/io/reader.py` | Multi-format input (h5ad, gef, gem, loom, h5ms) |
| Writer | `stereo/io/writer.py` | Multi-format output |
| H5AD helpers | `stereo/io/h5ad.py` | Low-level HDF5 read/write |
| Tools | `stereo/tools/*.py` | High-level analysis (clustering, markers, dim_reduce) |
| Algorithms | `stereo/algorithm/*.py` | Low-level compute (mannwhitneyu, sctransform) |
| Plots | `stereo/plots/*.py` | Visualization |
| Config | `stereo/stereo_config.py` | Global settings |
| Logging | `stereo/log_manager.py` | `logger` instance |

## Known Bug Patterns

When diagnosing issues, check these patterns first:

1. **KeyError in result.py** — key not registered, or DataFrame column renamed
   - Check `RENAME_DICT`, `CLUSTER_NAMES`, `MARKER_GENES_NAMES`
   - Result value can be `dict` or `DataFrame` — caller must handle both

2. **sparse/dense mismatch** — code assumes `ndarray` but gets `csr_matrix`
   - Always use `issparse()` before `.toarray()`, indexing, or arithmetic
   - `exp_matrix` type depends on file format and preprocessing history

3. **H5AD format confusion** — Stereopy adds custom groups (`exp_matrix@raw`, `sn`, `layers`)
   - Standard AnnData readers won't find these
   - `reader.py` handles both formats, check `isinstance(f[k], h5py.Group)` vs `Dataset`

4. **MSData scope_key** — `generate_scope_key(scope)` not `generate_scope_key(_names)`
   - `scope` is the correct parameter, not the internal `_names` attribute

5. **mannwhitneyu overflow** — NaN/inf or all-zero columns in input
   - Pre-filter with `x_mask` for valid indices

6. **DataFrame column mismatch** — `df['gene_name']` vs `df['genes']` vs `var.index`
   - Use `.loc[df['genes'], 'real_gene_name']` pattern for safe access

## Code Style

- 4-space indentation, no tabs
- Google-style docstrings with `Parameters` / `Returns` sections
- Import order: stdlib → third-party (`numpy`, `pandas`, `scipy`, `anndata`) → local (`stereo.*`)
- Logging: `from stereo.log_manager import logger`
- Type hints optional but encouraged
- Tools registered via `StPipeline` methods with `@logit` decorator

## Development Workflow

### Fixing a Bug

1. **Read the traceback** — extract file path, line number, error type
2. **Locate the code** — read the full function containing the bug
3. **Understand context** — trace data flow through the pipeline
4. **Check type guards** — is it a sparse/dense or dict/DataFrame issue?
5. **Minimal fix** — only change what's broken
6. **Verify syntax** — `python -c "import ast; ast.parse(open('file.py').read())"`
7. **Verify import** — `PYTHONPATH=. python -c "from stereo.module import Class"`
8. **Commit** — `fix #N: brief description`

### Adding a New Tool

1. Create `stereo/tools/your_tool.py`
2. Add method to `StPipeline` in `stereo/core/st_pipeline.py`
3. Register result key in `Result` categories if needed
4. Add corresponding test in `tests/test_your_tool.py`
5. Add plotting if applicable in `stereo/plots/`

### Adding a New Algorithm

1. Create `stereo/algorithm/your_algo.py` or subpackage
2. Wire it from a tool in `stereo/tools/`
3. Handle sparse/dense input explicitly
4. Add to `__init__.py` exports if public

### Modifying I/O

1. Reader changes go in `stereo/io/reader.py`
2. Writer changes go in `stereo/io/writer.py`
3. Low-level HDF5 operations use `stereo/io/h5ad.py`
4. Always handle both `h5py.Group` and `h5py.Dataset` for H5AD keys
5. Test with both standard AnnData and Stereopy-extended H5AD files

## Testing

- Test files: `tests/test_*.py`, pytest style
- Stereopy has heavy dependencies — avoid `pip install stereopy` in CI
- Use `PYTHONPATH=.` for import-based testing
- For quick validation: AST parse + import check
- Full test suite run via `pytest tests/ -x --tb=short`

## File Format Reference

| Format | Ext | Reader Function | Notes |
|--------|-----|----------------|-------|
| H5AD | `.h5ad` | `read_stereo_h5ad` | Stereopy extended with `@` groups |
| H5MS | `.h5ms` | `read_h5ms` | Multi-sample, Stereopy-specific |
| GEF | `.gef` | `read_gef` | BGI spatial format |
| GEM | `.gem` | `read_gem` | Tab-separated text |
| Loom | `.loom` | `read_loom` | HDF5-based |

## Constraints

- Only modify `stereo/` and `tests/`
- Never change `pyproject.toml`, `.github/`, or version numbers
- `requires-python: >=3.8, <3.9` — be careful with newer syntax
- Dependencies managed in `requirements.txt`, not inline
- MIT License
