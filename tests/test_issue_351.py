"""
Tests for issue #351: LinAlgError: Singular matrix in sctransform.

When a gene has near-zero or constant expression across sampled cells,
the Hessian matrix in qpois_reg becomes singular and np.linalg.inv raises
LinAlgError. The fix uses pinv as fallback and lets one_row_fit_poission
return None on failure, which fit_poisson then replaces with NaN and warns.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from stereo.algorithm.sctransform.utils import (
    fit_poisson,
    one_row_fit_poission,
    qpois_reg,
    make_cell_attr,
)


def _make_regressor(n_cells=50, seed=42):
    rng = np.random.RandomState(seed)
    log_umi = rng.uniform(2, 5, size=n_cells)
    df = pd.DataFrame({"log_umi": log_umi}, index=[f"cell_{i}" for i in range(n_cells)])
    from patsy.highlevel import dmatrix
    return dmatrix("~log_umi", df, return_type='dataframe')


class TestQpoisReg:
    """qpois_reg should not raise LinAlgError even for degenerate inputs."""

    def test_normal_gene(self):
        rng = np.random.RandomState(0)
        n = 50
        log_umi = rng.uniform(2, 5, size=n)
        X = np.column_stack([np.ones(n), log_umi])
        y = rng.poisson(lam=3, size=n).astype(float)
        result = qpois_reg(X, y, tol=1e-9, maxiters=100, minphi=1.0001, returnfit=True)
        assert "coefficients" in result
        assert result["fitted"] is not None
        assert len(result["coefficients"]) == 2

    def test_all_zero_gene_no_linalg_error(self):
        """All-zero expression gene must not raise LinAlgError."""
        n = 50
        log_umi = np.linspace(2, 5, n)
        X = np.column_stack([np.ones(n), log_umi])
        y = np.zeros(n)
        result = qpois_reg(X, y, tol=1e-9, maxiters=100, minphi=1.0001, returnfit=True)
        assert "coefficients" in result

    def test_constant_expression_gene_no_linalg_error(self):
        """Constant expression gene must not raise LinAlgError."""
        n = 50
        log_umi = np.linspace(2, 5, n)
        X = np.column_stack([np.ones(n), log_umi])
        y = np.ones(n) * 2.0
        result = qpois_reg(X, y, tol=1e-9, maxiters=100, minphi=1.0001, returnfit=True)
        assert "coefficients" in result


class TestOneRowFitPoisson:
    """one_row_fit_poission should return None (not raise) for degenerate genes."""

    def test_returns_tuple_for_normal_gene(self):
        rng = np.random.RandomState(1)
        n = 50
        regressor_data = _make_regressor(n_cells=n, seed=1)
        y = rng.poisson(lam=5, size=n).astype(float)
        result = one_row_fit_poission(regressor_data, y, theta_estimation_fun='theta.ml')
        assert result is not None
        assert len(result) == 3
        theta, intercept, log_umi_coef = result
        assert np.isfinite(theta)

    def test_returns_none_for_all_zero_gene(self):
        """All-zero gene: regression should return None without raising."""
        n = 50
        regressor_data = _make_regressor(n_cells=n)
        y = np.zeros(n)
        result = one_row_fit_poission(regressor_data, y, theta_estimation_fun='theta.ml')
        assert result is None


class TestFitPoisson:
    """fit_poisson should handle NaN-producing genes gracefully."""

    def test_all_valid_genes(self):
        rng = np.random.RandomState(2)
        n_genes, n_cells = 10, 50
        counts = rng.poisson(lam=5, size=(n_genes, n_cells)).astype(float)
        umi = csr_matrix(counts)
        log_umi = np.log10(counts.sum(0) + 1)
        data = pd.DataFrame({"log_umi": log_umi}, index=[f"c{i}" for i in range(n_cells)])
        result = fit_poisson(umi, model_str="y~log_umi", data=data)
        assert result.shape == (n_genes, 3)
        assert list(result.columns) == ["theta", "Intercept", "log_umi"]

    def test_mixed_valid_and_zero_genes(self):
        """Dataset with some all-zero genes: result rows exist but those are NaN."""
        rng = np.random.RandomState(3)
        n_cells = 50
        normal_gene = rng.poisson(lam=5, size=(1, n_cells)).astype(float)
        zero_gene = np.zeros((1, n_cells))
        counts = np.vstack([normal_gene, zero_gene])
        umi = csr_matrix(counts)
        log_umi = np.log10(counts.sum(0) + 1)
        data = pd.DataFrame({"log_umi": log_umi}, index=[f"c{i}" for i in range(n_cells)])
        result = fit_poisson(umi, model_str="y~log_umi", data=data)
        assert result.shape == (2, 3)
        assert result.iloc[0].notna().all(), "Normal gene should have valid parameters"
        assert result.iloc[1].isna().all(), "All-zero gene should be NaN"

    def test_no_linalg_error_raised(self):
        """The original issue: all-zero gene should not propagate LinAlgError."""
        rng = np.random.RandomState(4)
        n_cells = 30
        counts = np.zeros((5, n_cells))
        counts[0] = rng.poisson(lam=3, size=n_cells)
        umi = csr_matrix(counts)
        log_umi = np.log10(counts.sum(0) + 1)
        data = pd.DataFrame({"log_umi": log_umi}, index=[f"c{i}" for i in range(n_cells)])
        try:
            result = fit_poisson(umi, model_str="y~log_umi", data=data)
            assert result.shape[0] == 5
        except np.linalg.LinAlgError:
            pytest.fail("fit_poisson should not raise LinAlgError for degenerate genes")
