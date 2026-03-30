"""Test that sctransform handles singular matrices gracefully."""

import numpy as np
import pytest


def test_qpois_reg_singular_matrix():
    """qpois_reg should not crash on degenerate input that produces a singular Hessian."""
    from stereo.algorithm.sctransform.utils import qpois_reg

    n = 50
    X = np.column_stack([np.ones(n), np.log10(np.ones(n))])
    Y = np.zeros(n, dtype=np.double)

    result = qpois_reg(X, Y, 1e-9, 100, 1.0001, True)
    assert 'coefficients' in result
    assert 'fitted' in result
    assert result['fitted'] is not None


def test_one_row_fit_poission_fallback():
    """one_row_fit_poission should return fallback values on degenerate input."""
    import pandas as pd
    from stereo.algorithm.sctransform.utils import one_row_fit_poission

    n = 50
    regressor_data = pd.DataFrame({
        'Intercept': np.ones(n),
        'log_umi': np.log10(np.ones(n))
    })
    y = np.zeros(n, dtype=np.double)

    theta, intercept, log_umi = one_row_fit_poission(regressor_data, y)
    assert np.isfinite(theta)
    assert np.isfinite(intercept)
    assert np.isfinite(log_umi)
