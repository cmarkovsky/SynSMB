"""
test_gaussianize.py
===================
Tests for GaussianTransform.

Run directly for step-by-step printed output:
    python test_gaussianize.py

Run via pytest for pass/fail:
    pytest test_gaussianize.py -v
"""

import numpy as np
import xarray as xr
import pytest
from scipy.stats import norm
from syn_smb.core.gaussianize import GaussianTransform


# ======================================================================
# Fixtures — shared test data
# ======================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def normal_data(rng):
    """Simple N(0,1) data — easy to verify transform behavior."""
    return rng.standard_normal(500)


@pytest.fixture
def skewed_data(rng):
    """
    Right-skewed data — mimics real SMB residuals.
    Lognormal shifted to have both positive and negative values.
    """
    x = rng.lognormal(mean=0.0, sigma=0.5, size=500)
    return x - np.mean(x)


@pytest.fixture
def smb_like_data(rng):
    """
    Closer to actual PIG SMB residuals:
    right-skewed, ~500 points (45 yrs × 12 months), small values.
    """
    x = rng.lognormal(mean=-3.0, sigma=0.8, size=540)
    return x - np.mean(x)


@pytest.fixture
def fitted_transform(skewed_data):
    """A GaussianTransform already fitted on skewed data."""
    gt = GaussianTransform()
    gt.fit(skewed_data)
    return gt, skewed_data


@pytest.fixture
def xarray_data(skewed_data):
    """Same skewed data wrapped as an xarray DataArray with a time coord."""
    import pandas as pd
    time = xr.cftime_range(start="1979", periods=len(skewed_data), freq="MS")
    return xr.DataArray(skewed_data, coords={"time": time}, dims=["time"], name="smb_resid")


# ======================================================================
# 1. fit()
# ======================================================================

class TestFit:

    def test_sets_n(self, skewed_data):
        gt = GaussianTransform()
        gt.fit(skewed_data)
        assert gt.n == len(skewed_data)

    def test_x_sorted_is_sorted(self, skewed_data):
        gt = GaussianTransform()
        gt.fit(skewed_data)
        assert np.all(np.diff(gt.x_sorted) >= 0), "x_sorted should be non-decreasing"

    def test_x_sorted_contains_all_values(self, skewed_data):
        gt = GaussianTransform()
        gt.fit(skewed_data)
        # sorted values should be a permutation of the input
        np.testing.assert_array_equal(gt.x_sorted, np.sort(skewed_data))

    def test_quantiles_strictly_in_unit_interval(self, skewed_data):
        gt = GaussianTransform()
        gt.fit(skewed_data)
        assert np.all(gt.quantiles > 0), "quantiles should be > 0 (no -inf from norm.ppf)"
        assert np.all(gt.quantiles < 1), "quantiles should be < 1 (no +inf from norm.ppf)"

    def test_quantiles_length_matches_n(self, skewed_data):
        gt = GaussianTransform()
        gt.fit(skewed_data)
        assert len(gt.quantiles) == gt.n

    def test_is_fitted_flag(self, skewed_data):
        gt = GaussianTransform()
        assert not gt.is_fitted
        gt.fit(skewed_data)
        assert gt.is_fitted

    def test_returns_self_for_chaining(self, skewed_data):
        gt = GaussianTransform()
        result = gt.fit(skewed_data)
        assert result is gt, "fit() should return self for method chaining"

    def test_raises_on_too_few_observations(self):
        gt = GaussianTransform()
        with pytest.raises(ValueError, match="Too few observations"):
            gt.fit(np.array([1.0, 2.0, 3.0]))

    def test_accepts_xarray_input(self, xarray_data):
        gt = GaussianTransform()
        gt.fit(xarray_data)
        assert gt.is_fitted
        assert gt.n == len(xarray_data)


# ======================================================================
# 2. transform()
# ======================================================================

class TestTransform:

    def test_output_is_approximately_normal(self, fitted_transform):
        from scipy.stats import ks_1samp, norm
        gt, data = fitted_transform
        g = gt.transform(data)
        g_vals = g if isinstance(g, np.ndarray) else g.values
        _, pvalue = ks_1samp(g_vals, norm.cdf)
        assert pvalue > 0.05, f"KS test rejected normality (p={pvalue:.4f})"

    def test_output_mean_near_zero(self, fitted_transform):
        gt, data = fitted_transform
        g = gt.transform(data)
        g_vals = g if isinstance(g, np.ndarray) else g.values
        assert abs(np.mean(g_vals)) < 0.1, f"Transform mean too large: {np.mean(g_vals):.4f}"

    def test_output_std_near_one(self, fitted_transform):
        gt, data = fitted_transform
        g = gt.transform(data)
        g_vals = g if isinstance(g, np.ndarray) else g.values
        assert abs(np.std(g_vals) - 1.0) < 0.1, f"Transform std too far from 1: {np.std(g_vals):.4f}"

    def test_preserves_rank_ordering(self, fitted_transform):
        gt, data = fitted_transform
        g = gt.transform(data)
        g_vals = g if isinstance(g, np.ndarray) else g.values
        # rank order of input should equal rank order of output
        np.testing.assert_array_equal(
            np.argsort(data),
            np.argsort(g_vals),
            err_msg="transform() should preserve rank ordering"
        )

    def test_output_has_no_inf_or_nan(self, fitted_transform):
        gt, data = fitted_transform
        g = gt.transform(data)
        g_vals = g if isinstance(g, np.ndarray) else g.values
        assert not np.any(np.isnan(g_vals)), "transform output contains NaN"
        assert not np.any(np.isinf(g_vals)), "transform output contains Inf"

    def test_raises_when_not_fitted(self, skewed_data):
        gt = GaussianTransform()
        with pytest.raises(RuntimeError, match="not fitted"):
            gt.transform(skewed_data)

    def test_numpy_input_returns_numpy(self, fitted_transform):
        gt, data = fitted_transform
        result = gt.transform(data)
        assert isinstance(result, np.ndarray)

    def test_xarray_input_returns_xarray(self, skewed_data, xarray_data):
        gt = GaussianTransform()
        gt.fit(skewed_data)
        result = gt.transform(xarray_data)
        assert isinstance(result, xr.DataArray)

    def test_xarray_output_preserves_coords(self, skewed_data, xarray_data):
        gt = GaussianTransform()
        gt.fit(skewed_data)
        result = gt.transform(xarray_data)
        assert result.dims == xarray_data.dims
        np.testing.assert_array_equal(result.coords["time"], xarray_data.coords["time"])

    def test_different_length_input_works(self, fitted_transform):
        """transform() on data of different length than fit() should not raise."""
        gt, data = fitted_transform
        shorter = data[:100]
        result = gt.transform(shorter)
        assert len(result) == 100


# ======================================================================
# 3. inverse_transform()
# ======================================================================

class TestInverseTransform:

    def test_output_is_zero_mean(self, fitted_transform, rng):
        gt, data = fitted_transform
        g = np.array(gt.transform(data))
        r = gt.inverse_transform(g)
        assert abs(np.mean(r)) < 1e-10, (
            f"inverse_transform output is not zero-mean: mean = {np.mean(r):.6e}"
        )

    def test_zero_mean_holds_for_scaled_input(self, fitted_transform, rng):
        gt, data = fitted_transform
        g = np.array(gt.transform(data))
        for scale in [0.1, 2.0, 5.0, 10.0]:
            r = gt.inverse_transform(g * scale)
            assert abs(np.mean(r)) < 1e-10, (
                f"inverse_transform not zero-mean at scale={scale}: mean={np.mean(r):.6e}"
            )

    def test_roundtrip_recovers_centered_data(self, fitted_transform):
        gt, data = fitted_transform
        g = np.array(gt.transform(data))
        r = gt.inverse_transform(g)
        x_centered = data - np.mean(data)
        max_err = np.max(np.abs(r - x_centered))
        assert max_err < 1e-6, f"Round-trip error too large: {max_err:.2e}"

    def test_std_scales_beyond_unit_scale(self, fitted_transform, rng):
        """
        Semi-parametric tails fix: std should keep growing at high scale factors,
        not saturate at the observed max as with the old purely empirical approach.
        """
        gt, data = fitted_transform
        g = np.array(gt.transform(data))

        stds = {}
        for scale in [1.0, 2.0, 5.0, 10.0]:
            r = gt.inverse_transform(g * scale)
            stds[scale] = float(np.std(r))

        # std at 2x should be larger than at 1x
        assert stds[2.0] > stds[1.0], "std should grow when scale increases from 1x to 2x"
        # std at 10x should be larger than at 2x (old clipping approach would saturate here)
        assert stds[10.0] > stds[2.0], (
            f"std saturated between 2x and 10x: std(2x)={stds[2.0]:.4f}, std(10x)={stds[10.0]:.4f}\n"
            "This suggests the old clipping approach is still active."
        )

    def test_tail_continuity_lower(self, fitted_transform):
        """
        At the lower splice point, empirical and parametric branches must agree.
        """
        gt, _ = fitted_transform
        p_low = float(gt.quantiles[0])
        empirical  = float(gt.x_sorted[0])
        parametric = float(norm.ppf(p_low, loc=gt.tail_loc, scale=gt.tail_scale)
                           + gt.lower_tail_offset)
        assert abs(empirical - parametric) < 1e-10, (
            f"Lower splice discontinuity: empirical={empirical:.6f}, "
            f"parametric={parametric:.6f}, diff={abs(empirical-parametric):.2e}"
        )

    def test_tail_continuity_upper(self, fitted_transform):
        """
        At the upper splice point, empirical and parametric branches must agree.
        """
        gt, _ = fitted_transform
        p_high = float(gt.quantiles[-1])
        empirical  = float(gt.x_sorted[-1])
        parametric = float(norm.ppf(p_high, loc=gt.tail_loc, scale=gt.tail_scale)
                           + gt.upper_tail_offset)
        assert abs(empirical - parametric) < 1e-10, (
            f"Upper splice discontinuity: empirical={empirical:.6f}, "
            f"parametric={parametric:.6f}, diff={abs(empirical-parametric):.2e}"
        )

    def test_raises_when_not_fitted(self, skewed_data):
        gt = GaussianTransform()
        with pytest.raises(RuntimeError, match="not fitted"):
            gt.inverse_transform(skewed_data)

    def test_numpy_input_returns_numpy(self, fitted_transform):
        gt, data = fitted_transform
        g = np.array(gt.transform(data))
        result = gt.inverse_transform(g)
        assert isinstance(result, np.ndarray)

    def test_xarray_input_returns_xarray(self, skewed_data, xarray_data):
        gt = GaussianTransform()
        gt.fit(skewed_data)
        g_xr = gt.transform(xarray_data)
        result = gt.inverse_transform(g_xr)
        assert isinstance(result, xr.DataArray)
        assert result.dims == xarray_data.dims


# ======================================================================
# 4. validate()
# ======================================================================

class TestValidate:

    def test_returns_expected_keys(self, fitted_transform):
        gt, data = fitted_transform
        results = gt.validate(data, verbose=False)
        expected_keys = {
            "ks_statistic", "ks_pvalue", "transform_mean",
            "transform_std", "roundtrip_max_err",
            "lower_splice_err", "upper_splice_err", "passed"
        }
        assert set(results.keys()) == expected_keys

    def test_passes_on_good_data(self, fitted_transform):
        gt, data = fitted_transform
        results = gt.validate(data, verbose=False)
        assert results["passed"], f"validate() failed:\n{results}"

    def test_roundtrip_error_is_small(self, fitted_transform):
        gt, data = fitted_transform
        results = gt.validate(data, verbose=False)
        assert results["roundtrip_max_err"] < 1e-6

    def test_splice_errors_are_negligible(self, fitted_transform):
        gt, data = fitted_transform
        results = gt.validate(data, verbose=False)
        assert results["lower_splice_err"] < 1e-10
        assert results["upper_splice_err"] < 1e-10

    def test_raises_when_not_fitted(self, skewed_data):
        gt = GaussianTransform()
        with pytest.raises(RuntimeError, match="not fitted"):
            gt.validate(skewed_data)


# ======================================================================
# 5. tail attributes set by fit()
# ======================================================================

class TestTailFitting:

    def test_tail_attributes_set_after_fit(self, fitted_transform):
        gt, _ = fitted_transform
        assert gt.tail_loc is not None
        assert gt.tail_scale is not None
        assert gt.lower_tail_offset is not None
        assert gt.upper_tail_offset is not None

    def test_tail_scale_is_positive(self, fitted_transform):
        gt, _ = fitted_transform
        assert gt.tail_scale > 0

    def test_tail_loc_near_data_mean(self, skewed_data):
        gt = GaussianTransform()
        gt.fit(skewed_data)
        # tail_loc should be close to the sample mean (MLE estimate)
        assert abs(gt.tail_loc - np.mean(skewed_data)) < 1e-6


# ======================================================================
# 5. repr
# ======================================================================

class TestRepr:

    def test_repr_unfitted(self):
        gt = GaussianTransform()
        assert "fitted=False" in repr(gt)

    def test_repr_fitted(self, fitted_transform):
        gt, _ = fitted_transform
        r = repr(gt)
        assert "fitted=True" in r
        assert "n=" in r
        assert "x_range" in r


# ======================================================================
# Standalone debug runner
# Prints step-by-step output so you can inspect intermediate values.
# Run with: python test_gaussianize.py
# ======================================================================

def run_debug():
    import pandas as pd

    print("=" * 60)
    print("GaussianTransform — step-by-step debug run")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # --- Build some SMB-like data ---
    print("\n[1] Building test data (right-skewed, SMB-like)...")
    x_raw = rng.lognormal(mean=-3.0, sigma=0.8, size=540)
    x = x_raw - np.mean(x_raw)
    print(f"    n={len(x)}, mean={np.mean(x):.6f}, std={np.std(x):.4f}, "
          f"min={np.min(x):.4f}, max={np.max(x):.4f}")

    # --- fit() ---
    print("\n[2] fit()...")
    gt = GaussianTransform()
    gt.fit(x)
    print(f"    n stored:              {gt.n}")
    print(f"    quantiles range:       [{gt.quantiles[0]:.6f}, {gt.quantiles[-1]:.6f}]")
    print(f"    x_sorted range:        [{gt.x_sorted[0]:.4f}, {gt.x_sorted[-1]:.4f}]")
    print(f"    tail normal fit:       N({gt.tail_loc:.4f}, {gt.tail_scale:.4f})")
    print(f"    lower_tail_offset:     {gt.lower_tail_offset:.6f}")
    print(f"    upper_tail_offset:     {gt.upper_tail_offset:.6f}")
    print(f"    is_fitted:             {gt.is_fitted}")

    # --- transform() ---
    print("\n[3] transform()...")
    g = gt.transform(x)
    print(f"    output mean:  {np.mean(g):+.6f}  (target: ~0)")
    print(f"    output std:   {np.std(g):.6f}   (target: ~1)")
    print(f"    output min:   {np.min(g):.4f}")
    print(f"    output max:   {np.max(g):.4f}")
    print(f"    any NaN/Inf:  {np.any(np.isnan(g)) or np.any(np.isinf(g))}")

    # --- inverse_transform() on training data ---
    print("\n[4] inverse_transform(transform(x)) — round-trip check...")
    r = gt.inverse_transform(g)
    x_centered = x - np.mean(x)
    max_err = np.max(np.abs(r - x_centered))
    print(f"    recovered mean:     {np.mean(r):+.2e}  (target: ~0, mean shift fix)")
    print(f"    max round-trip err: {max_err:.2e}  (tolerance: 1e-6)")

    # --- inverse_transform() on SCALED Gaussian — simulates band scaling ---
    print("\n[5] Semi-parametric tails — inverse_transform on scaled Gaussian inputs...")
    print("    std should now keep growing (no saturation at observed max).")
    prev_std = None
    for scale in [0.1, 1.0, 2.0, 5.0, 10.0]:
        g_scaled = g * scale
        r_scaled = gt.inverse_transform(g_scaled)
        current_std = np.std(r_scaled)
        growing = "" if prev_std is None else ("✓" if current_std > prev_std else "✗ SATURATED")
        print(f"    scale={scale:4.1f}x  →  mean: {np.mean(r_scaled):+.2e}  "
              f"std: {current_std:.5f}  {growing}")
        prev_std = current_std

    # --- xarray input ---
    print("\n[6] xarray input/output preservation...")
    time = xr.cftime_range(start="1979", periods=len(x), freq="MS")
    x_da = xr.DataArray(x, coords={"time": time}, dims=["time"], name="smb_resid")
    g_da = gt.transform(x_da)
    r_da = gt.inverse_transform(g_da)
    print(f"    input type:   {type(x_da).__name__}")
    print(f"    output type (transform):         {type(g_da).__name__}")
    print(f"    output type (inverse_transform): {type(r_da).__name__}")
    print(f"    coords preserved: {list(r_da.coords)}")

    # --- validate() ---
    print("\n[7] validate()...")
    results = gt.validate(x, verbose=True)

    print("\n" + "=" * 60)
    print("Debug run complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_debug()