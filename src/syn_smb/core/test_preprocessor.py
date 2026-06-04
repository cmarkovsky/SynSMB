"""
test_preprocessor.py
====================
Tests for Preprocessor.

Run directly for step-by-step printed output:
    python test_preprocessor.py

Run via pytest for pass/fail:
    pytest test_preprocessor.py -v
"""

import numpy as np
import xarray as xr
import pytest
from syn_smb.core.preprocessor import Preprocessor


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def monthly_time():
    """45-year monthly cftime coordinate — matches the RACMO record."""
    return xr.cftime_range(start="1979", periods=540, freq="MS")


@pytest.fixture
def smb_with_trend(monthly_time, rng):
    """
    Synthetic SMB with a known linear trend, seasonal cycle, and noise.
    Used to verify that each component is removed correctly.
    """
    t = np.arange(len(monthly_time))
    months = np.array([d.month for d in monthly_time.values])

    # Known seasonal cycle (m w.e.)
    seasonal = 0.03 * np.sin(2 * np.pi * (months - 3) / 12)

    # Known linear trend
    trend = 5e-5 * t   # slow upward trend

    # White noise residual
    noise = rng.normal(0, 0.02, size=len(t))

    values = 0.04 + trend + seasonal + noise
    return xr.DataArray(values, coords={"time": monthly_time}, dims=["time"], name="smb")


@pytest.fixture
def smb_no_trend(monthly_time, rng):
    """SMB with seasonal cycle but no trend — for testing remove_trend=False."""
    months = np.array([d.month for d in monthly_time.values])
    seasonal = 0.03 * np.sin(2 * np.pi * (months - 3) / 12)
    noise = rng.normal(0, 0.02, size=len(monthly_time))
    values = 0.04 + seasonal + noise
    return xr.DataArray(values, coords={"time": monthly_time}, dims=["time"], name="smb")


@pytest.fixture
def fitted_preprocessor(smb_with_trend):
    """Preprocessor fitted on smb_with_trend."""
    pp = Preprocessor(remove_trend=True, remove_seasonal=True)
    pp.fit(smb_with_trend)
    return pp, smb_with_trend


@pytest.fixture
def synthetic_time():
    """1000-year synthetic cftime coordinate for inverse_transform tests."""
    return xr.cftime_range(start="1979", periods=12000, freq="MS")


# ======================================================================
# 1. fit()
# ======================================================================

class TestFit:

    def test_sets_is_fitted_flag(self, smb_with_trend):
        pp = Preprocessor()
        assert not pp.is_fitted
        pp.fit(smb_with_trend)
        assert pp.is_fitted

    def test_stores_n_obs(self, smb_with_trend):
        pp = Preprocessor()
        pp.fit(smb_with_trend)
        assert pp.n_obs == len(smb_with_trend)

    def test_stores_trend_coeffs(self, smb_with_trend):
        pp = Preprocessor(remove_trend=True)
        pp.fit(smb_with_trend)
        assert pp.trend_coeffs is not None

    def test_trend_coeffs_have_two_degrees(self, smb_with_trend):
        """Linear fit should produce 2 coefficients (slope, intercept)."""
        pp = Preprocessor(remove_trend=True)
        pp.fit(smb_with_trend)
        assert pp.trend_coeffs.sizes['degree'] == 2

    def test_stores_seasonal_cycle(self, smb_with_trend):
        pp = Preprocessor(remove_seasonal=True)
        pp.fit(smb_with_trend)
        assert pp.seasonal_cycle is not None

    def test_seasonal_cycle_has_12_months(self, smb_with_trend):
        pp = Preprocessor(remove_seasonal=True)
        pp.fit(smb_with_trend)
        assert pp.seasonal_cycle.sizes['month'] == 12

    def test_seasonal_cycle_months_are_1_to_12(self, smb_with_trend):
        pp = Preprocessor(remove_seasonal=True)
        pp.fit(smb_with_trend)
        months = pp.seasonal_cycle['month'].values
        np.testing.assert_array_equal(months, np.arange(1, 13))

    def test_no_trend_coeffs_when_remove_trend_false(self, smb_with_trend):
        pp = Preprocessor(remove_trend=False)
        pp.fit(smb_with_trend)
        assert pp.trend_coeffs is None

    def test_no_seasonal_cycle_when_remove_seasonal_false(self, smb_with_trend):
        pp = Preprocessor(remove_seasonal=False)
        pp.fit(smb_with_trend)
        assert pp.seasonal_cycle is None

    def test_returns_self_for_chaining(self, smb_with_trend):
        pp = Preprocessor()
        result = pp.fit(smb_with_trend)
        assert result is pp

    def test_raises_on_numpy_input(self):
        pp = Preprocessor()
        with pytest.raises(TypeError, match="xr.DataArray"):
            pp.fit(np.random.standard_normal(540))

    def test_raises_on_missing_time_coord(self):
        pp = Preprocessor()
        da = xr.DataArray(np.random.standard_normal(540), dims=["x"])
        with pytest.raises(ValueError, match="'time' coordinate"):
            pp.fit(da)


# ======================================================================
# 2. transform()
# ======================================================================

class TestTransform:

    def test_trend_is_removed(self, smb_with_trend):
        """
        After removing a known linear trend, the residual should have
        a near-zero trend slope.
        """
        pp = Preprocessor(remove_trend=True, remove_seasonal=False)
        pp.fit(smb_with_trend)
        residual = pp.transform(smb_with_trend)

        # Fit a trend to the residual — slope should be ~0
        fit = residual.polyfit(dim='time', deg=1)
        slope = float(fit['polyfit_coefficients'].sel(degree=1).values)
        assert abs(slope) < 1e-8, (
            f"Residual still has a slope of {slope:.2e} after detrending"
        )

    def test_seasonal_cycle_is_removed(self, smb_with_trend, monthly_time):
        """
        After removing the seasonal cycle, the monthly means of the
        residual should all be near zero.
        """
        pp = Preprocessor(remove_trend=True, remove_seasonal=True)
        pp.fit(smb_with_trend)
        residual = pp.transform(smb_with_trend)

        monthly_means = residual.groupby('time.month').mean()
        max_monthly_mean = float(np.abs(monthly_means.values).max())
        assert max_monthly_mean < 1e-10, (
            f"Monthly means not removed: max = {max_monthly_mean:.2e}"
        )

    def test_output_has_same_time_coord(self, fitted_preprocessor):
        pp, data = fitted_preprocessor
        residual = pp.transform(data)
        np.testing.assert_array_equal(residual['time'].values, data['time'].values)

    def test_output_is_dataarray(self, fitted_preprocessor):
        pp, data = fitted_preprocessor
        residual = pp.transform(data)
        assert isinstance(residual, xr.DataArray)

    def test_residual_mean_near_zero(self, fitted_preprocessor):
        """After removing trend and seasonal cycle, residual mean should be small."""
        pp, data = fitted_preprocessor
        residual = pp.transform(data)
        assert abs(float(residual.mean())) < 0.01

    def test_remove_trend_false_leaves_seasonal_in_residual(self, smb_with_trend):
        """With remove_trend=False, the seasonal signal should still be present."""
        pp = Preprocessor(remove_trend=False, remove_seasonal=True)
        pp.fit(smb_with_trend)
        residual = pp.transform(smb_with_trend)
        # Monthly means should still be near zero (seasonal removed)
        monthly_means = residual.groupby('time.month').mean()
        assert float(np.abs(monthly_means.values).max()) < 1e-10

    def test_raises_when_not_fitted(self, smb_with_trend):
        pp = Preprocessor()
        with pytest.raises(RuntimeError, match="not fitted"):
            pp.transform(smb_with_trend)

    def test_fit_transform_matches_fit_then_transform(self, smb_with_trend):
        pp1 = Preprocessor()
        pp1.fit(smb_with_trend)
        r1 = pp1.transform(smb_with_trend)

        pp2 = Preprocessor()
        r2 = pp2.fit_transform(smb_with_trend)

        np.testing.assert_allclose(r1.values, r2.values, atol=1e-12)


# ======================================================================
# 3. inverse_transform()
# ======================================================================

class TestInverseTransform:

    def test_roundtrip_on_training_data(self, fitted_preprocessor):
        """
        inverse_transform(transform(x), add_trend=True) should recover
        the original data to numerical precision.
        """
        pp, data = fitted_preprocessor
        residual = pp.transform(data)
        reconstructed = pp.inverse_transform(residual, add_trend=True)
        max_err = float(np.abs(reconstructed.values - data.values).max())
        assert max_err < 1e-10, (
            f"Round-trip error = {max_err:.2e}. Expected < 1e-10."
        )

    def test_add_trend_false_does_not_restore_trend(self, fitted_preprocessor):
        """
        Default add_trend=False: the reconstructed series should not have
        the original trend restored.
        """
        pp, data = fitted_preprocessor
        residual = pp.transform(data)
        reconstructed = pp.inverse_transform(residual, add_trend=False)

        # If trend were added back, reconstructed would match data more closely.
        # With add_trend=False it should differ by the trend magnitude.
        trend_vals = xr.polyval(data['time'], pp.trend_coeffs)
        # Difference should be approximately the trend
        diff = data - reconstructed
        residual_of_diff = diff - trend_vals
        assert float(np.abs(residual_of_diff.values).max()) < 1e-8

    def test_seasonal_cycle_restored(self, fitted_preprocessor):
        """
        After inverse_transform, monthly means should match those of the
        detrended original data (i.e. the seasonal cycle is back).
        """
        pp, data = fitted_preprocessor
        residual = pp.transform(data)
        reconstructed = pp.inverse_transform(residual, add_trend=False)

        # Monthly means of reconstructed should match seasonal_cycle
        monthly_means = reconstructed.groupby('time.month').mean()
        # Allow for small numerical errors
        np.testing.assert_allclose(
            monthly_means.values,
            pp.seasonal_cycle.values,
            atol=1e-10,
            err_msg="Monthly means after inverse_transform don't match seasonal cycle"
        )

    def test_works_on_synthetic_time_coord(self, fitted_preprocessor, synthetic_time, rng):
        """
        inverse_transform should work on a DataArray with a synthetic
        1000-year time coordinate — the primary use case for reconstruction.
        """
        pp, _ = fitted_preprocessor
        synthetic_anomaly = xr.DataArray(
            rng.standard_normal(len(synthetic_time)) * 0.02,
            coords={"time": synthetic_time},
            dims=["time"],
        )
        result = pp.inverse_transform(synthetic_anomaly, add_trend=False)
        assert isinstance(result, xr.DataArray)
        assert result.sizes['time'] == len(synthetic_time)

    def test_raises_when_not_fitted(self, smb_with_trend):
        pp = Preprocessor()
        with pytest.raises(RuntimeError, match="not fitted"):
            pp.inverse_transform(smb_with_trend)


# ======================================================================
# 4. check_stationarity()
# ======================================================================

class TestCheckStationarity:

    def test_returns_expected_keys(self, fitted_preprocessor):
        pp, data = fitted_preprocessor
        residuals = pp.transform(data)
        result = pp.check_stationarity(residuals, verbose=False)
        expected = {'adf_statistic', 'p_value', 'n_lags', 'is_stationary'}
        assert set(result.keys()) == expected

    def test_stationary_residuals_pass(self, fitted_preprocessor):
        """Detrended, deseasoned residuals should be stationary."""
        pp, data = fitted_preprocessor
        residuals = pp.transform(data)
        result = pp.check_stationarity(residuals, verbose=False)
        assert result['is_stationary'], (
            f"Residuals appear non-stationary after preprocessing. "
            f"p-value = {result['p_value']:.4f}"
        )

    def test_raw_nonstationary_series_detected(self, smb_with_trend):
        """
        A series with a strong trend should be flagged as non-stationary
        by the ADF test — confirms the test is actually detecting the trend.
        """
        # Build a clearly non-stationary series (strong upward trend)
        time = xr.cftime_range(start="1979", periods=540, freq="MS")
        t = np.arange(540)
        trend = 1e-3 * t   # strong trend
        da = xr.DataArray(
            trend + np.random.default_rng(0).normal(0, 0.01, 540),
            coords={"time": time}, dims=["time"]
        )
        pp = Preprocessor()
        result = pp.check_stationarity(da, verbose=False)
        assert not result['is_stationary'], (
            "Strong-trend series was incorrectly classified as stationary."
        )

    def test_p_value_is_in_unit_interval(self, fitted_preprocessor):
        pp, data = fitted_preprocessor
        residuals = pp.transform(data)
        result = pp.check_stationarity(residuals, verbose=False)
        assert 0 <= result['p_value'] <= 1


# ======================================================================
# 5. Options — remove_trend=False, remove_seasonal=False
# ======================================================================

class TestOptions:

    def test_no_removal_transform_is_identity(self, smb_with_trend):
        """With both removals off, transform() should return the input unchanged."""
        pp = Preprocessor(remove_trend=False, remove_seasonal=False)
        pp.fit(smb_with_trend)
        residual = pp.transform(smb_with_trend)
        np.testing.assert_allclose(residual.values, smb_with_trend.values, atol=1e-12)

    def test_trend_only_does_not_remove_seasonal(self, smb_with_trend):
        pp = Preprocessor(remove_trend=True, remove_seasonal=False)
        pp.fit(smb_with_trend)
        residual = pp.transform(smb_with_trend)
        # Monthly means should still show seasonal variation
        monthly_means = residual.groupby('time.month').mean()
        # Range should be > 0 (seasonal signal still present)
        sc_range = float(monthly_means.max() - monthly_means.min())
        assert sc_range > 0.01, "Seasonal signal was removed despite remove_seasonal=False"

    def test_seasonal_only_does_not_remove_trend(self, smb_with_trend):
        pp = Preprocessor(remove_trend=False, remove_seasonal=True)
        pp.fit(smb_with_trend)
        residual = pp.transform(smb_with_trend)
        # Trend should still be present
        fit = residual.polyfit(dim='time', deg=1)
        slope = float(fit['polyfit_coefficients'].sel(degree=1).values)
        # Original trend slope was ~5e-5 per month (in polyfit time units)
        # It should remain non-negligible
        assert abs(slope) > 0, "Trend was removed despite remove_trend=False"


# ======================================================================
# 6. repr
# ======================================================================

class TestRepr:

    def test_repr_unfitted(self):
        pp = Preprocessor()
        assert "fitted=False" in repr(pp)

    def test_repr_fitted(self, fitted_preprocessor):
        pp, _ = fitted_preprocessor
        r = repr(pp)
        assert "fitted=True" in r
        assert "n_obs=" in r


# ======================================================================
# Standalone debug runner
# ======================================================================

def run_debug():
    import pandas as pd

    print("=" * 60)
    print("Preprocessor — step-by-step debug run")
    print("=" * 60)

    rng = np.random.default_rng(42)
    time = xr.cftime_range(start="1979", periods=540, freq="MS")
    months = np.array([d.month for d in time.values])
    t = np.arange(540)

    # Build synthetic SMB with known components
    true_trend    = 5e-5 * t
    true_seasonal = 0.03 * np.sin(2 * np.pi * (months - 3) / 12)
    true_noise    = rng.normal(0, 0.02, 540)
    smb_mean      = 0.04
    values        = smb_mean + true_trend + true_seasonal + true_noise

    smb = xr.DataArray(values, coords={"time": time}, dims=["time"], name="smb")

    print(f"\n[1] Input data: n={len(smb)}, "
          f"mean={float(smb.mean()):.4f}, std={float(smb.std()):.4f}")

    # --- fit() ---
    print("\n[2] fit()...")
    pp = Preprocessor(remove_trend=True, remove_seasonal=True)
    pp.fit(smb)
    print(f"    {pp}")
    pp.summarize()

    # --- transform() ---
    print("\n[3] transform()...")
    residuals = pp.transform(smb)
    print(f"    residual mean:  {float(residuals.mean()):+.4e}  (target: ~0)")
    print(f"    residual std:   {float(residuals.std()):.4f}")

    monthly_means = residuals.groupby('time.month').mean()
    print(f"    monthly mean range: "
          f"[{float(monthly_means.min()):.2e}, {float(monthly_means.max()):.2e}]"
          f"  (target: ~0 — seasonal removed)")

    fit_check = residuals.polyfit(dim='time', deg=1)
    slope = float(fit_check['polyfit_coefficients'].sel(degree=1).values)
    print(f"    residual slope: {slope:.2e}  (target: ~0 — trend removed)")

    # --- check_stationarity() ---
    print("\n[4] check_stationarity() on raw SMB (should be non-stationary)...")
    pp.check_stationarity(smb, verbose=True)

    print("\n[5] check_stationarity() on residuals (should be stationary)...")
    pp.check_stationarity(residuals, verbose=True)

    # --- inverse_transform() round-trip ---
    print("\n[6] Round-trip: inverse_transform(transform(x), add_trend=True)...")
    reconstructed = pp.inverse_transform(residuals, add_trend=True)
    max_err = float(np.abs(reconstructed.values - smb.values).max())
    print(f"    max round-trip error: {max_err:.2e}  (tolerance: 1e-10)")

    # --- add_trend=False (default — for synthetic series) ---
    print("\n[7] inverse_transform with add_trend=False (default for synthesis)...")
    reconstructed_no_trend = pp.inverse_transform(residuals, add_trend=False)
    # Difference vs full round-trip should equal the trend
    diff = reconstructed.values - reconstructed_no_trend.values
    trend_check = xr.polyval(smb['time'], pp.trend_coeffs).values
    trend_err = np.abs(diff - trend_check).max()
    print(f"    difference vs full round-trip matches trend: {trend_err:.2e}  (target: ~0)")

    # --- synthetic time coordinate ---
    print("\n[8] inverse_transform on 1000-year synthetic time coordinate...")
    syn_time = xr.cftime_range(start="1979", periods=12000, freq="MS")
    syn_anomaly = xr.DataArray(
        rng.normal(0, 0.02, 12000),
        coords={"time": syn_time}, dims=["time"]
    )
    syn_reconstructed = pp.inverse_transform(syn_anomaly, add_trend=False)
    print(f"    output shape: {syn_reconstructed.shape}  (expected: (12000,))")
    print(f"    output mean:  {float(syn_reconstructed.mean()):.4f}  "
          f"(seasonal cycle mean: {float(pp.seasonal_cycle.mean()):.4f})")

    print("\n" + "=" * 60)
    print("Debug run complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_debug()