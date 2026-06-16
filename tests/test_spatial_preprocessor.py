"""
test_spatial_preprocessor.py
=============================
Tests for SpatialPreprocessor.

All tests use synthetic data — no RACMO file needed. Fixtures construct
DataArrays that replicate the structure of SMBFieldLoader output:
  (time, rlat, rlon) with NaN outside a basin mask.

Run:
    pytest test_spatial_preprocessor.py -v

Real-data integration:
    python test_spatial_preprocessor.py \
        ./data/RACMO2.4p1_ANT11.nc \
        ./data/IceBoundaries_Antarctica_V2.shp \
        PineIsland
"""

from __future__ import annotations

import sys
import numpy as np
import pytest
import xarray as xr

from syn_smb import SpatialPreprocessor


# ======================================================================
# Fixtures — synthetic (time, rlat, rlon) fields
# ======================================================================

NY, NX, NT = 12, 15, 60   # 5 years monthly

# Basin mask: ~50% of cells are valid
RNG = np.random.default_rng(42)
_MASK = RNG.random((NY, NX)) > 0.4   # True inside basin


def _make_field(
    add_trend:   bool = True,
    add_seasonal: bool = True,
    with_nan:    bool = True,
    n_time:      int  = NT,
    seed:        int  = 0,
) -> xr.DataArray:
    """
    Synthetic SMB field (time, rlat, rlon).

    Optionally contains:
      - a linear trend per grid cell
      - a seasonal cycle (sin + cos harmonics) per grid cell
      - NaN outside the basin mask
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)

    # Stochastic noise (time, rlat, rlon)
    data = rng.normal(0.04, 0.02, size=(n_time, NY, NX))

    if add_trend:
        # Each cell gets its own slope
        slopes = rng.uniform(0.0001, 0.0005, size=(NY, NX))
        data  += slopes[np.newaxis, :, :] * t[:, np.newaxis, np.newaxis]

    if add_seasonal:
        months = t % 12   # 0-11 for each time step
        cos_cycle = 0.015 * np.cos(2 * np.pi * months / 12)
        sin_cycle = 0.010 * np.sin(2 * np.pi * months / 12)
        seasonal = (cos_cycle + sin_cycle)[:, np.newaxis, np.newaxis]
        # Add spatially-varying amplitude
        amp = rng.uniform(0.5, 1.5, size=(NY, NX))
        data += seasonal * amp[np.newaxis, :, :]

    time = xr.cftime_range("1979-01", periods=n_time, freq="MS")
    da = xr.DataArray(
        data.astype(np.float32),
        dims=["time", "rlat", "rlon"],
        coords={"time": time},
        attrs={"units": "m w.e."},
    )

    if with_nan:
        da = da.where(_MASK)   # NaN outside basin

    return da


@pytest.fixture(scope="module")
def field() -> xr.DataArray:
    return _make_field()


@pytest.fixture(scope="module")
def pp(field) -> SpatialPreprocessor:
    sp = SpatialPreprocessor()
    sp.fit(field)
    return sp


@pytest.fixture(scope="module")
def residuals(pp, field) -> xr.DataArray:
    return pp.transform(field)


# ======================================================================
# 1. Construction and validation
# ======================================================================

class TestConstruction:

    def test_repr_before_fit(self):
        sp = SpatialPreprocessor()
        assert "fitted=False" in repr(sp)

    def test_repr_after_fit(self, pp):
        r = repr(pp)
        assert "fitted=False" not in r
        assert str(NY) in r or str(NX) in r

    def test_not_fitted_raises_on_transform(self, field):
        sp = SpatialPreprocessor()
        with pytest.raises(RuntimeError, match="not fitted"):
            sp.transform(field)

    def test_not_fitted_raises_on_inverse(self, field):
        sp = SpatialPreprocessor()
        with pytest.raises(RuntimeError, match="not fitted"):
            sp.inverse_transform(field)

    def test_non_dataarray_raises(self):
        sp = SpatialPreprocessor()
        with pytest.raises(TypeError, match="xr.DataArray"):
            sp.fit(np.zeros((10, 5, 5)))   # type: ignore

    def test_missing_time_dim_raises(self):
        sp = SpatialPreprocessor()
        bad = xr.DataArray(np.ones((5, 5)), dims=["rlat", "rlon"])
        with pytest.raises(ValueError, match="time"):
            sp.fit(bad)

    def test_fit_returns_self(self, field):
        sp = SpatialPreprocessor()
        result = sp.fit(field)
        assert result is sp

    def test_is_fitted_after_fit(self, pp):
        assert pp.is_fitted is True


# ======================================================================
# 2. Stored attributes after fit()
# ======================================================================

class TestFitAttributes:

    def test_field_mean_shape(self, pp):
        """field_mean must be (rlat, rlon) — no time dimension."""
        assert "time"  not in pp.field_mean.dims
        assert pp.field_mean.shape == (NY, NX)

    def test_field_mean_values(self, pp, field):
        """field_mean should match field.mean(dim='time')."""
        expected = field.mean(dim="time")
        np.testing.assert_allclose(
            pp.field_mean.values,
            expected.values,
            rtol=1e-4,
            equal_nan=True,
        )

    def test_seasonal_means_shape(self, pp):
        """seasonal_means must have a 'month' dimension of size 12."""
        assert "month" in pp.seasonal_means.dims
        assert pp.seasonal_means.sizes["month"] == 12
        assert pp.seasonal_means.shape == (12, NY, NX)

    def test_seasonal_means_zero_mean(self, pp):
        """
        Monthly anomalies must sum to ~0 across the 12 months at each
        valid grid cell (by construction of the anomaly calculation).
        """
        monthly_sum = pp.seasonal_means.sum(dim="month")
        valid = _MASK
        inside = monthly_sum.values[valid]
        np.testing.assert_allclose(inside, 0.0, atol=1e-4)

    def test_trend_coeffs_shape(self, pp):
        """trend_coeffs: (degree+1, rlat, rlon)."""
        n_coeffs = pp.deg + 1
        assert pp.trend_coeffs.sizes["degree"] == n_coeffs
        spatial_sizes = {
            d: s for d, s in pp.trend_coeffs.sizes.items()
            if d != "degree"
        }
        assert set(spatial_sizes.values()) == {NY, NX}

    def test_spatial_dims(self, pp):
        assert pp.spatial_dims == ["rlat", "rlon"]

    def test_n_time(self, pp):
        assert pp.n_time == NT


# ======================================================================
# 3. transform() output
# ======================================================================

class TestTransform:

    def test_residuals_shape(self, residuals):
        assert residuals.shape == (NT, NY, NX)

    def test_residuals_have_time_dim(self, residuals):
        assert "time" in residuals.dims

    def test_nan_cells_preserved(self, residuals):
        """Cells outside the basin must still be NaN after transform."""
        outside = ~_MASK
        outside_vals = residuals.isel(time=0).values[outside]
        assert np.all(np.isnan(outside_vals)), (
            "Expected NaN outside basin in residuals."
        )

    def test_valid_cells_not_nan(self, residuals):
        """Cells inside the basin must be finite in the residuals."""
        inside = _MASK
        inside_vals = residuals.values[:, inside]
        assert np.all(np.isfinite(inside_vals)), (
            "Found NaN inside basin in residuals."
        )

    def test_residuals_near_zero_mean(self, residuals):
        """
        After removing trend + seasonal cycle, the time-mean of
        each valid grid cell's residuals should be near zero.
        """
        resid_mean = residuals.mean(dim="time")
        valid = _MASK
        inside_means = resid_mean.values[valid]
        # Tolerance is loose since we used a synthetic signal, not 45 years
        np.testing.assert_allclose(inside_means, 0.0, atol=0.005)

    def test_seasonal_cycle_removed(self, residuals):
        """
        After transform(), the monthly means of the residuals should be
        near zero — confirming the seasonal cycle was successfully removed.
        """
        monthly_means = (
            residuals.groupby("time.month")
            .mean("time")
        )
        valid = _MASK
        # For each of the 12 months, the spatial mean over valid cells
        # should be close to zero
        for m in range(12):
            month_vals = monthly_means.isel(month=m).values[valid]
            np.testing.assert_allclose(
                month_vals, 0.0, atol=0.005,
                err_msg=f"Seasonal cycle not removed for month {m+1}"
            )

    def test_trend_reduced(self, residuals):
        """
        Linear trend of residuals should be much smaller than that of
        the original field with trend.
        """
        field_with_trend = _make_field(add_trend=True, add_seasonal=False,
                                       with_nan=False)
        sp = SpatialPreprocessor()
        res = sp.fit_transform(field_with_trend)

        # Trend of residuals per cell (fitted on integer index)
        t = np.arange(NT, dtype=float)
        orig_vals = field_with_trend.values[:, _MASK]
        resid_vals = res.values[:, _MASK]

        from numpy.polynomial.polynomial import polyfit as npfit
        orig_slopes  = np.array([npfit(t, orig_vals[:, i], 1)[1]
                                  for i in range(orig_vals.shape[1])])
        resid_slopes = np.array([npfit(t, resid_vals[:, i], 1)[1]
                                  for i in range(resid_vals.shape[1])])

        assert np.abs(resid_slopes).mean() < np.abs(orig_slopes).mean() * 0.1, (
            "Trend not significantly reduced after detrending."
        )


# ======================================================================
# 4. fit_transform() convenience
# ======================================================================

class TestFitTransform:

    def test_fit_transform_equals_fit_then_transform(self, field):
        sp1 = SpatialPreprocessor()
        sp1.fit(field)
        r1 = sp1.transform(field)

        sp2 = SpatialPreprocessor()
        r2 = sp2.fit_transform(field)

        np.testing.assert_allclose(r1.values, r2.values, equal_nan=True)


# ======================================================================
# 5. inverse_transform() round-trip
# ======================================================================

class TestInverseTransform:

    def test_roundtrip_with_mean_restores_original_minus_trend(
        self, field, pp, residuals
    ):
        """
        inverse_transform(add_mean=True) should recover the original
        field minus the linear trend.
        The trend is intentionally NOT restored.
        """
        reconstructed = pp.inverse_transform(residuals, add_mean=True)

        # What we expect: original field minus trend
        t_idx = xr.DataArray(
            np.arange(NT, dtype=float),
            coords={"time": field.time}, dims=["time"]
        )
        trend = pp._eval_trend(t_idx)
        centered = field - pp.field_mean
        expected = centered - trend + pp.field_mean

        np.testing.assert_allclose(
            reconstructed.values,
            expected.values,
            rtol=1e-4,
            atol=1e-6,
            equal_nan=True,
        )

    def test_roundtrip_preserves_nan(self, pp, residuals):
        reconstructed = pp.inverse_transform(residuals, add_mean=True)
        outside = ~_MASK
        outside_vals = reconstructed.isel(time=0).values[outside]
        assert np.all(np.isnan(outside_vals))

    def test_inverse_without_mean_has_zero_time_mean(
        self, pp, residuals
    ):
        """
        Without add_mean, the long-run time-mean of the reconstruction
        should be ~0 (field_mean not yet added).
        """
        reconstructed = pp.inverse_transform(residuals, add_mean=False)
        recon_mean    = reconstructed.mean(dim="time")
        valid_means   = recon_mean.values[_MASK]
        np.testing.assert_allclose(valid_means, 0.0, atol=0.005)


# ======================================================================
# 6. Handles NaN-only cells (outside basin)
# ======================================================================

class TestNanHandling:

    def test_all_nan_cells_remain_nan_in_trend_coeffs(self, pp):
        """
        Grid cells that are always NaN should have NaN trend coefficients.
        """
        outside = ~_MASK
        # Check highest-degree coefficient (slope)
        slope = pp.trend_coeffs.sel(degree=1)
        outside_slopes = slope.values[outside]
        assert np.all(np.isnan(outside_slopes)), (
            "Expected NaN trend coefficients outside basin."
        )

    def test_all_nan_cells_remain_nan_in_seasonal_means(self, pp):
        outside = ~_MASK
        for m in range(12):
            month_sm = pp.seasonal_means.isel(month=m).values
            assert np.all(np.isnan(month_sm[outside])), (
                f"Expected NaN seasonal mean outside basin for month {m+1}."
            )


# ======================================================================
# 7. Works with different grid sizes and time lengths
# ======================================================================

class TestFlexibility:

    def test_short_time_series(self):
        """Should work with as few as 24 months (2 years)."""
        sp    = SpatialPreprocessor()
        field = _make_field(n_time=24, with_nan=False)
        res   = sp.fit_transform(field)
        assert res.shape == (24, NY, NX)

    def test_no_nan_field(self):
        """Works on a field with no NaN (no basin mask applied)."""
        sp    = SpatialPreprocessor()
        field = _make_field(with_nan=False)
        res   = sp.fit_transform(field)
        assert np.all(np.isfinite(res.values))

    def test_no_trend_no_seasonal(self):
        """Pure noise — residuals should equal zero-mean noise."""
        sp    = SpatialPreprocessor()
        field = _make_field(add_trend=False, add_seasonal=False, with_nan=False)
        res   = sp.fit_transform(field)
        # Residuals should be very close to the zero-mean noise
        # (any deviation is from the polynomial fit capturing noise)
        residual_std  = float(res.std())
        original_std  = float(field.std())
        assert abs(residual_std - original_std) / original_std < 0.15

    def test_transform_on_different_length_field(self, pp, field):
        """
        transform() should work on a field of different time length
        than the training data (e.g. for out-of-sample validation).
        """
        short_field = _make_field(n_time=24)
        res         = pp.transform(short_field)
        assert res.sizes["time"] == 24

    def test_higher_degree_polynomial(self, field):
        """deg=2 should work without errors."""
        sp  = SpatialPreprocessor(deg=2)
        res = sp.fit_transform(field)
        assert res.shape == (NT, NY, NX)
        assert sp.trend_coeffs.sizes["degree"] == 3


# ======================================================================
# Real-data integration
# ======================================================================

def run_real_data(
    racmo_path: str,
    shp_path:   str,
    basin_name: str,
) -> None:
    """
    Integration test on real RACMO + shapefile.
    Run as: python test_spatial_preprocessor.py <racmo.nc> <shp> <basin>
    """
    print(f"\n{'='*60}")
    print(f"SpatialPreprocessor — real data: {basin_name}")
    print(f"{'='*60}")

    from syn_smb import SMBFieldLoader

    loader = SMBFieldLoader(racmo_path, shp_path, basin_name)
    field  = loader.load()

    sp       = SpatialPreprocessor()
    residuals = sp.fit_transform(field)

    print(f"\nField shape:         {dict(field.sizes)}")
    print(f"Residuals shape:     {dict(residuals.sizes)}")

    # Field mean stats
    fm = sp.field_mean.values[loader.basin_mask.values]
    print(f"\nfield_mean — basin mean: {fm.mean():.4f} m w.e.")
    print(f"field_mean — spatial std: {fm.std():.4f} m w.e.")

    # Residual stats
    resid_vals = residuals.values[:, loader.basin_mask.values]
    print(f"\nResiduals — time-mean (should be ≈ 0): "
          f"{resid_vals.mean():.6f} m w.e.")
    print(f"Residuals — std: {resid_vals.std():.4f} m w.e.")

    # Seasonal means check
    monthly = (
        residuals.groupby("time.month").mean("time")
        .mean(dim=["rlat", "rlon"], skipna=True)
    )
    print(f"\nBasin-mean monthly residuals (should all be ≈ 0):")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    for m, (label, val) in enumerate(zip(month_labels, monthly.values)):
        print(f"  {label}: {val:+.5f}")

    print("\nGenerating diagnostic figure...")
    sp.plot_decomposition(
        field,
        save_path=f"spatial_preprocessor_{basin_name}.png",
    )

    # Sanity assertions
    assert abs(resid_vals.mean()) < 0.01,  "Residual mean too large."
    assert resid_vals.std()       > 0.001, "Residual variance near zero."

    print("\n✓ All sanity checks passed.")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_real_data(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(
            "Usage: python test_spatial_preprocessor.py "
            "<racmo.nc> <shapefile.shp> <basin_name>\n"
            "Running pytest instead..."
        )
        import subprocess
        subprocess.run(["python", "-m", "pytest", __file__, "-v"])