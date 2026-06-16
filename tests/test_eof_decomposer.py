"""
test_eof_decomposer.py
======================
Tests for EOFDecomposer.

Unit tests use synthetic residual fields constructed to have a known
EOF structure, so we can test mathematical properties exactly
rather than just checking shapes and types.

Run unit tests:
    pytest test_eof_decomposer.py -v

Real-data integration (runs full Steps 1-3):
    python test_eof_decomposer.py \
        ./data/RACMO2.4p1_ANT11.nc \
        ./data/IceBoundaries_Antarctica_V2.shp \
        PineIsland
"""

from __future__ import annotations

import sys
import numpy as np
import pytest
import xarray as xr

from syn_smb import EOFDecomposer


# ======================================================================
# Fixtures — synthetic residual fields with known structure
# ======================================================================

NY, NX, NT = 14, 16, 60
N_MODES     = 4

# Basin mask — ~60% valid cells
RNG    = np.random.default_rng(99)
_MASK  = RNG.random((NY, NX)) > 0.35


def _make_residuals(
    n_modes_true: int   = 4,
    noise_frac:   float = 0.05,
    with_nan:     bool  = True,
    seed:         int   = 0,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """
    Construct a synthetic residual field from known EOFs + PCs.

    The signal is normalised to unit RMS before noise is added, so
    noise_frac is the noise standard deviation *relative to the signal
    RMS*. With noise_frac=0.05 the noise is 5% of the signal amplitude,
    making the reconstruction accuracy predictable.

    Returns
    -------
    residuals : xr.DataArray (time, rlat, rlon) with NaN outside mask
    true_eofs : np.ndarray (n_modes_true, n_valid)  — ground-truth patterns
    true_pcs  : np.ndarray (NT, n_modes_true)        — ground-truth time series
    """
    rng     = np.random.default_rng(seed)
    n_valid = int(_MASK.sum())

    # Random orthogonal EOF patterns (n_modes_true, n_valid)
    raw_eofs  = rng.normal(0, 1, (n_modes_true, n_valid))
    Q, _      = np.linalg.qr(raw_eofs.T)     # orthogonalise columns
    true_eofs = Q.T[:n_modes_true]            # (n_modes_true, n_valid), orthonormal

    # PC time series with decreasing variance
    variances = np.array([1.0 / (i + 1) for i in range(n_modes_true)])
    true_pcs  = rng.normal(0, 1, (NT, n_modes_true)) * np.sqrt(variances)

    # Build signal and normalise to unit RMS so noise_frac is relative
    signal    = true_pcs @ true_eofs          # (NT, n_valid)
    signal_rms = np.sqrt(np.mean(signal ** 2))
    signal    = signal / signal_rms           # unit RMS

    # Add noise at noise_frac × signal RMS = noise_frac (since RMS = 1)
    noise   = rng.normal(0, noise_frac, signal.shape)
    X_valid = signal + noise

    # Map into full grid with zeros for non-basin cells
    flat = np.zeros((NT, NY * NX))
    flat[:, _MASK.flatten()] = X_valid
    data = flat.reshape(NT, NY, NX).astype(np.float32)

    time = xr.cftime_range("1979-01", periods=NT, freq="MS")
    da   = xr.DataArray(
        data, dims=["time", "rlat", "rlon"],
        coords={"time": time},
    )
    if with_nan:
        da = da.where(_MASK)

    return da, true_eofs, true_pcs


@pytest.fixture(scope="module")
def residuals() -> xr.DataArray:
    da, _, _ = _make_residuals()
    return da


@pytest.fixture(scope="module")
def eof(residuals) -> EOFDecomposer:
    e = EOFDecomposer(n_modes=N_MODES)
    e.fit(residuals)
    return e


# ======================================================================
# 1. Construction and error handling
# ======================================================================

class TestConstruction:

    def test_repr_before_fit(self):
        e = EOFDecomposer(n_modes=5)
        assert "fitted=False" in repr(e)

    def test_repr_after_fit(self, eof):
        r = repr(eof)
        assert "fitted=False" not in r
        assert "n_modes=" in r

    def test_not_fitted_transform_raises(self, residuals):
        e = EOFDecomposer()
        with pytest.raises(RuntimeError, match="not fitted"):
            e.transform(residuals)

    def test_not_fitted_inverse_raises(self):
        e = EOFDecomposer()
        with pytest.raises(RuntimeError, match="not fitted"):
            e.inverse_transform(np.zeros((10, 4)))

    def test_non_dataarray_raises(self):
        e = EOFDecomposer()
        with pytest.raises(TypeError, match="DataArray"):
            e.fit(np.zeros((10, 5, 5)))   # type: ignore

    def test_missing_time_dim_raises(self):
        e   = EOFDecomposer()
        bad = xr.DataArray(np.ones((5, 5)), dims=["rlat", "rlon"])
        with pytest.raises(ValueError, match="time"):
            e.fit(bad)

    def test_2d_input_raises(self):
        e   = EOFDecomposer()
        bad = xr.DataArray(
            np.ones((10, 5)),
            dims=["time", "rlat"],
            coords={"time": np.arange(10)},
        )
        with pytest.raises(ValueError, match="3D"):
            e.fit(bad)

    def test_fit_returns_self(self, residuals):
        e = EOFDecomposer(n_modes=2)
        assert e.fit(residuals) is e


# ======================================================================
# 2. Shape of outputs
# ======================================================================

class TestShapes:

    def test_pcs_shape(self, eof):
        assert eof.pcs.shape == (NT, N_MODES)

    def test_singular_values_shape(self, eof):
        assert eof.singular_values.shape == (N_MODES,)

    def test_explained_variance_ratio_shape(self, eof):
        assert eof.explained_variance_ratio.shape == (N_MODES,)

    def test_eofs_as_field_shape(self, eof):
        eofs = eof.eofs_as_field()
        assert eofs.shape == (N_MODES, NY, NX)
        assert "mode" in eofs.dims

    def test_n_valid_cells(self, eof):
        assert eof.n_valid_cells == int(_MASK.sum())

    def test_inverse_transform_shape(self, eof):
        pcs_syn = np.random.randn(100, N_MODES)
        field   = eof.inverse_transform(pcs_syn)
        assert field.shape == (100, NY, NX)
        assert "time" in field.dims

    def test_transform_shape(self, eof, residuals):
        pcs = eof.transform(residuals)
        assert pcs.shape == (NT, N_MODES)


# ======================================================================
# 3. Mathematical properties
# ======================================================================

class TestMathProperties:

    def test_explained_variance_sums_to_leq_one(self, eof):
        """Each mode explains a positive fraction; total ≤ 1."""
        ev = eof.explained_variance_ratio
        assert np.all(ev > 0), "Some modes explain negative variance."
        assert ev.sum() <= 1.0 + 1e-6

    def test_explained_variance_decreasing(self, eof):
        """Modes must be ordered from highest to lowest variance."""
        ev = eof.explained_variance_ratio
        assert np.all(np.diff(ev) <= 1e-10), (
            "Explained variance not monotonically decreasing."
        )

    def test_pcs_approximately_uncorrelated(self, eof):
        """
        PCs should be approximately orthogonal.
        The PC correlation matrix should be close to diagonal.
        """
        pcs = eof.pcs
        corr = np.corrcoef(pcs.T)          # (n_modes, n_modes)
        off_diag = corr - np.eye(N_MODES)
        assert np.abs(off_diag).max() < 0.05, (
            f"PCs are correlated: max off-diagonal |corr| = "
            f"{np.abs(off_diag).max():.4f}"
        )

    def test_singular_values_positive_decreasing(self, eof):
        sv = eof.singular_values
        assert np.all(sv > 0)
        assert np.all(np.diff(sv) <= 0)

    def test_roundtrip_reconstruction_accuracy(self, eof, residuals):
        """
        transform() then inverse_transform() should recover the signal
        accurately for a synthetic field built from exactly N_MODES modes.

        With noise_frac=0.05 (5% relative noise) and N_MODES=4 modes
        covering a rank-4 signal, the relative RMSE after reconstruction
        should be well below 10%.

        If this fails, the most likely causes are:
          - a sign flip in the EOFs between fit() and transform()
          - incorrect weighting/un-weighting in inverse_transform()
          - noise_frac too large relative to signal (check _make_residuals)
        """
        pcs   = eof.transform(residuals)
        recon = eof.inverse_transform(pcs, time_coord=residuals.time)

        valid_orig  = residuals.values[:, _MASK]
        valid_recon = recon.values[:, _MASK]

        rmse     = np.sqrt(np.mean((valid_orig - valid_recon) ** 2))
        scale    = np.sqrt(np.mean(valid_orig ** 2))
        rel_rmse = rmse / scale

        assert rel_rmse < 0.10, (
            f"Round-trip relative RMSE = {rel_rmse:.4f} (expected < 0.10). "
            f"Signal RMS = {scale:.4f}, RMSE = {rmse:.4f}. "
            f"With 5% relative noise and {N_MODES} modes on a rank-{N_MODES} "
            f"signal this should be <10%. "
            f"Check that noise_frac is relative (signal normalised to unit RMS)."
        )

    def test_roundtrip_better_with_more_modes(self, residuals):
        """More modes → lower reconstruction error."""
        errors = []
        for n in [1, 2, 4]:
            e = EOFDecomposer(n_modes=n)
            e.fit(residuals)
            pcs   = e.transform(residuals)
            recon = e.inverse_transform(pcs)

            v_o = residuals.values[:, _MASK]
            v_r = recon.values[:, _MASK]
            rmse = np.sqrt(np.mean((v_o - v_r) ** 2))
            errors.append(rmse)

        assert errors[0] > errors[1] > errors[2], (
            "Adding more modes should reduce reconstruction error."
        )

    def test_eofs_sign_consistent(self, eof):
        """
        First non-zero loading of each EOF should be positive
        (sign standardisation applied during fit).
        """
        eofs = eof.eofs_as_field()
        for i in range(N_MODES):
            vals = eofs.isel(mode=i).values.flatten()
            first_val = next((v for v in vals if not np.isnan(v) and v != 0),
                             None)
            if first_val is not None:
                assert first_val > 0, (
                    f"EOF {i+1}: first non-zero loading is negative "
                    f"({first_val:.4f}). Sign standardisation failed."
                )


# ======================================================================
# 4. NaN handling
# ======================================================================

class TestNanHandling:

    def test_nan_outside_basin_in_eofs(self, eof):
        """Cells outside the basin mask must be NaN in eofs_as_field()."""
        eofs    = eof.eofs_as_field()
        outside = ~_MASK
        for i in range(N_MODES):
            outside_vals = eofs.isel(mode=i).values[outside]
            assert np.all(np.isnan(outside_vals)), (
                f"EOF {i+1}: non-NaN values outside basin mask."
            )

    def test_nan_outside_basin_in_reconstruction(self, eof):
        """Reconstructed field must be NaN outside the basin."""
        pcs_syn = np.random.randn(20, N_MODES)
        recon   = eof.inverse_transform(pcs_syn)
        outside = ~_MASK
        recon_outside = recon.values[:, outside]
        assert np.all(np.isnan(recon_outside))

    def test_valid_cells_finite_in_reconstruction(self, eof):
        pcs_syn = np.random.randn(20, N_MODES)
        recon   = eof.inverse_transform(pcs_syn)
        inside  = _MASK
        recon_inside = recon.values[:, inside]
        assert np.all(np.isfinite(recon_inside))


# ======================================================================
# 5. suggest_n_modes and variance diagnostics
# ======================================================================

class TestVarianceDiagnostics:

    def test_suggest_n_modes_returns_int(self, eof):
        n = eof.suggest_n_modes(threshold=0.80)
        assert isinstance(n, int)
        assert n >= 1

    def test_suggest_n_modes_respects_threshold(self, eof):
        """Cumulative variance at returned n must exceed the threshold."""
        for threshold in [0.50, 0.80, 0.95]:
            n = eof.suggest_n_modes(threshold=threshold)
            cumvar = float(np.cumsum(eof._expvar)[n - 1])
            assert cumvar >= threshold * 0.99, (  # 1% tolerance
                f"suggest_n_modes({threshold}) returned n={n} but "
                f"cumvar={cumvar:.3f} < threshold."
            )

    def test_suggest_n_modes_increases_with_threshold(self, eof):
        n80 = eof.suggest_n_modes(0.80)
        n95 = eof.suggest_n_modes(0.95)
        assert n95 >= n80

    def test_n_modes_capped_at_rank(self, residuals):
        """Requesting more modes than the rank should warn and cap."""
        n_valid = int(_MASK.sum())
        huge    = min(NT, n_valid) + 100
        e       = EOFDecomposer(n_modes=huge)
        with pytest.warns(UserWarning, match="n_modes"):
            e.fit(residuals)
        assert e.n_modes <= min(NT - 1, n_valid - 1)


# ======================================================================
# 6. Area weighting
# ======================================================================

class TestAreaWeighting:

    def test_with_lat_runs(self, residuals):
        """Providing a lat DataArray for area weighting should not error."""
        lat = xr.DataArray(
            np.linspace(-80, -70, NY)[:, np.newaxis] * np.ones((NY, NX)),
            dims=["rlat", "rlon"],
        )
        e = EOFDecomposer(n_modes=3)
        e.fit(residuals, lat=lat)
        assert e.is_fitted
        assert e.pcs.shape == (NT, 3)

    def test_weighted_and_unweighted_give_different_eofs(self, residuals):
        """Area weighting should produce different EOF patterns."""
        lat = xr.DataArray(
            np.linspace(-80, -70, NY)[:, np.newaxis] * np.ones((NY, NX)),
            dims=["rlat", "rlon"],
        )
        e_unw = EOFDecomposer(n_modes=3)
        e_unw.fit(residuals)

        e_w = EOFDecomposer(n_modes=3)
        e_w.fit(residuals, lat=lat)

        # EOFs should differ (weighting changes the decomposition)
        eofs_unw = e_unw.eofs_as_field().values
        eofs_w   = e_w.eofs_as_field().values

        # At least one mode should differ by more than numerical noise
        max_diff = np.nanmax(np.abs(eofs_unw - eofs_w))
        assert max_diff > 1e-6, (
            "Weighted and unweighted EOFs are identical — "
            "area weighting has no effect."
        )


# ======================================================================
# 7. transform() on held-out data
# ======================================================================

class TestOutOfSampleProjection:

    def test_transform_held_out_shape(self, eof):
        """transform() on a held-out field returns correct shape."""
        held_out, _, _ = _make_residuals(seed=99, with_nan=True)
        pcs_new = eof.transform(held_out)
        assert pcs_new.shape == (NT, N_MODES)

    def test_transform_held_out_different_from_training(self, eof, residuals):
        """
        Projection of a different field should give different PCs.
        """
        other, _, _ = _make_residuals(seed=999, with_nan=True)
        pcs_train = eof.transform(residuals)
        pcs_other = eof.transform(other)
        assert not np.allclose(pcs_train, pcs_other), (
            "Training and held-out PCs are identical — transform() "
            "is not using the field values."
        )


# ======================================================================
# Real-data integration (Steps 1 + 2 + 3)
# ======================================================================

def run_real_data(
    racmo_path: str,
    shp_path:   str,
    basin_name: str,
    n_modes:    int = 10,
) -> None:
    """
    Full integration test: SMBFieldLoader → SpatialPreprocessor → EOFDecomposer.
    """
    print(f"\n{'='*60}")
    print(f"EOFDecomposer — real data: {basin_name}")
    print(f"{'='*60}")

    from syn_smb     import SMBFieldLoader
    from syn_smb import SpatialPreprocessor

    # ── Step 1: Load field ──
    loader    = SMBFieldLoader(racmo_path, shp_path, basin_name)
    field     = loader.load()

    # ── Step 2: Preprocess ──
    sp        = SpatialPreprocessor()
    residuals = sp.fit_transform(field)

    # ── Step 3: EOF decomposition ──
    eof = EOFDecomposer(n_modes=n_modes)
    eof.fit(residuals, lat=loader.lat)

    eof.summary()

    # Suggest optimal n_modes
    n95 = eof.suggest_n_modes(0.95)
    n80 = eof.suggest_n_modes(0.80)
    print(f"\n  Recommended n_modes (80% var): {n80}")
    print(f"  Recommended n_modes (95% var): {n95}")

    # Reconstruction round-trip
    pcs_train = eof.transform(residuals)
    recon     = eof.inverse_transform(pcs_train, time_coord=residuals.time)

    valid      = loader.basin_mask.values
    orig_vals  = residuals.values[:, valid]
    recon_vals = recon.values[:, valid]
    rmse       = np.sqrt(np.mean((orig_vals - recon_vals) ** 2))
    scale      = np.sqrt(np.mean(orig_vals ** 2))
    print(f"\n  Round-trip RMSE: {rmse:.5f} m w.e.")
    print(f"  Signal RMS:      {scale:.5f} m w.e.")
    print(f"  Relative RMSE:   {rmse/scale:.2%}")

    # Figures
    print("\nGenerating figures...")
    eof.plot_variance(save_path=f"eof_variance_{basin_name}.png")
    eof.plot_eofs(n=min(6, n_modes),
                  save_path=f"eof_patterns_{basin_name}.png")
    eof.plot_pcs(n=min(4, n_modes),
                 save_path=f"eof_pcs_{basin_name}.png")

    # Sanity checks
    assert eof.n_valid_cells > 0
    assert pcs_train.shape[1] == n_modes
    assert rmse / scale < 0.5,  "Reconstruction error too large."
    print("\n✓ All sanity checks passed.")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        n_modes = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        run_real_data(sys.argv[1], sys.argv[2], sys.argv[3], n_modes)
    else:
        print(
            "Usage: python test_eof_decomposer.py "
            "<racmo.nc> <shp> <basin_name> [n_modes]\n"
            "Running pytest instead..."
        )
        import subprocess
        subprocess.run(["python", "-m", "pytest", __file__, "-v"])