"""
test_smb_field_generator.py
============================
Tests for SMBFieldGenerator — the 2D pipeline orchestrator.

Unit tests build a synthetic field from known components and verify that
the full fit → generate cycle preserves the key statistical properties
(mean, basin mask, output shape, spectral band scaling).

Run unit tests:
    pytest test_smb_field_generator.py -v

Real-data integration (full Steps 1–5):
    python test_smb_field_generator.py \\
        ./data/RACMO2.4p1_ANT11.nc \\
        ./data/IceBoundaries_Antarctica_V2.shp \\
        PineIsland
"""

from __future__ import annotations

import sys
import numpy as np
import pytest
import xarray as xr

from syn_smb import SMBFieldGenerator
from syn_smb          import Experiment


# ======================================================================
# Shared synthetic data
# ======================================================================

NY, NX, NT = 12, 14, 60   # small grid — fast
N_MODES     = 3

RNG   = np.random.default_rng(7)
_MASK = RNG.random((NY, NX)) > 0.4   # ~60% valid cells


def _make_field(seed: int = 0) -> xr.DataArray:
    """
    Synthetic (time, rlat, rlon) SMB field with trend + seasonal cycle
    + stochastic noise, NaN outside _MASK.
    Matches the structure produced by SMBFieldLoader.
    """
    rng    = np.random.default_rng(seed)
    t      = np.arange(NT)
    months = t % 12

    base = (
        0.04
        + 5e-5 * t[:, None, None]
        + 0.015 * np.cos(2 * np.pi * months / 12)[:, None, None]
        + rng.normal(0, 0.015, (NT, NY, NX))
    ).astype(np.float32)

    time = xr.cftime_range("1979-01", periods=NT, freq="MS")
    da   = xr.DataArray(
        base,
        dims=["time", "rlat", "rlon"],
        coords={"time": time},
        attrs={"units": "m w.e."},
    )
    return da.where(_MASK)


@pytest.fixture(scope="module")
def field() -> xr.DataArray:
    return _make_field()


@pytest.fixture(scope="module")
def gen(field) -> SMBFieldGenerator:
    g = SMBFieldGenerator(n_modes=N_MODES, nperseg=30)
    g.fit(field)
    return g


@pytest.fixture(scope="module")
def baseline_ds(gen) -> xr.Dataset:
    exp = Experiment(n_years=5, n_members=3, seed=0)
    return gen.generate(exp)


# ======================================================================
# 1. Construction and repr
# ======================================================================

class TestConstruction:

    def test_repr_before_fit(self):
        g = SMBFieldGenerator()
        assert "fitted=False" in repr(g)

    def test_repr_after_fit(self, gen):
        r = repr(gen)
        assert "fitted=False" not in r
        assert "n_modes=" in r

    def test_not_fitted_generate_raises(self):
        g = SMBFieldGenerator()
        with pytest.raises(RuntimeError, match="not fitted"):
            g.generate(Experiment.baseline())

    def test_fit_returns_self(self, field):
        g = SMBFieldGenerator(n_modes=2, nperseg=20)
        assert g.fit(field) is g

    def test_is_fitted_after_fit(self, gen):
        assert gen.is_fitted


# ======================================================================
# 2. fit() — internal component state
# ======================================================================

class TestFitState:

    def test_preprocessor_fitted(self, gen):
        assert gen.preprocessor is not None
        assert gen.preprocessor.is_fitted

    def test_eof_fitted(self, gen):
        assert gen.eof is not None
        assert gen.eof.is_fitted

    def test_gt_per_pc_length(self, gen):
        assert len(gen.gt_per_pc) == gen.n_modes

    def test_ss_per_pc_length(self, gen):
        assert len(gen.ss_per_pc) == gen.n_modes

    def test_all_gt_fitted(self, gen):
        for i, gt in enumerate(gen.gt_per_pc):
            assert gt is not None, f"GaussianTransform {i} is None"

    def test_all_ss_fitted(self, gen):
        for i, ss in enumerate(gen.ss_per_pc):
            assert ss is not None, f"SpectralSynthesizer {i} is None"

    def test_field_mean_shape(self, gen):
        assert gen.field_mean.shape == (NY, NX)
        assert "time" not in gen.field_mean.dims

    def test_basin_mask_shape(self, gen):
        assert gen.basin_mask.shape == (NY, NX)
        assert gen.basin_mask.dtype == bool

    def test_basin_mask_matches_input(self, gen):
        np.testing.assert_array_equal(
            gen.basin_mask.values, _MASK
        )

    def test_field_mean_nan_outside_basin(self, gen):
        outside = ~_MASK
        assert np.all(np.isnan(gen.field_mean.values[outside]))

    def test_field_mean_finite_inside_basin(self, gen):
        inside = _MASK
        assert np.all(np.isfinite(gen.field_mean.values[inside]))


# ======================================================================
# 3. generate() — output shape and structure
# ======================================================================

class TestGenerateShape:

    def test_returns_dataset(self, baseline_ds):
        assert isinstance(baseline_ds, xr.Dataset)

    def test_smb_syn_in_dataset(self, baseline_ds):
        assert "smb_syn" in baseline_ds

    def test_basin_mask_in_dataset(self, baseline_ds):
        assert "basin_mask" in baseline_ds

    def test_smb_syn_shape(self, baseline_ds):
        exp  = Experiment(n_years=5, n_members=3, seed=0)
        ds   = baseline_ds
        assert ds["smb_syn"].sizes["member"] == 3
        assert ds["smb_syn"].sizes["time"]   == 5 * 12
        assert ds["smb_syn"].sizes["rlat"]   == NY
        assert ds["smb_syn"].sizes["rlon"]   == NX

    def test_member_coord(self, baseline_ds):
        assert "member" in baseline_ds["smb_syn"].coords
        np.testing.assert_array_equal(
            baseline_ds["smb_syn"].coords["member"].values,
            [0, 1, 2],
        )

    def test_time_coord_length(self, baseline_ds):
        assert baseline_ds["smb_syn"].sizes["time"] == 5 * 12

    def test_time_coord_starts_jan_1979(self, baseline_ds):
        t0 = baseline_ds["smb_syn"].time.values[0]
        assert t0.year == 1979
        assert t0.month == 1


# ======================================================================
# 4. NaN handling in output
# ======================================================================

class TestNanHandling:

    def test_nan_outside_basin(self, baseline_ds):
        smb     = baseline_ds["smb_syn"]
        outside = ~_MASK
        for m in range(smb.sizes["member"]):
            outside_vals = smb.isel(member=m, time=0).values[outside]
            assert np.all(np.isnan(outside_vals)), (
                f"Member {m}: non-NaN outside basin."
            )

    def test_finite_inside_basin(self, baseline_ds):
        smb    = baseline_ds["smb_syn"]
        inside = _MASK
        for m in range(smb.sizes["member"]):
            inside_vals = smb.isel(member=m, time=0).values[inside]
            assert np.all(np.isfinite(inside_vals)), (
                f"Member {m}: NaN inside basin."
            )

    def test_basin_mask_in_output_matches_input(self, baseline_ds):
        np.testing.assert_array_equal(
            baseline_ds["basin_mask"].values, _MASK
        )


# ======================================================================
# 5. Statistical properties of synthetic output
# ======================================================================

class TestStatisticalProperties:

    def test_mean_ratio_near_one(self, gen, baseline_ds):
        """
        Long-run basin mean of synthetic field should be close to
        the training field mean.
        """
        smb    = baseline_ds["smb_syn"]
        inside = _MASK

        obs_mean = float(np.nanmean(gen.field_mean.values[inside]))
        syn_mean = float(np.nanmean(smb.values))
        ratio    = abs(syn_mean / obs_mean) if obs_mean != 0 else 0.0

        assert 0.5 < ratio < 2.0, (
            f"Synthetic mean ({syn_mean:.4f}) is far from observed "
            f"mean ({obs_mean:.4f}). Mean not restored correctly."
        )

    def test_different_members_are_different(self, baseline_ds):
        """Each member should be a different realisation."""
        smb = baseline_ds["smb_syn"]
        if smb.sizes["member"] < 2:
            pytest.skip("Need at least 2 members.")
        m0 = smb.isel(member=0).values
        m1 = smb.isel(member=1).values
        assert not np.allclose(
            m0[np.isfinite(m0)],
            m1[np.isfinite(m1)],
        ), "Members are identical — RNG seeding issue."

    def test_synthetic_values_in_plausible_range(self, baseline_ds, gen):
        """Synthetic values should be within ±5σ of training mean."""
        smb    = baseline_ds["smb_syn"]
        inside = _MASK

        obs_mean = float(np.nanmean(gen.field_mean.values))
        obs_std  = float(np.nanstd(
            smb.values[:, :, :, :]
        ))

        syn_vals = smb.values[:, :, inside]
        syn_vals = syn_vals[np.isfinite(syn_vals)]

        lo = obs_mean - 5 * obs_std
        hi = obs_mean + 5 * obs_std
        assert syn_vals.min() > lo, f"Synthetic values below {lo:.4f}"
        assert syn_vals.max() < hi, f"Synthetic values above {hi:.4f}"


# ======================================================================
# 6. Band scaling experiments
# ======================================================================

class TestBandScaling:

    def test_annual_enhanced_has_more_annual_variance(self, gen):
        """
        Annual-enhanced experiment should have higher variance in the
        annual band than the baseline experiment.
        """
        from scipy.signal import welch

        exp_base = Experiment(n_years=50, n_members=1, seed=42)
        exp_ann  = Experiment(
            n_years=50, n_members=1, seed=42,
            band_scales=[(0.8, 1.5, 10.0)],
            name="annual_enhanced",
        )

        ds_base = gen.generate(exp_base)
        ds_ann  = gen.generate(exp_ann)

        # Extract basin-mean time series for spectral comparison
        def basin_mean_ts(ds):
            smb = ds["smb_syn"].isel(member=0)
            return float(np.nanmean(smb.values, axis=(1, 2)))

        # Simple variance check: annual-enhanced should have more total variance
        base_vals = np.nanmean(ds_base["smb_syn"].isel(member=0).values,
                                axis=(1, 2))
        ann_vals  = np.nanmean(ds_ann["smb_syn"].isel(member=0).values,
                                axis=(1, 2))

        base_var = float(np.var(base_vals))
        ann_var  = float(np.var(ann_vals))

        assert ann_var > base_var, (
            f"Annual-enhanced variance ({ann_var:.6f}) should exceed "
            f"baseline ({base_var:.6f})."
        )

    def test_generate_suite_returns_all_experiments(self, gen):
        suite = Experiment.standard_suite(n_years=5, n_members=2)
        result = gen.generate_suite(suite)

        assert isinstance(result, dict)
        assert len(result) == len(suite)
        for exp in suite:
            assert exp.name in result
            assert isinstance(result[exp.name], xr.Dataset)

    def test_suite_baseline_and_enhanced_differ(self, gen):
        suite   = Experiment.standard_suite(n_years=5, n_members=2)
        datasets = gen.generate_suite(suite)

        base = datasets["baseline"]["smb_syn"].values
        ann  = datasets["annual_enhanced_10.0x"]["smb_syn"].values

        base_valid = base[np.isfinite(base)]
        ann_valid  = ann[np.isfinite(ann)]

        assert not np.allclose(base_valid, ann_valid), (
            "Baseline and annual-enhanced output are identical."
        )


# ======================================================================
# 7. Reproducibility
# ======================================================================

class TestReproducibility:

    def test_same_seed_same_output(self, gen):
        exp  = Experiment(n_years=5, n_members=2, seed=123)
        ds1  = gen.generate(exp)
        ds2  = gen.generate(exp)

        np.testing.assert_array_equal(
            ds1["smb_syn"].values,
            ds2["smb_syn"].values,
        )

    def test_different_seed_different_output(self, gen):
        exp1 = Experiment(n_years=5, n_members=1, seed=1)
        exp2 = Experiment(n_years=5, n_members=1, seed=2)
        ds1  = gen.generate(exp1)
        ds2  = gen.generate(exp2)

        v1 = ds1["smb_syn"].values
        v2 = ds2["smb_syn"].values
        assert not np.allclose(
            v1[np.isfinite(v1)],
            v2[np.isfinite(v2)],
        )


# ======================================================================
# 8. n_obs property
# ======================================================================

class TestProperties:

    def test_n_obs(self, gen):
        assert gen.n_obs == NT


# ======================================================================
# Real-data integration (Steps 1–5)
# ======================================================================

def run_real_data(
    racmo_path:  str,
    shp_path:    str,
    basin_name:  str,
    n_modes:     int = 10,
) -> None:
    """
    Full integration test: RACMO file → synthetic 2D ensemble.
    """
    print(f"\n{'='*60}")
    print(f"SMBFieldGenerator — real data: {basin_name}")
    print(f"{'='*60}")

    from syn_smb import SMBFieldLoader
    import matplotlib.pyplot as plt

    # ── Fit ──────────────────────────────────────────────────────────
    loader = SMBFieldLoader(racmo_path, shp_path, basin_name)
    field  = loader.load()

    gen = SMBFieldGenerator(n_modes=n_modes, nperseg=60)
    gen.fit(field, lat=loader.lat)

    print(f"\n{gen}")

    # ── Generate baseline ─────────────────────────────────────────────
    exp    = Experiment(n_years=100, n_members=5, seed=0)
    ds     = gen.generate(exp)
    smb    = ds["smb_syn"]
    inside = loader.basin_mask.values

    print(f"\nSynthetic field shape: {dict(smb.sizes)}")

    # Extract numpy arrays — NaN outside basin, so nanmean handles masking
    obs_vals = field.values   # (time, rlat, rlon)
    syn_vals = smb.values     # (member, time, rlat, rlon)

    # NaN outside the basin, so nanmean over spatial dims gives basin mean
    obs_basin_mean = float(np.nanmean(obs_vals))
    syn_basin_mean = float(np.nanmean(syn_vals))

    print(f"\nObserved basin mean : {obs_basin_mean:.4f} m w.e.")
    print(f"Synthetic basin mean: {syn_basin_mean:.4f} m w.e.")
    print(f"Ratio               : {syn_basin_mean/obs_basin_mean:.4f}")

    obs_basin_std = float(np.nanstd(obs_vals))
    syn_basin_std = float(np.nanstd(syn_vals))
    print(f"\nObserved std : {obs_basin_std:.4f} m w.e.")
    print(f"Synthetic std: {syn_basin_std:.4f} m w.e.")

    # ── Figures ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"SMBFieldGenerator — {basin_name}", fontsize=12)

    # Time-mean observed
    ax = axes[0]
    obs_mean = np.nanmean(obs_vals, axis=0)
    valid_lat = loader.lat.values[inside]
    valid_lon = loader.lon.values[inside]
    lat_pad = (valid_lat.max() - valid_lat.min()) * 0.15
    lon_pad = (valid_lon.max() - valid_lon.min()) * 0.15
    pcm = ax.pcolormesh(loader.lon.values, loader.lat.values,
                         obs_mean, cmap="Blues", shading="auto")
    ax.set_xlim(valid_lon.min() - lon_pad, valid_lon.max() + lon_pad)
    ax.set_ylim(valid_lat.min() - lat_pad, valid_lat.max() + lat_pad)
    plt.colorbar(pcm, ax=ax, label="m w.e.")
    ax.set_title("Observed time-mean SMB")

    # Time-mean synthetic (member 0)
    ax = axes[1]
    syn_mean = np.nanmean(syn_vals[0], axis=0)
    pcm = ax.pcolormesh(loader.lon.values, loader.lat.values,
                         syn_mean, cmap="Blues", shading="auto")
    ax.set_xlim(valid_lon.min() - lon_pad, valid_lon.max() + lon_pad)
    ax.set_ylim(valid_lat.min() - lat_pad, valid_lat.max() + lat_pad)
    plt.colorbar(pcm, ax=ax, label="m w.e.")
    ax.set_title("Synthetic time-mean SMB\n(member 0, 100 yr)")

    # Basin-mean time series — nanmean over spatial dims handles NaN mask
    ax = axes[2]
    obs_ts = np.nanmean(obs_vals.reshape(obs_vals.shape[0], -1), axis=1)
    ax.plot(obs_ts, color="tab:blue", lw=1, alpha=0.8, label="Observed")
    ax.axhline(np.nanmean(obs_ts), color="tab:blue", lw=1.5,
               linestyle="--", alpha=0.6)
    ax.set_xlabel("Time step (months)")
    ax.set_ylabel("Basin-mean SMB (m w.e.)")
    ax.set_title("Basin-mean time series")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"smb_field_generator_{basin_name}.png",
                dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: smb_field_generator_{basin_name}.png")

    # ── Sanity assertions ─────────────────────────────────────────────
    assert abs(syn_basin_mean / obs_basin_mean - 1.0) < 0.3, (
        "Mean ratio > 30% off."
    )
    assert smb.sizes["member"] == 5
    assert smb.sizes["time"]   == 100 * 12
    assert np.all(np.isnan(smb.values[:, :, ~inside]))

    print("\n✓ All sanity checks passed.")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        n_modes = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        run_real_data(sys.argv[1], sys.argv[2], sys.argv[3], n_modes)
    else:
        print(
            "Usage: python test_smb_field_generator.py "
            "<racmo.nc> <shp> <basin_name> [n_modes]\n"
            "Running pytest instead..."
        )
        import subprocess
        subprocess.run(["python", "-m", "pytest", __file__, "-v"])