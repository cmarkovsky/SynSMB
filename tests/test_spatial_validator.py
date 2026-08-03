"""
test_spatial_validator.py
=========================
Tests for SpatialValidator.

Unit tests use synthetic data — no RACMO file or shapefile needed.
The fixture constructs a minimal SMBFieldGenerator from synthetic data
so we can validate against known properties.

Run unit tests:
    pytest test_spatial_validator.py -v

Real-data integration (full Steps 1–5 + 7):
    python test_spatial_validator.py \\
        ./data/RACMO2.4p1_ANT11.nc \\
        ./data/IceBoundaries_Antarctica_V2.shp \\
        PineIsland
"""

from __future__ import annotations

import sys
import numpy as np
import pytest
import xarray as xr
import matplotlib
matplotlib.use("Agg")

from syn_smb import SMBFieldGenerator
from syn_smb    import SpatialValidator
from syn_smb           import Experiment


# ======================================================================
# Shared synthetic data (same structure as test_smb_field_generator)
# ======================================================================

NY, NX, NT = 12, 14, 60
N_MODES     = 3
RNG         = np.random.default_rng(7)
_MASK       = RNG.random((NY, NX)) > 0.4


def _make_field(seed: int = 0) -> xr.DataArray:
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
        base, dims=["time", "rlat", "rlon"],
        coords={"time": time}, attrs={"units": "m w.e."},
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
    return gen.generate(Experiment(n_years=5, n_members=3, seed=0))


@pytest.fixture(scope="module")
def val(gen, field) -> SpatialValidator:
    return SpatialValidator(gen, field)


# ======================================================================
# 1. Construction
# ======================================================================

class TestConstruction:

    def test_repr(self, val):
        r = repr(val)
        assert "SpatialValidator" in r
        assert "n_valid=" in r
        assert "n_modes=" in r

    def test_unfitted_generator_raises(self, field):
        g = SMBFieldGenerator()
        with pytest.raises(ValueError, match="fitted"):
            SpatialValidator(g, field)

    def test_basin_mask_shape(self, val):
        assert val._basin_mask.shape == (NY, NX)

    def test_obs_mean_map_shape(self, val):
        assert val._obs_mean_map.shape == (NY, NX)

    def test_obs_var_map_shape(self, val):
        assert val._obs_var_map.shape == (NY, NX)

    def test_obs_flat_shape(self, val):
        n_valid = int(_MASK.sum())
        assert val._obs_flat.shape == (NT, n_valid)


# ======================================================================
# 2. compute_metrics() — structure and values
# ======================================================================

class TestComputeMetrics:

    def test_returns_dict(self, val, baseline_ds):
        m = val.compute_metrics(baseline_ds, verbose=False)
        assert isinstance(m, dict)

    def test_all_keys_present(self, val, baseline_ds):
        m = val.compute_metrics(baseline_ds, verbose=False)
        required = {
            "mean_ratio", "variance_ratio",
            "mean_map_corr", "variance_map_corr",
            "variance_map_ratio", "eof_cumvar", "pc_metrics",
        }
        assert required.issubset(m.keys())

    def test_mean_ratio_finite(self, val, baseline_ds):
        m = val.compute_metrics(baseline_ds, verbose=False)
        assert np.isfinite(m["mean_ratio"])

    def test_variance_ratio_finite(self, val, baseline_ds):
        m = val.compute_metrics(baseline_ds, verbose=False)
        assert np.isfinite(m["variance_ratio"])

    def test_mean_ratio_in_plausible_range(self, val, baseline_ds):
        """With correct mean restoration, ratio should be near 1."""
        m = val.compute_metrics(baseline_ds, verbose=False)
        assert 0.3 < m["mean_ratio"] < 3.0, (
            f"mean_ratio={m['mean_ratio']:.4f} outside plausible range."
        )

    def test_variance_ratio_positive(self, val, baseline_ds):
        m = val.compute_metrics(baseline_ds, verbose=False)
        assert m["variance_ratio"] > 0

    def test_map_correlations_in_minus1_to_1(self, val, baseline_ds):
        m = val.compute_metrics(baseline_ds, verbose=False)
        assert -1.0 <= m["mean_map_corr"]     <= 1.0
        assert -1.0 <= m["variance_map_corr"] <= 1.0

    def test_eof_cumvar_in_0_to_1(self, val, baseline_ds):
        m = val.compute_metrics(baseline_ds, verbose=False)
        assert 0.0 < m["eof_cumvar"] <= 1.0 + 1e-6

    def test_pc_metrics_length(self, val, baseline_ds):
        m = val.compute_metrics(baseline_ds, verbose=False)
        assert len(m["pc_metrics"]) == N_MODES

    def test_pc_metrics_have_mode_key(self, val, baseline_ds):
        m = val.compute_metrics(baseline_ds, verbose=False)
        for pc in m["pc_metrics"]:
            assert "mode" in pc

    def test_metrics_differ_across_experiments(self, val, gen):
        """Band-scaled experiment should give different metrics from baseline."""
        exp_base = Experiment(n_years=10, n_members=3, seed=0)
        exp_ann  = Experiment(
            n_years=10, n_members=3, seed=0,
            band_scales=[(0.8, 1.5, 10.0)], name="annual_enhanced",
        )
        ds_base = gen.generate(exp_base)
        ds_ann  = gen.generate(exp_ann)

        m_base = val.compute_metrics(ds_base, verbose=False)
        m_ann  = val.compute_metrics(ds_ann,  verbose=False)

        # Band scaling should change the variance ratio
        assert m_base["variance_ratio"] != m_ann["variance_ratio"]


# ======================================================================
# 3. Plot methods (smoke tests — just check they run without error)
# ======================================================================

class TestPlotMethods:

    def test_plot_validation_suite_runs(self, val, baseline_ds):
        val.plot_validation_suite(baseline_ds, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")

    def test_plot_variance_maps_runs(self, val, baseline_ds):
        val.plot_variance_maps(baseline_ds, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")

    def test_plot_pc_validation_runs(self, val):
        val.plot_pc_validation(n_pcs=2, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")

    def test_plot_pc_validation_more_than_n_modes_clips(self, val):
        """Requesting more PCs than n_modes should not raise."""
        val.plot_pc_validation(n_pcs=100, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")


# ======================================================================
# 4. Calibration split test
# ======================================================================

class TestCalibrationSplit:

    def test_returns_dict_with_two_halves(self, val):
        results = val.calibration_split_test(
            n_members=2, n_years_syn=5, verbose=False
        )
        assert "first_half"  in results
        assert "second_half" in results

    def test_each_half_has_metrics(self, val):
        results = val.calibration_split_test(
            n_members=2, n_years_syn=5, verbose=False
        )
        required = {"mean_ratio", "variance_ratio", "mean_map_corr"}
        for half in ("first_half", "second_half"):
            assert required.issubset(results[half].keys())

    def test_split_metrics_finite(self, val):
        results = val.calibration_split_test(
            n_members=2, n_years_syn=5, verbose=False
        )
        for half in ("first_half", "second_half"):
            for key in ("mean_ratio", "variance_ratio", "mean_map_corr"):
                val_ = results[half][key]
                assert np.isfinite(val_), (
                    f"{half}.{key} = {val_} is not finite."
                )


# ======================================================================
# 5. Spatial structure of output
# ======================================================================

class TestSpatialStructure:

    def test_mean_map_nan_outside_basin(self, val, baseline_ds):
        smb = baseline_ds["smb_syn"]
        syn_mean_map = np.nanmean(smb.values, axis=(0, 1))
        outside = ~_MASK
        # Outside basin should be NaN (from nanmean of all-NaN slices)
        assert np.all(np.isnan(syn_mean_map[outside]))

    def test_obs_var_map_nan_outside_basin(self, val):
        outside = ~_MASK
        assert np.all(np.isnan(val._obs_var_map[outside]))

    def test_obs_var_map_positive_inside_basin(self, val):
        inside = _MASK
        assert np.all(val._obs_var_map[inside] >= 0)


# ======================================================================
# Real-data integration (Steps 1–5 + 7)
# ======================================================================

def run_real_data(
    racmo_path:  str,
    shp_path:    str,
    basin_name:  str,
    n_modes:     int = 10,
) -> None:
    """
    Full integration test including SpatialValidator.
    """
    print(f"\n{'='*60}")
    print(f"SpatialValidator — real data: {basin_name}")
    print(f"{'='*60}")

    from syn_smb import SMBFieldLoader
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    # ── Steps 1–5: Fit generator ─────────────────────────────────────
    loader = SMBFieldLoader(racmo_path, shp_path, basin_name)
    field  = loader.load()

    gen = SMBFieldGenerator(n_modes=n_modes, nperseg=60)
    gen.fit(field, lat=loader.lat)

    # ── Generate baseline ─────────────────────────────────────────────
    # Small ensemble for the integration test — validates the pipeline
    # without OOM. Production runs use n_years=1000, n_members=30.
    exp = Experiment(n_years=20, n_members=3, seed=0)
    ds  = gen.generate(exp)

    # ── Step 7: Validate ──────────────────────────────────────────────
    val = SpatialValidator(gen, field)
    print(f"\n{val}")

    print("\n[Baseline metrics]")
    metrics = val.compute_metrics(ds, verbose=True)

    # Figures
    print("\nGenerating figures...")
    val.plot_validation_suite(
        ds, save_path=f"spatial_validator_{basin_name}_suite.png"
    )
    val.plot_variance_maps(
        ds, save_path=f"spatial_validator_{basin_name}_variance.png"
    )
    val.plot_pc_validation(
        n_pcs=min(4, n_modes),
        save_path=f"spatial_validator_{basin_name}_pcs.png",
    )

    # Calibration split
    print("\n[Calibration split test]")
    split_results = val.calibration_split_test(
        n_members=5, n_years_syn=50, verbose=True
    )

    assert abs(metrics["mean_ratio"] - 1.0) < 0.5, "Mean ratio far from 1."
    assert metrics["eof_cumvar"] > 0.5,             "EOF cumvar < 50%"
    assert np.isfinite(metrics["mean_map_corr"]),   "mean_map_corr not finite"
    assert ds["smb_syn"].sizes["member"] == 3
    assert ds["smb_syn"].sizes["time"]   == 20 * 12

    print(f"\n✓ All sanity checks passed.")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        n_modes = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        run_real_data(sys.argv[1], sys.argv[2], sys.argv[3], n_modes)
    else:
        print(
            "Usage: python test_spatial_validator.py "
            "<racmo.nc> <shp> <basin_name> [n_modes]\n"
            "Running pytest instead..."
        )
        import subprocess
        subprocess.run(["python", "-m", "pytest", __file__, "-v"])