"""
test_paper_figures.py
=====================
Tests and integration script for the paper-figure additions:
  - Validator.plot_convergence()
  - Validator.plot_calibration_split()
  - Validator.plot_band_comparison()
  - multi_basin_run()
  - plot_multibasin_psd()
  - multibasin_metrics_table()
  - plot_multibasin_metrics()

Unit tests (fake data):
    pytest test_paper_figures.py -v

Full integration with real data (multiple basins):
    python test_paper_figures.py /path/PIG.nc /path/Thwaites.nc ...
"""

import sys
import numpy as np
import xarray as xr
import pytest
import matplotlib
matplotlib.use("Agg")

from syn_smb import SMBGenerator
from syn_smb import Experiment
from syn_smb import Validator
from syn_smb.core.multi_basin import (
    multi_basin_run,
    plot_multibasin_psd,
    multibasin_metrics_table,
    plot_multibasin_metrics,
    plot_multibasin_split_test,
)


# ======================================================================
# Shared fixtures
# ======================================================================

@pytest.fixture(scope="module")
def two_basin_results(tmp_path_factory):
    """
    Two synthetic 'basins' with slightly different SMB statistics,
    run through multi_basin_run() using a temporary NetCDF approach.
    Because multi_basin_run takes file paths, we create two tiny NetCDF
    files from synthetic data.
    """
    import tempfile, os
    import netCDF4 as nc4

    rng  = np.random.default_rng(0)
    time = xr.cftime_range(start="1979", periods=540, freq="MS")
    t    = np.arange(540)
    months = np.array([d.month for d in time.values])

    paths = {}
    for i, (basin, mean_smb, std_noise) in enumerate([
        ("BasinA", 0.04, 0.02),
        ("BasinB", 0.06, 0.03),
    ]):
        values = (
            mean_smb
            + 5e-5 * t
            + std_noise * np.sin(2 * np.pi * (months - 3) / 12)
            + rng.normal(0, std_noise * 0.5, 540)
        )
        da = xr.DataArray(
            values,
            coords={"time": time},
            dims=["time"],
            name="smbgl",
            attrs={"units": "m w.e. a$^{-1}$"},
        )
        ds = xr.Dataset({"smbgl": da})
        path = str(tmp_path_factory.mktemp("data") / f"{basin}_smb.nc")
        ds.to_netcdf(path)
        paths[basin] = path

    return multi_basin_run(
        paths,
        suite=Experiment.standard_suite(n_years=20, n_members=5, seed=0),
        nperseg=30,
        verbose=False,
    )


@pytest.fixture(scope="module")
def single_validator(two_basin_results):
    basin  = list(two_basin_results.keys())[0]
    res    = two_basin_results[basin]
    return res["validator"], res["datasets"]


# ======================================================================
# 1. plot_convergence
# ======================================================================

class TestPlotConvergence:

    def test_runs_without_error(self, single_validator):
        val, _ = single_validator
        conv = val.convergence_test(
            member_counts=[3, 5, 8], n_years=10, verbose=False
        )
        val.plot_convergence(conv, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")

    def test_accepts_partial_counts(self, single_validator):
        val, _ = single_validator
        conv = val.convergence_test(
            member_counts=[3, 5], n_years=10, verbose=False
        )
        val.plot_convergence(conv, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")


# ======================================================================
# 2. plot_calibration_split
# ======================================================================

class TestPlotCalibrationSplit:

    def test_runs_without_error(self, single_validator):
        val, _ = single_validator
        split = val.calibration_split_test(
            n_members=3, n_years_syn=10, verbose=False
        )
        val.plot_calibration_split(split, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")

    def test_both_halves_present(self, single_validator):
        val, _ = single_validator
        split = val.calibration_split_test(
            n_members=3, n_years_syn=10, verbose=False
        )
        assert "first_half"  in split
        assert "second_half" in split


# ======================================================================
# 3. plot_band_comparison
# ======================================================================

class TestPlotBandComparison:

    def test_runs_without_error(self, single_validator):
        val, datasets = single_validator
        val.plot_band_comparison(datasets, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")

    def test_handles_single_experiment(self, single_validator):
        val, datasets = single_validator
        subset = {k: v for k, v in datasets.items() if k == "baseline"}
        val.plot_band_comparison(subset, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")


# ======================================================================
# 4. multi_basin_run
# ======================================================================

class TestMultiBasinRun:

    def test_returns_dict_with_all_basins(self, two_basin_results):
        assert "BasinA" in two_basin_results
        assert "BasinB" in two_basin_results

    def test_each_result_has_expected_keys(self, two_basin_results):
        for basin, res in two_basin_results.items():
            assert "generator" in res
            assert "smb"       in res
            assert "datasets"  in res
            assert "validator" in res
            assert "metrics"   in res

    def test_generators_are_fitted(self, two_basin_results):
        for basin, res in two_basin_results.items():
            assert res["generator"].is_fitted

    def test_metrics_computed_for_all_experiments(self, two_basin_results):
        for basin, res in two_basin_results.items():
            suite = Experiment.standard_suite(n_years=20, n_members=5)
            for exp in suite:
                assert exp.name in res["metrics"]

    def test_mean_ratios_near_one_for_all_basins(self, two_basin_results):
        for basin, res in two_basin_results.items():
            mr = res["metrics"]["baseline"]["mean_ratio"]
            assert abs(mr - 1.0) < 0.05, (
                f"{basin} baseline mean_ratio = {mr:.4f}"
            )


# ======================================================================
# 5. plot_multibasin_psd
# ======================================================================

class TestPlotMultibasinPsd:

    def test_runs_without_error(self, two_basin_results):
        plot_multibasin_psd(two_basin_results, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")

    def test_handles_single_basin(self, two_basin_results):
        single = {"BasinA": two_basin_results["BasinA"]}
        plot_multibasin_psd(single, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")


# ======================================================================
# 6. multibasin_metrics_table
# ======================================================================

class TestMultibasinMetricsTable:

    def test_returns_dict_with_all_basins(self, two_basin_results):
        table = multibasin_metrics_table(
            two_basin_results, experiment="baseline", verbose=False
        )
        assert "BasinA" in table
        assert "BasinB" in table

    def test_each_basin_has_all_metrics(self, two_basin_results):
        table = multibasin_metrics_table(
            two_basin_results, experiment="baseline", verbose=False
        )
        expected = {
            "mean_ratio", "variance_ratio", "ks_statistic", "ks_pvalue",
            "psd_rms_error", "acf_lag1_error", "acf_lag12_error", "seasonal_rmse"
        }
        for basin, m in table.items():
            assert set(m.keys()) == expected

    def test_plot_multibasin_metrics_runs(self, two_basin_results):
        plot_multibasin_metrics(
            two_basin_results, experiment="baseline", save_path=None
        )
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")


# ======================================================================
# Full integration with real RACMO data
# Run with: python test_paper_figures.py /path/PIG.nc /path/Thwaites.nc
# ======================================================================

def run_real_data(paths: list[str]) -> None:
    """
    Full paper figure generation from real RACMO files.
    Pass one or more NetCDF paths as command-line arguments.
    Basin names are inferred from filenames.
    """
    import os
    matplotlib.use("TkAgg")   # switch back to display for interactive use

    basin_paths = {}
    for p in paths:
        name = os.path.basename(p).replace("_smb.nc", "").replace(".nc", "")
        basin_paths[name] = p

    print(f"Basins: {list(basin_paths.keys())}")

    # ── Run pipeline across all basins ──
    results = multi_basin_run(basin_paths, verbose=True)

    # ── Multi-basin PSD comparison (Figure 3 in paper) ──
    print("\nGenerating multi-basin PSD comparison...")
    plot_multibasin_psd(results, save_path="paper_fig_multibasin_psd.png")

    # ── Metrics table (Table 2 in paper) ──
    print("\nValidation metrics table (baseline experiment):")
    multibasin_metrics_table(results, experiment="baseline", verbose=True)
    plot_multibasin_metrics(results, save_path="paper_fig_metrics_heatmap.png")

    # ── Per-basin figures ──
    for basin, res in results.items():
        val      = res["validator"]
        datasets = res["datasets"]

        print(f"\n── {basin} ──")

        # Validation suite
        val.plot_validation_suite(
            datasets["baseline"],
            save_path=f"paper_fig_{basin}_validation_suite.png",
        )

        # Band comparison
        val.plot_band_comparison(
            datasets,
            save_path=f"paper_fig_{basin}_band_comparison.png",
        )

        # Calibration split
        split = val.calibration_split_test(n_members=30, n_years_syn=100)
        val.plot_calibration_split(
            split,
            save_path=f"paper_fig_{basin}_calibration_split.png",
        )

        # Convergence
        conv = val.convergence_test(
            member_counts=[5, 10, 15, 20, 30, 50],
            n_years=100,
        )
        val.plot_convergence(
            conv,
            save_path=f"paper_fig_{basin}_convergence.png",
        )

        # Running windows
        val.plot_running_windows(
            datasets["baseline"],
            save_path=f"paper_fig_{basin}_running_windows.png",
        )

        # Return periods
        val.plot_return_periods(
            datasets["baseline"],
            save_path=f"paper_fig_{basin}_return_periods.png",
        )

    # ── Multi-basin calibration split comparison ──
    print("\nMulti-basin calibration split test...")
    plot_multibasin_split_test(results, save_path="paper_fig_multibasin_split.png")

    print("\nAll figures saved. Ready for Paper 1.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_real_data(sys.argv[1:])
    else:
        print("Usage: python test_paper_figures.py /path/PIG.nc [/path/Thwaites.nc ...]")
        print("Running pytest unit tests instead...")
        import subprocess
        subprocess.run(["python", "-m", "pytest", __file__, "-v"])