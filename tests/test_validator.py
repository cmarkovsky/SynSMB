"""
test_validator.py
=================
Tests for Validator.

Run directly for step-by-step printed output:
    python test_validator.py

Run via pytest for pass/fail:
    pytest test_validator.py -v
"""

import numpy as np
import xarray as xr
import pytest
from syn_smb.core.generator import SMBGenerator
from syn_smb.core.experiment import Experiment
from syn_smb.core.validator import Validator


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def monthly_time():
    return xr.cftime_range(start="1979", periods=540, freq="MS")


@pytest.fixture
def synthetic_smb(monthly_time, rng):
    t      = np.arange(540)
    months = np.array([d.month for d in monthly_time.values])
    values = (0.04
              + 5e-5 * t
              + 0.03 * np.sin(2 * np.pi * (months - 3) / 12)
              + rng.normal(0, 0.02, 540))
    return xr.DataArray(
        values, coords={"time": monthly_time}, dims=["time"], name="smb"
    )


@pytest.fixture
def fitted_generator(synthetic_smb):
    gen = SMBGenerator(nperseg=30)
    gen.fit(synthetic_smb)
    return gen, synthetic_smb


@pytest.fixture
def validator(fitted_generator):
    gen, smb = fitted_generator
    return Validator(gen, smb), gen, smb


@pytest.fixture
def baseline_dataset(fitted_generator):
    gen, _ = fitted_generator
    return gen.generate(Experiment(n_years=100, n_members=10, seed=0))


# ======================================================================
# 1. Construction
# ======================================================================

class TestConstruction:

    def test_raises_on_unfitted_generator(self, synthetic_smb):
        gen = SMBGenerator()
        with pytest.raises(RuntimeError, match="fitted"):
            Validator(gen, synthetic_smb)

    def test_stores_generator_and_smb(self, validator):
        val, gen, smb = validator
        assert val.generator is gen
        np.testing.assert_array_equal(val.smb.values, smb.values)

    def test_repr(self, validator):
        val, _, _ = validator
        assert "Validator" in repr(val)


# ======================================================================
# 2. compute_metrics
# ======================================================================

class TestComputeMetrics:

    def test_returns_expected_keys(self, validator, baseline_dataset):
        val, _, _ = validator
        metrics = val.compute_metrics(baseline_dataset, verbose=False)
        expected = {
            "mean_ratio", "variance_ratio", "ks_statistic", "ks_pvalue",
            "psd_rms_error", "acf_lag1_error", "acf_lag12_error",
            "seasonal_rmse",
        }
        assert set(metrics.keys()) == expected

    def test_mean_ratio_near_one(self, validator, baseline_dataset):
        val, _, _ = validator
        m = val.compute_metrics(baseline_dataset, verbose=False)
        assert abs(m["mean_ratio"] - 1.0) < 0.05, (
            f"mean_ratio = {m['mean_ratio']:.4f}, expected ~1.0"
        )

    def test_variance_ratio_near_one(self, validator, baseline_dataset):
        val, _, _ = validator
        m = val.compute_metrics(baseline_dataset, verbose=False)
        assert abs(m["variance_ratio"] - 1.0) < 0.3, (
            f"variance_ratio = {m['variance_ratio']:.4f}, expected ~1.0"
        )

    def test_seasonal_rmse_near_zero(self, validator, baseline_dataset):
        """
        The seasonal cycle is restored deterministically, so the RMSE
        between observed and synthetic monthly climatology should be tiny.
        """
        val, _, _ = validator
        m = val.compute_metrics(baseline_dataset, verbose=False)
        assert m["seasonal_rmse"] < 0.005, (
            f"seasonal_rmse = {m['seasonal_rmse']:.6f}, expected < 0.005"
        )

    def test_psd_rms_error_reasonable(self, validator, baseline_dataset):
        val, _, _ = validator
        m = val.compute_metrics(baseline_dataset, verbose=False)
        assert m["psd_rms_error"] < 1.0, (
            f"psd_rms_error = {m['psd_rms_error']:.4f}, expected < 1.0"
        )

    def test_all_values_are_finite(self, validator, baseline_dataset):
        val, _, _ = validator
        m = val.compute_metrics(baseline_dataset, verbose=False)
        for key, val_item in m.items():
            assert np.isfinite(val_item), f"{key} = {val_item} is not finite"


# ======================================================================
# 3. calibration_split_test
# ======================================================================

class TestCalibrationSplitTest:

    def test_returns_two_halves(self, validator):
        val, _, _ = validator
        results = val.calibration_split_test(
            n_members=5, n_years_syn=20, verbose=False
        )
        assert "first_half" in results
        assert "second_half" in results

    def test_each_half_has_metrics(self, validator):
        val, _, _ = validator
        results = val.calibration_split_test(
            n_members=5, n_years_syn=20, verbose=False
        )
        for half in ["first_half", "second_half"]:
            assert "mean_ratio" in results[half]
            assert "variance_ratio" in results[half]

    def test_metrics_are_finite(self, validator):
        val, _, _ = validator
        results = val.calibration_split_test(
            n_members=5, n_years_syn=20, verbose=False
        )
        for half in ["first_half", "second_half"]:
            for k, v in results[half].items():
                assert np.isfinite(v), f"{half}/{k} = {v} is not finite"

    def test_mean_ratios_near_one(self, validator):
        """
        In the calibration split test, mean_ratio is NOT expected to be
        close to 1.0 when the data has a trend — the two halves of a
        trended record will have genuinely different means, and a
        generator calibrated on one half will produce series with the
        training-half mean, not the held-out-half mean. This is the
        correct behaviour.

        We use a loose tolerance here to catch obvious failures (e.g.
        sign errors or 10x errors) while allowing for real trend effects.
        The spectral metrics (psd_rms_error, variance_ratio) are the
        meaningful ones for the calibration split test.
        """
        val, _, _ = validator
        results = val.calibration_split_test(
            n_members=10, n_years_syn=50, verbose=False
        )
        for half in ["first_half", "second_half"]:
            mr = results[half]["mean_ratio"]
            assert 0.3 < mr < 2.0, (
                f"{half} mean_ratio = {mr:.4f} is outside the plausible "
                f"range (0.3, 2.0). This suggests an error beyond normal "
                f"trend effects."
            )


# ======================================================================
# 4. convergence_test
# ======================================================================

class TestConvergenceTest:

    def test_returns_expected_keys(self, validator):
        val, _, _ = validator
        results = val.convergence_test(
            member_counts=[5, 10], n_years=20, verbose=False
        )
        expected = {
            "member_counts", "variance_ratio", "psd_rms_error",
            "ks_statistic", "mean_ratio"
        }
        assert set(results.keys()) == expected

    def test_lists_match_member_counts(self, validator):
        val, _, _ = validator
        counts = [5, 10, 20]
        results = val.convergence_test(
            member_counts=counts, n_years=20, verbose=False
        )
        for key in ["variance_ratio", "psd_rms_error", "ks_statistic"]:
            assert len(results[key]) == len(counts)

    def test_all_values_finite(self, validator):
        val, _, _ = validator
        results = val.convergence_test(
            member_counts=[5, 10], n_years=20, verbose=False
        )
        for key in ["variance_ratio", "psd_rms_error", "ks_statistic",
                    "mean_ratio"]:
            for v in results[key]:
                assert np.isfinite(v), f"{key} contains non-finite value {v}"

    def test_mean_ratios_near_one_for_all_sizes(self, validator):
        val, _, _ = validator
        results = val.convergence_test(
            member_counts=[5, 10, 20], n_years=50, verbose=False
        )
        for mr in results["mean_ratio"]:
            assert abs(mr - 1.0) < 0.1, (
                f"mean_ratio = {mr:.4f} not near 1.0"
            )


# ======================================================================
# 5. Plotting (smoke tests — just verify they run without error)
# ======================================================================

class TestPlotting:

    def test_plot_validation_suite_runs(self, validator, baseline_dataset):
        import matplotlib
        matplotlib.use("Agg")
        val, _, _ = validator
        val.plot_validation_suite(baseline_dataset, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")

    def test_plot_ensemble_spaghetti_runs(self, validator, baseline_dataset):
        import matplotlib
        matplotlib.use("Agg")
        val, _, _ = validator
        val.plot_ensemble_spaghetti(baseline_dataset, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")

    def test_plot_running_windows_runs(self, validator, baseline_dataset):
        import matplotlib
        matplotlib.use("Agg")
        val, _, _ = validator
        val.plot_running_windows(baseline_dataset, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")

    def test_plot_return_periods_runs(self, validator, baseline_dataset):
        import matplotlib
        matplotlib.use("Agg")
        val, _, _ = validator
        val.plot_return_periods(baseline_dataset, save_path=None)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close("all")


# ======================================================================
# Standalone debug runner
# ======================================================================

def run_debug():
    import matplotlib
    matplotlib.use("Agg")  # no display needed for debug

    rng = np.random.default_rng(42)
    time = xr.cftime_range(start="1979", periods=540, freq="MS")
    months = np.array([d.month for d in time.values])
    t = np.arange(540)
    values = (0.04 + 5e-5 * t
              + 0.03 * np.sin(2 * np.pi * (months - 3) / 12)
              + rng.normal(0, 0.02, 540))
    smb = xr.DataArray(values, coords={"time": time}, dims=["time"])

    print("=" * 60)
    print("Validator — step-by-step debug run")
    print("=" * 60)

    gen = SMBGenerator(nperseg=30)
    gen.fit(smb)
    val = Validator(gen, smb)
    print(f"\n{val}")

    print("\n[1] compute_metrics() on baseline...")
    ds = gen.generate(Experiment(n_years=100, n_members=10, seed=0))
    metrics = val.compute_metrics(ds, verbose=True)

    print("\n[2] compute_metrics() across standard suite...")
    suite    = Experiment.standard_suite(n_years=50, n_members=5, seed=0)
    datasets = gen.generate_suite(suite)
    print(f"    {'Experiment':<35}  {'mean_ratio':>10}  {'var_ratio':>10}  {'psd_rms':>10}")
    for name, result_ds in datasets.items():
        m = val.compute_metrics(result_ds, verbose=False)
        print(f"    {name:<35}  {m['mean_ratio']:>10.4f}  "
              f"{m['variance_ratio']:>10.4f}  {m['psd_rms_error']:>10.4f}")

    print("\n[3] calibration_split_test()...")
    split = val.calibration_split_test(n_members=5, n_years_syn=30, verbose=True)

    print("\n[4] convergence_test()...")
    conv = val.convergence_test(
        member_counts=[5, 10, 20, 30], n_years=50, verbose=True
    )

    print("\n[5] All plot methods (non-display mode)...")
    for method_name in ["plot_validation_suite", "plot_ensemble_spaghetti",
                        "plot_running_windows", "plot_return_periods"]:
        getattr(val, method_name)(ds, save_path=None)
        print(f"    {method_name}() — OK")

    print("\n" + "=" * 60)
    print("Debug run complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_debug()