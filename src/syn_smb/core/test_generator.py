"""
test_generator.py
=================
Tests for SMBGenerator.

Run directly for step-by-step printed output:
    python test_generator.py

Run via pytest for pass/fail:
    pytest test_generator.py -v
"""

import numpy as np
import xarray as xr
import pytest
from syn_smb.core.generator import SMBGenerator
from syn_smb.core.experiment import Experiment


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
    """
    Realistic synthetic SMB: mean + trend + seasonal cycle + noise.
    Known components so we can verify each step.
    """
    t      = np.arange(len(monthly_time))
    months = np.array([d.month for d in monthly_time.values])

    mean     = 0.04
    trend    = 5e-5 * t
    seasonal = 0.03 * np.sin(2 * np.pi * (months - 3) / 12)
    noise    = rng.normal(0, 0.02, len(t))

    values = mean + trend + seasonal + noise
    return xr.DataArray(
        values,
        coords={"time": monthly_time},
        dims=["time"],
        name="smb",
        attrs={"units": "m w.e. a$^{-1}$"},
    )


@pytest.fixture
def fitted_generator(synthetic_smb):
    """SMBGenerator fitted on the synthetic_smb fixture."""
    gen = SMBGenerator(nperseg=30)  # shorter nperseg for fast tests
    gen.fit(synthetic_smb)
    return gen, synthetic_smb


# ======================================================================
# 1. fit()
# ======================================================================

class TestFit:

    def test_sets_is_fitted(self, synthetic_smb):
        gen = SMBGenerator(nperseg=30)
        gen.fit(synthetic_smb)
        assert gen.is_fitted

    def test_stores_smb_mean(self, synthetic_smb):
        gen = SMBGenerator(nperseg=30)
        gen.fit(synthetic_smb)
        assert abs(gen.smb_mean - float(synthetic_smb.mean())) < 1e-10

    def test_stores_n_obs(self, synthetic_smb):
        gen = SMBGenerator(nperseg=30)
        gen.fit(synthetic_smb)
        assert gen.n_obs == len(synthetic_smb)

    def test_stores_start_year(self, synthetic_smb):
        gen = SMBGenerator(nperseg=30)
        gen.fit(synthetic_smb)
        assert gen.smb_start_year == 1979

    def test_all_components_fitted(self, fitted_generator):
        gen, _ = fitted_generator
        assert gen.preprocessor.is_fitted
        assert gen.gaussian_transform.is_fitted
        assert gen.spectral_synthesizer.is_fitted

    def test_returns_self_for_chaining(self, synthetic_smb):
        gen = SMBGenerator(nperseg=30)
        result = gen.fit(synthetic_smb)
        assert result is gen

    def test_raises_on_missing_time_coord(self):
        gen = SMBGenerator()
        da = xr.DataArray(np.random.standard_normal(100), dims=["x"])
        with pytest.raises(ValueError, match="'time' coordinate"):
            gen.fit(da)

    def test_properties_available_after_fit(self, fitted_generator):
        gen, _ = fitted_generator
        assert gen.freqs is not None
        assert gen.psd is not None
        assert gen.psd_ci_lower is not None
        assert gen.psd_ci_upper is not None
        assert gen.seasonal_cycle is not None


# ======================================================================
# 2. generate()
# ======================================================================

class TestGenerate:

    def test_returns_dataset(self, fitted_generator):
        gen, _ = fitted_generator
        ds = gen.generate(Experiment(n_years=10, n_members=3, seed=0))
        assert isinstance(ds, xr.Dataset)

    def test_dataset_has_expected_variables(self, fitted_generator):
        gen, _ = fitted_generator
        ds = gen.generate(Experiment(n_years=10, n_members=3, seed=0))
        assert "smb_syn"   in ds
        assert "resid_syn" in ds
        assert "g_syn"     in ds

    def test_output_shape_correct(self, fitted_generator):
        gen, _ = fitted_generator
        n_years, n_members = 10, 5
        ds = gen.generate(Experiment(n_years=n_years, n_members=n_members, seed=0))
        expected_time = n_years * 12
        assert ds["smb_syn"].shape   == (n_members, expected_time)
        assert ds["resid_syn"].shape == (n_members, expected_time)
        assert ds["g_syn"].shape     == (n_members, expected_time)

    def test_time_coord_starts_at_start_year(self, fitted_generator):
        gen, _ = fitted_generator
        ds = gen.generate(Experiment(n_years=10, n_members=2, seed=0))
        first_year = int(str(ds["smb_syn"]["time"].values[0])[:4])
        assert first_year == gen.smb_start_year

    def test_mean_preservation_baseline(self, fitted_generator):
        """
        smb_syn mean should be close to observed smb_mean.
        This is the end-to-end mean preservation test.
        """
        gen, _ = fitted_generator
        ds = gen.generate(Experiment(n_years=100, n_members=10, seed=0))
        all_vals = ds["smb_syn"].values.ravel()
        assert abs(all_vals.mean() - gen.smb_mean) < 0.01, (
            f"Mean not preserved: synthetic={all_vals.mean():.5f}, "
            f"observed={gen.smb_mean:.5f}"
        )

    def test_mean_preservation_band_scaled(self, fitted_generator):
        """
        Mean preservation must hold even with 10x band scaling.
        This tests the full mean shift fix end-to-end.
        """
        gen, _ = fitted_generator
        exp = Experiment.annual_enhanced(factor=10.0, n_years=100, n_members=10)
        ds = gen.generate(exp)
        all_vals = ds["smb_syn"].values.ravel()
        assert abs(all_vals.mean() - gen.smb_mean) < 0.01, (
            f"Mean not preserved under 10x annual scaling: "
            f"synthetic={all_vals.mean():.5f}, observed={gen.smb_mean:.5f}"
        )

    def test_g_syn_has_unit_variance(self, fitted_generator):
        """Each g_syn member should have variance ~1.0."""
        gen, _ = fitted_generator
        ds = gen.generate(Experiment(n_years=50, n_members=5, seed=0))
        for i in range(5):
            v = float(ds["g_syn"].isel(member=i).var())
            assert abs(v - 1.0) < 1e-10, f"g_syn member {i} variance = {v:.6f}"

    def test_resid_syn_is_zero_mean(self, fitted_generator):
        """resid_syn should be approximately zero-mean (mean shift fix)."""
        gen, _ = fitted_generator
        ds = gen.generate(Experiment(n_years=50, n_members=5, seed=0))
        for i in range(5):
            m = float(ds["resid_syn"].isel(member=i).mean())
            assert abs(m) < 1e-9, f"resid_syn member {i} mean = {m:.2e}"

    def test_members_are_independent(self, fitted_generator):
        gen, _ = fitted_generator
        ds = gen.generate(Experiment(n_years=50, n_members=2, seed=0))
        m0 = ds["smb_syn"].isel(member=0).values
        m1 = ds["smb_syn"].isel(member=1).values
        assert not np.allclose(m0, m1), "Members are identical"

    def test_reproducible_with_same_experiment(self, fitted_generator):
        gen, _ = fitted_generator
        exp = Experiment(n_years=10, n_members=3, seed=99)
        ds1 = gen.generate(exp)
        ds2 = gen.generate(exp)
        np.testing.assert_array_equal(
            ds1["smb_syn"].values, ds2["smb_syn"].values
        )

    def test_dataset_has_experiment_metadata(self, fitted_generator):
        gen, _ = fitted_generator
        exp = Experiment.annual_enhanced(factor=5.0)
        ds = gen.generate(exp)
        assert "experiment_name" in ds.attrs
        assert ds.attrs["experiment_name"] == exp.name

    def test_raises_when_not_fitted(self):
        gen = SMBGenerator()
        with pytest.raises(RuntimeError, match="not fitted"):
            gen.generate(Experiment.baseline())


# ======================================================================
# 3. generate_suite()
# ======================================================================

class TestGenerateSuite:

    def test_returns_dict(self, fitted_generator):
        gen, _ = fitted_generator
        suite = Experiment.standard_suite(n_years=5, n_members=2)
        results = gen.generate_suite(suite)
        assert isinstance(results, dict)

    def test_keys_match_experiment_names(self, fitted_generator):
        gen, _ = fitted_generator
        suite = Experiment.standard_suite(n_years=5, n_members=2)
        results = gen.generate_suite(suite)
        for exp in suite:
            assert exp.name in results

    def test_each_value_is_dataset(self, fitted_generator):
        gen, _ = fitted_generator
        suite = [Experiment.baseline(n_years=5, n_members=2),
                 Experiment.annual_enhanced(n_years=5, n_members=2)]
        results = gen.generate_suite(suite)
        for ds in results.values():
            assert isinstance(ds, xr.Dataset)

    def test_all_means_preserved(self, fitted_generator):
        """Mean preservation must hold for every experiment in the suite."""
        gen, _ = fitted_generator
        suite = Experiment.standard_suite(n_years=50, n_members=5)
        results = gen.generate_suite(suite)
        for name, ds in results.items():
            m = float(ds["smb_syn"].values.ravel().mean())
            assert abs(m - gen.smb_mean) < 0.01, (
                f"{name}: mean={m:.5f}, expected={gen.smb_mean:.5f}"
            )


# ======================================================================
# 4. validate()
# ======================================================================

class TestValidate:

    def test_returns_expected_keys(self, fitted_generator):
        gen, _ = fitted_generator
        results = gen.validate(verbose=False)
        expected = {"gaussian_transform", "spectral_synthesizer",
                    "stationarity", "passed"}
        assert set(results.keys()) == expected

    def test_passes_on_good_data(self, fitted_generator):
        gen, _ = fitted_generator
        results = gen.validate(verbose=False)
        assert results["passed"], f"validate() failed:\n{results}"

    def test_raises_when_not_fitted(self):
        gen = SMBGenerator()
        with pytest.raises(RuntimeError, match="not fitted"):
            gen.validate()


# ======================================================================
# 5. save() / persistence
# ======================================================================

class TestSave:

    def test_save_creates_file(self, fitted_generator, tmp_path):
        gen, _ = fitted_generator
        ds = gen.generate(Experiment(n_years=5, n_members=2, seed=0))
        outfile = tmp_path / "test_ensemble.nc"
        gen.save(ds, str(outfile))
        assert outfile.exists()

    def test_saved_file_is_loadable(self, fitted_generator, tmp_path):
        gen, _ = fitted_generator
        ds = gen.generate(Experiment(n_years=5, n_members=2, seed=0))
        outfile = tmp_path / "test_ensemble.nc"
        gen.save(ds, str(outfile))
        loaded = xr.open_dataset(outfile)
        assert "smb_syn" in loaded

    def test_experiment_metadata_in_saved_file(self, fitted_generator, tmp_path):
        gen, _ = fitted_generator
        exp = Experiment.annual_enhanced(n_years=5, n_members=2)
        ds = gen.generate(exp)
        outfile = tmp_path / "test_ensemble.nc"
        gen.save(ds, str(outfile), experiment=exp)
        loaded = xr.open_dataset(outfile)
        assert loaded.attrs.get("name") == exp.name


# ======================================================================
# 6. repr
# ======================================================================

class TestRepr:

    def test_repr_unfitted(self):
        gen = SMBGenerator()
        assert "fitted=False" in repr(gen)

    def test_repr_fitted(self, fitted_generator):
        gen, _ = fitted_generator
        r = repr(gen)
        assert "fitted=True" in r
        assert "n_obs=" in r
        assert "smb_mean=" in r


# ======================================================================
# Standalone debug runner
# ======================================================================

def run_debug():
    import pandas as pd

    print("=" * 60)
    print("SMBGenerator — step-by-step debug run")
    print("=" * 60)

    rng = np.random.default_rng(42)
    time = xr.cftime_range(start="1979", periods=540, freq="MS")
    months = np.array([d.month for d in time.values])
    t = np.arange(540)

    values = (0.04 + 5e-5 * t
              + 0.03 * np.sin(2 * np.pi * (months - 3) / 12)
              + rng.normal(0, 0.02, 540))
    smb = xr.DataArray(values, coords={"time": time}, dims=["time"], name="smb")

    # --- fit() ---
    print("\n[1] fit()...")
    gen = SMBGenerator(nperseg=30)
    gen.fit(smb)
    print(f"    {gen}")
    gen.summarize()

    # --- validate() ---
    print("\n[2] validate()...")
    results = gen.validate(verbose=True)

    # --- generate() — baseline ---
    print("\n[3] generate() — baseline (5 members, 100 years)...")
    exp = Experiment.baseline(n_years=100, n_members=5)
    ds = gen.generate(exp)
    print(f"    Dataset: {dict(ds.dims)}")
    print(f"    smb_syn mean: {float(ds['smb_syn'].mean()):.5f}  "
          f"(observed: {gen.smb_mean:.5f})")
    print(f"    smb_syn std:  {float(ds['smb_syn'].std()):.5f}  "
          f"(observed: {float(smb.std()):.5f})")
    print(f"    g_syn variance (per member): "
          f"{[float(ds['g_syn'].isel(member=i).var()) for i in range(5)]}")

    # --- generate_suite() ---
    print("\n[4] generate_suite() — standard suite (3 members, 10 years)...")
    suite = Experiment.standard_suite(n_years=10, n_members=3, seed=0)
    results_suite = gen.generate_suite(suite)
    print(f"    Experiments run: {list(results_suite.keys())}")
    print(f"\n    Mean preservation across suite:")
    print(f"    {'Experiment':<35}  {'mean':>8}  {'diff':>10}")
    for name, result_ds in results_suite.items():
        m = float(result_ds["smb_syn"].values.ravel().mean())
        print(f"    {name:<35}  {m:>8.5f}  {abs(m - gen.smb_mean):>10.2e}")

    # --- save() ---
    print("\n[5] save() and reload...")
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmppath = f.name
    gen.save(ds, tmppath, experiment=exp)
    loaded = xr.open_dataset(tmppath)
    print(f"    Loaded variables: {list(loaded.data_vars)}")
    print(f"    Loaded attrs: {dict(loaded.attrs)}")
    os.unlink(tmppath)

    print("\n" + "=" * 60)
    print("Debug run complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_debug()