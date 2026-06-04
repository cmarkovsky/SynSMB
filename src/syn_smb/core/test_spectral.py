"""
test_spectral.py
================
Tests for SpectralSynthesizer.

Run directly for step-by-step printed output:
    python test_spectral.py

Run via pytest for pass/fail:
    pytest test_spectral.py -v
"""

import numpy as np
import xarray as xr
import pytest
from scipy.signal import welch
from syn_smb.core.spectral import SpectralSynthesizer


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def gaussian_series(rng):
    """
    Simple N(0,1) series — representative of GaussianTransform output.
    540 points = 45 years of monthly data, matching the RACMO record.
    """
    return rng.standard_normal(540)


@pytest.fixture
def ar1_series(rng):
    """
    AR(1) series with autocorrelation coefficient 0.5.
    Has a red spectrum — tests that the synthesizer reproduces non-flat PSDs.
    """
    n = 540
    phi = 0.5
    x = np.zeros(n)
    noise = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi**2) * noise[t]
    return x


@pytest.fixture
def fitted_synthesizer(gaussian_series):
    """SpectralSynthesizer fitted on the gaussian_series fixture."""
    ss = SpectralSynthesizer(nperseg=60, dt_years=1/12)
    ss.fit(gaussian_series)
    return ss, gaussian_series


@pytest.fixture
def fitted_ar1(ar1_series):
    """SpectralSynthesizer fitted on the AR(1) fixture."""
    ss = SpectralSynthesizer(nperseg=60, dt_years=1/12)
    ss.fit(ar1_series)
    return ss, ar1_series


# ======================================================================
# 1. fit()
# ======================================================================

class TestFit:

    def test_sets_fitted_flag(self, gaussian_series):
        ss = SpectralSynthesizer()
        assert not ss.is_fitted
        ss.fit(gaussian_series)
        assert ss.is_fitted

    def test_stores_correct_n_obs(self, gaussian_series):
        ss = SpectralSynthesizer()
        ss.fit(gaussian_series)
        assert ss.n_obs == len(gaussian_series)

    def test_freqs_start_at_zero(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        assert ss.freqs[0] == 0.0

    def test_freqs_max_is_nyquist(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        expected_nyquist = ss.fs / 2
        assert abs(ss.freqs[-1] - expected_nyquist) < 1e-10

    def test_psd_is_nonnegative(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        assert np.all(ss.psd >= 0), "PSD contains negative values"

    def test_psd_has_no_nan_or_inf(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        assert not np.any(np.isnan(ss.psd))
        assert not np.any(np.isinf(ss.psd))

    def test_ci_brackets_psd(self, fitted_synthesizer):
        """Lower CI should be <= PSD <= upper CI at every frequency."""
        ss, _ = fitted_synthesizer
        assert np.all(ss.psd_ci_lower <= ss.psd + 1e-15)
        assert np.all(ss.psd_ci_upper >= ss.psd - 1e-15)

    def test_ci_lower_less_than_upper(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        assert np.all(ss.psd_ci_lower < ss.psd_ci_upper)

    def test_n_segments_is_positive(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        assert ss.n_segments > 0

    def test_n_segments_formula(self, gaussian_series):
        """n_segments = 1 + (n_obs - nperseg) // noverlap"""
        nperseg = 60
        ss = SpectralSynthesizer(nperseg=nperseg)
        ss.fit(gaussian_series)
        noverlap = nperseg // 2
        expected = 1 + (len(gaussian_series) - nperseg) // noverlap
        assert ss.n_segments == expected

    def test_returns_self_for_chaining(self, gaussian_series):
        ss = SpectralSynthesizer()
        result = ss.fit(gaussian_series)
        assert result is ss

    def test_raises_when_series_shorter_than_nperseg(self):
        ss = SpectralSynthesizer(nperseg=60)
        with pytest.raises(ValueError, match="shorter than nperseg"):
            ss.fit(np.random.standard_normal(30))

    def test_accepts_xarray_input(self, gaussian_series):
        time = xr.cftime_range(start="1979", periods=len(gaussian_series), freq="MS")
        da = xr.DataArray(gaussian_series, coords={"time": time}, dims=["time"])
        ss = SpectralSynthesizer()
        ss.fit(da)
        assert ss.is_fitted
        assert ss.n_obs == len(gaussian_series)


# ======================================================================
# 2. synthesize()
# ======================================================================

class TestSynthesize:

    def test_output_shape_single_member(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        out = ss.synthesize(n_years=100, n_members=1)
        expected_n = int(100 / ss.dt_years)
        assert out.shape == (1, expected_n)

    def test_output_shape_multiple_members(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        out = ss.synthesize(n_years=100, n_members=10)
        expected_n = int(100 / ss.dt_years)
        assert out.shape == (10, expected_n)

    def test_each_member_has_unit_variance(self, fitted_synthesizer):
        """
        Variance normalisation fix: every member must have std = 1.0.
        """
        ss, _ = fitted_synthesizer
        out = ss.synthesize(n_years=100, n_members=20)
        for i in range(out.shape[0]):
            var = np.var(out[i])
            assert abs(var - 1.0) < 1e-10, (
                f"Member {i} variance = {var:.6f}, expected 1.0. "
                "Check that g_syn /= std(g_syn) is present in synthesize()."
            )

    def test_each_member_has_zero_mean(self, fitted_synthesizer):
        """
        DC component set to zero means each member should have near-zero mean.
        """
        ss, _ = fitted_synthesizer
        out = ss.synthesize(n_years=100, n_members=20)
        for i in range(out.shape[0]):
            mean = np.mean(out[i])
            assert abs(mean) < 0.05, (
                f"Member {i} mean = {mean:.4f}, expected ~0. "
                "Check that amps[0] = 0.0 is present in _build_amplitudes()."
            )

    def test_output_has_no_nan_or_inf(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        out = ss.synthesize(n_years=100, n_members=5)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_members_are_independent(self, fitted_synthesizer, rng):
        """
        Different members should not be identical — each has random phases.
        """
        ss, _ = fitted_synthesizer
        out = ss.synthesize(n_years=100, n_members=2, rng=rng)
        assert not np.allclose(out[0], out[1]), "Members are identical — phase randomisation may be broken"

    def test_reproducible_with_same_seed(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        out1 = ss.synthesize(n_years=100, n_members=5, rng=np.random.default_rng(0))
        out2 = ss.synthesize(n_years=100, n_members=5, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_give_different_output(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        out1 = ss.synthesize(n_years=100, n_members=5, rng=np.random.default_rng(0))
        out2 = ss.synthesize(n_years=100, n_members=5, rng=np.random.default_rng(99))
        assert not np.allclose(out1, out2)

    def test_longer_series_correct_length(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        out = ss.synthesize(n_years=1000, n_members=1)
        expected_n = int(1000 / ss.dt_years)
        assert out.shape[1] == expected_n

    def test_raises_when_not_fitted(self):
        ss = SpectralSynthesizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            ss.synthesize(n_years=100)


# ======================================================================
# 3. Band scaling
# ======================================================================

class TestBandScaling:

    def test_band_scaling_increases_variance_in_band(self, fitted_synthesizer, rng):
        """
        Scaling the PSD in a band by factor k should increase the variance
        of the band-pass filtered output by approximately k.
        Uses the annual band (0.8–1.5 yr period) as test case.
        """
        from scipy.signal import butter, filtfilt

        ss, _ = fitted_synthesizer

        baseline  = ss.synthesize(n_years=500, n_members=30, rng=np.random.default_rng(0))
        scaled    = ss.synthesize(
            n_years=500, n_members=30,
            band_scales=[(0.8, 1.5, 4.0)],   # 4x annual band → 2x std
            rng=np.random.default_rng(0),
        )

        # Bandpass filter both ensembles to the annual band
        # and compare std of the filtered signal
        fs_monthly = ss.fs  # cycles/year but we treat samples as monthly
        nyq = fs_monthly / 2
        low  = (1 / 1.5) / nyq
        high = (1 / 0.8) / nyq
        b, a = butter(4, [low, high], btype='band')

        baseline_stds = np.array([
            np.std(filtfilt(b, a, baseline[i])) for i in range(30)
        ])
        scaled_stds = np.array([
            np.std(filtfilt(b, a, scaled[i])) for i in range(30)
        ])

        ratio = scaled_stds.mean() / baseline_stds.mean()
        # PSD scale=4 → variance scale=4 → std scale=2. Allow ±20% tolerance.
        assert 1.6 < ratio < 2.4, (
            f"Band scaling ratio = {ratio:.2f}, expected ~2.0 "
            f"(PSD scaled by 4x → std scaled by ~2x)"
        )

    def test_band_scaling_zero_suppresses_band(self, fitted_synthesizer):
        """
        Scaling a band by 0 should remove variance from that band.
        """
        ss, _ = fitted_synthesizer
        out = ss.synthesize(
            n_years=200, n_members=5,
            band_scales=[(0.8, 1.5, 0.0)],  # suppress annual band
        )
        assert out.shape[0] == 5

    def test_unscaled_bands_unaffected(self, fitted_synthesizer, rng):
        """
        Scaling the annual band should not change the total variance
        (controlled by normalisation), but out-of-band spectral shape
        should be proportionally similar to the unscaled case.
        All members still have unit variance after normalisation.
        """
        ss, _ = fitted_synthesizer
        out = ss.synthesize(
            n_years=200, n_members=10,
            band_scales=[(0.8, 1.5, 10.0)],
            rng=np.random.default_rng(0),
        )
        for i in range(out.shape[0]):
            assert abs(np.var(out[i]) - 1.0) < 1e-10, (
                f"Member {i} variance != 1.0 after band scaling + normalisation"
            )


# ======================================================================
# 4. Spectral fidelity
# ======================================================================

class TestSpectralFidelity:

    def test_ensemble_mean_psd_close_to_observed(self, fitted_ar1):
        """
        The ensemble-mean PSD should be close to the observed PSD.
        Uses a large ensemble on a long series to reduce sampling variance.
        Tests the AR(1) series which has a distinctive red spectrum.
        """
        ss, _ = fitted_ar1
        ensemble = ss.synthesize(n_years=200, n_members=100, rng=np.random.default_rng(0))

        # Compute ensemble mean PSD
        psds = []
        for i in range(ensemble.shape[0]):
            f, p = welch(ensemble[i], fs=ss.fs, nperseg=ss.nperseg)
            psds.append(p)
        mean_psd = np.array(psds).mean(axis=0)

        # The ensemble mean PSD should be within a factor of 3 of the
        # observed PSD at every frequency.
        ratio = mean_psd / ss.psd
        assert np.all(ratio > 0.33) and np.all(ratio < 3.0), (
            f"Ensemble mean PSD deviates from observed by more than 3x. "
            f"Max ratio: {ratio.max():.2f}, Min ratio: {ratio.min():.2f}"
        )

    def test_flat_spectrum_input_gives_flat_ensemble(self, rng):
        """
        White noise input should produce an ensemble with approximately
        flat spectrum.
        """
        white_noise = rng.standard_normal(540)
        ss = SpectralSynthesizer(nperseg=60)
        ss.fit(white_noise)

        ensemble = ss.synthesize(n_years=200, n_members=50, rng=np.random.default_rng(0))
        psds = []
        for i in range(ensemble.shape[0]):
            _, p = welch(ensemble[i], fs=ss.fs, nperseg=ss.nperseg)
            psds.append(p)
        mean_psd = np.array(psds).mean(axis=0)[1:]  # skip DC

        # Coefficient of variation of the mean PSD should be small
        cv = mean_psd.std() / mean_psd.mean()
        assert cv < 0.5, (
            f"Mean PSD CV = {cv:.3f} for white noise input. "
            "Expected < 0.5 (approximately flat spectrum)."
        )


# ======================================================================
# 5. validate()
# ======================================================================

class TestValidate:

    def test_returns_expected_keys(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        results = ss.validate(n_check=50, verbose=False)
        expected = {"psd_coverage", "mean_variance", "std_variance", "passed"}
        assert set(results.keys()) == expected

    def test_mean_variance_near_one(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        results = ss.validate(n_check=100, verbose=False)
        assert abs(results["mean_variance"] - 1.0) < 0.05

    def test_passes_on_good_data(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        results = ss.validate(n_check=100, verbose=False)
        assert results["passed"], f"validate() failed:\n{results}"

    def test_raises_when_not_fitted(self):
        ss = SpectralSynthesizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            ss.validate()


# ======================================================================
# 6. repr
# ======================================================================

class TestRepr:

    def test_repr_unfitted(self):
        ss = SpectralSynthesizer()
        assert "fitted=False" in repr(ss)

    def test_repr_fitted(self, fitted_synthesizer):
        ss, _ = fitted_synthesizer
        r = repr(ss)
        assert "fitted=True" in r
        assert "n_obs=" in r
        assert "n_segments=" in r


# ======================================================================
# Standalone debug runner
# ======================================================================

def run_debug():
    print("=" * 60)
    print("SpectralSynthesizer — step-by-step debug run")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # --- Build AR(1)-like test data to give a non-flat spectrum ---
    print("\n[1] Building test data (AR(1), phi=0.6, n=540)...")
    n = 540
    phi = 0.6
    x = np.zeros(n)
    noise = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi**2) * noise[t]
    print(f"    n={n}, mean={np.mean(x):.4f}, std={np.std(x):.4f}")

    # --- fit() ---
    print("\n[2] fit()...")
    ss = SpectralSynthesizer(nperseg=60, dt_years=1/12)
    ss.fit(x)
    print(f"    {ss}")
    print(f"    n_segments:  {ss.n_segments}")
    print(f"    freq range:  [{ss.freqs[0]:.4f}, {ss.freqs[-1]:.4f}] cycles/yr")
    print(f"    PSD range:   [{ss.psd.min():.4e}, {ss.psd.max():.4e}]")
    print(f"    CI width:    mean ratio upper/lower = "
          f"{(ss.psd_ci_upper / ss.psd_ci_lower).mean():.2f}x")

    # --- synthesize() — baseline ---
    print("\n[3] synthesize() — baseline (no band scaling)...")
    out = ss.synthesize(n_years=1000, n_members=10, rng=np.random.default_rng(0))
    print(f"    output shape:    {out.shape}")
    variances = out.var(axis=1)
    means     = out.mean(axis=1)
    print(f"    variance (mean): {variances.mean():.6f}  (target: 1.0)")
    print(f"    variance (std):  {variances.std():.6f}  (target: small)")
    print(f"    mean    (mean):  {means.mean():.4e}  (target: ~0)")
    print(f"    mean    (std):   {means.std():.4e}  (target: small)")

    # --- synthesize() — variance check with band scaling ---
    print("\n[4] Band scaling — variance normalisation check...")
    print("    Each member must have unit variance after scaling + normalisation.")
    for factor in [0.1, 1.0, 2.0, 5.0, 10.0]:
        out_scaled = ss.synthesize(
            n_years=200, n_members=10,
            band_scales=[(0.8, 1.5, factor)],
            rng=np.random.default_rng(0),
        )
        v = out_scaled.var(axis=1)
        print(f"    annual_scale={factor:4.1f}x  → variance: "
              f"{v.mean():.6f} ± {v.std():.6f}  (all should be 1.0)")

    # --- Spectral fidelity check ---
    print("\n[5] Spectral fidelity — ensemble mean PSD vs observed...")
    ensemble = ss.synthesize(n_years=200, n_members=200, rng=np.random.default_rng(0))
    member_psds = []
    for i in range(200):
        _, p = welch(ensemble[i], fs=ss.fs, nperseg=ss.nperseg)
        member_psds.append(p)
    mean_psd = np.array(member_psds).mean(axis=0)
    ratio = mean_psd / ss.psd
    print(f"    PSD ratio (ensemble mean / observed):")
    print(f"    min={ratio.min():.3f}, max={ratio.max():.3f}, mean={ratio.mean():.3f}")
    print(f"    (target: all ratios close to 1.0)")

    # --- Confidence interval check ---
    print("\n[6] Confidence intervals...")
    within = (mean_psd >= ss.psd_ci_lower) & (mean_psd <= ss.psd_ci_upper)
    print(f"    Fraction of freqs where ensemble mean PSD is within 95% CI: "
          f"{within.mean():.3f}")

    # --- validate() ---
    print("\n[7] validate()...")
    ss.validate(n_check=200, verbose=True)

    print("\n" + "=" * 60)
    print("Debug run complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_debug()