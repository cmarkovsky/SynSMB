"""
spectral.py
===========
Fourier phase-randomisation synthesizer for Gaussianized residuals.

Operates on a unit-variance N(0,1) series (the output of GaussianTransform).
Estimates the power spectral density of the observed series and generates
an ensemble of synthetic realisations with the same spectral structure but
randomised Fourier phases.

This is Step 2 of the synthetic SMB generation pipeline:

    GaussianTransform.transform()  →  SpectralSynthesizer.fit()
                                       SpectralSynthesizer.synthesize()
                                   →  GaussianTransform.inverse_transform()
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import welch
from scipy.stats import chi2
from typing import Union

ArrayLike = Union[np.ndarray, xr.DataArray]


class SpectralSynthesizer:
    """
    Fourier phase-randomisation synthesizer.

    Fits a spectral model to a single Gaussianized time series and generates
    an ensemble of synthetic realisations with the same power spectral density
    but independent, uniformly randomised Fourier phases.

    Parameters
    ----------
    nperseg : int
        Segment length for Welch PSD estimation. Default 60 gives ~5-year
        segments for monthly data. Larger values give finer frequency
        resolution at the cost of fewer segments and wider confidence
        intervals.
    dt_years : float
        Sampling interval in years. Default 1/12 for monthly data.

    Attributes
    ----------
    freqs : np.ndarray
        Frequency grid in cycles/year, from Welch estimation.
    psd : np.ndarray
        Estimated one-sided PSD of the training series.
    psd_ci_lower : np.ndarray
        Lower bound of the (1-ci_alpha) confidence interval on the PSD.
    psd_ci_upper : np.ndarray
        Upper bound of the (1-ci_alpha) confidence interval on the PSD.
    n_obs : int
        Number of observations in the training series.
    n_segments : int
        Number of Welch segments used to estimate the PSD.
    is_fitted : bool
        Whether fit() has been called.

    Examples
    --------
    >>> ss = SpectralSynthesizer(nperseg=60, dt_years=1/12)
    >>> ss.fit(g_resid)                          # g_resid ~ N(0,1)
    >>> ensemble = ss.synthesize(n_years=1000, n_members=30)
    >>> ss.validate()
    """

    def __init__(self, nperseg: int = 60, dt_years: float = 1 / 12) -> None:
        self.nperseg = nperseg
        self.dt_years = dt_years
        self.fs = 1.0 / dt_years  # sampling frequency in cycles/year

        # Fitted attributes
        self.freqs: np.ndarray | None = None
        self.psd: np.ndarray | None = None
        self.psd_ci_lower: np.ndarray | None = None
        self.psd_ci_upper: np.ndarray | None = None
        self.n_obs: int | None = None
        self.n_segments: int | None = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "SpectralSynthesizer is not fitted. Call fit() before synthesize()."
            )

    @staticmethod
    def _extract_values(data: ArrayLike) -> np.ndarray:
        if isinstance(data, xr.DataArray):
            return data.values.ravel()
        return np.asarray(data).ravel()

    def _interpolate_psd(
        self, freqs_syn: np.ndarray, band_scales: list[tuple[float, float, float]] | None
    ) -> np.ndarray:
        """
        Interpolate the fitted PSD onto a new frequency grid and optionally
        apply band-specific scaling factors.

        Parameters
        ----------
        freqs_syn : array
            Target frequency grid (rfftfreq of the synthetic series).
        band_scales : list of (pmin, pmax, factor) or None
            Each tuple specifies a period band in years and a multiplicative
            factor. pmin is the shortest period (highest frequency) in the
            band; pmax is the longest period (lowest frequency).

        Returns
        -------
        psd_syn : array
            Interpolated (and optionally scaled) PSD on the synthetic grid.
        """
        psd_syn = np.interp(freqs_syn, self.freqs, self.psd)

        if band_scales is not None:
            for pmin, pmax, factor in band_scales:
                fmin = 1.0 / pmax   # lowest frequency in band
                fmax = 1.0 / pmin   # highest frequency in band
                mask = (freqs_syn >= fmin) & (freqs_syn <= fmax)
                psd_syn[mask] *= factor

        return psd_syn

    @staticmethod
    def _build_amplitudes(psd_syn: np.ndarray, N_syn: int, fs: float) -> np.ndarray:
        """
        Convert a one-sided PSD to Fourier amplitudes via Parseval's theorem.

        For interior frequencies (k = 1,...,N/2-1):
            |X[k]| = sqrt(P[k] * N * fs / 2)

        The factor of 2 arises because scipy.signal.welch doubles the
        two-sided PSD to produce the one-sided estimate. The irfft
        normalization then ensures var(x) = sum(P[k] * df), matching
        Parseval's theorem exactly for a rectangular window with one segment.

        The DC bin is set to zero to enforce a zero-mean output.
        """
        amps = np.sqrt(psd_syn * N_syn * fs / 2.0)
        amps[0] = 0.0   # DC = 0 → zero mean
        return amps

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: ArrayLike, ci_alpha: float = 0.05) -> SpectralSynthesizer:
        """
        Estimate the PSD of the Gaussianized residuals.

        Uses Welch's method with 50% segment overlap. Computes chi-squared
        confidence intervals based on the effective number of segments.

        Parameters
        ----------
        data : array-like, shape (n,)
            Gaussianized residuals ~ N(0,1). Must be the output of
            GaussianTransform.transform() — do not pass raw SMB residuals.
        ci_alpha : float
            Significance level for confidence intervals. Default 0.05
            produces 95% CIs.

        Returns
        -------
        self : SpectralSynthesizer
            Returns self for method chaining.

        Notes
        -----
        Segment length: nperseg=60 gives segments of 5 years for monthly
        data. With N=540 observations and 50% overlap, this produces
        approximately 17 effective segments and degrees of freedom ~34.
        The resulting 95% CI spans roughly a factor of 2.5 around the
        PSD estimate at each frequency, which is wide at low frequencies
        where the spectral estimate is least reliable.
        """
        x = self._extract_values(data)

        if len(x) < self.nperseg:
            raise ValueError(
                f"Series length ({len(x)}) is shorter than nperseg ({self.nperseg}). "
                "Reduce nperseg or provide a longer series."
            )

        self.n_obs = len(x)

        # --- Welch PSD estimate ---
        noverlap = self.nperseg // 2
        freqs, psd = welch(x, fs=self.fs, nperseg=self.nperseg, noverlap=noverlap)
        self.freqs = freqs
        self.psd = psd

        # --- Chi-squared confidence intervals ---
        # scipy.signal.welch uses a Hann window by default, which reduces
        # the effective degrees of freedom relative to a rectangular window.
        # The number of effective segments with 50% overlap and Hann window
        # is approximately 1.5 * (N - nperseg) / (nperseg/2) + 1 ≈ 1.5 * K_rect.
        # We use the rectangular approximation here (K_rect segments) which
        # gives slightly conservative (wider) CIs — appropriate for a
        # methods-section claim about spectral uncertainty.
        self.n_segments = 1 + (self.n_obs - self.nperseg) // noverlap
        dof = 2 * self.n_segments

        self.psd_ci_lower = psd * dof / chi2.ppf(1 - ci_alpha / 2, dof)
        self.psd_ci_upper = psd * dof / chi2.ppf(ci_alpha / 2, dof)

        self.is_fitted = True
        return self

    def synthesize(
        self,
        n_years: int,
        n_members: int = 1,
        band_scales: list[tuple[float, float, float]] | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Generate an ensemble of synthetic Gaussianized time series.

        Each member shares the same amplitude spectrum (derived from the
        fitted PSD) but has independently randomised Fourier phases,
        producing statistically independent realisations.

        Parameters
        ----------
        n_years : int
            Length of each synthetic series in years.
        n_members : int
            Number of ensemble members. Default 1.
        band_scales : list of (pmin, pmax, factor) or None
            Optional PSD scaling within specified period bands. Each tuple
            specifies the band as (shortest_period, longest_period, factor)
            in years. The PSD is multiplied by factor within
            [1/pmax, 1/pmin] cycles/year before amplitude computation.
            This scales the variance in that band by the same factor.
        rng : np.random.Generator or None
            Random number generator for reproducibility. If None, a new
            default generator is created.

        Returns
        -------
        ensemble : np.ndarray, shape (n_members, N_syn)
            Synthetic Gaussianized time series. Each row is one member.
            Each member has mean ~0 (DC=0 by construction) and std ~1
            (enforced by post-generation normalisation).

        Notes
        -----
        Variance normalisation: after irfft, each member is divided by its
        own standard deviation to enforce exact unit variance. This corrects
        for any variance drift introduced by PSD interpolation onto the
        synthetic frequency grid. It does not alter the spectral shape —
        only the absolute scale.

        Phase constraints: the DC phase is fixed at zero (zero mean). If
        N_syn is even, the Nyquist phase is also fixed at zero to ensure
        a real-valued output.
        """
        self._check_fitted()

        if rng is None:
            rng = np.random.default_rng()

        N_syn = int(n_years / self.dt_years)
        freqs_syn = np.fft.rfftfreq(N_syn, d=self.dt_years)
        n_freqs = freqs_syn.size

        # Interpolate PSD onto synthetic grid and apply band scaling
        psd_syn = self._interpolate_psd(freqs_syn, band_scales)

        # Convert PSD to amplitudes
        amps = self._build_amplitudes(psd_syn, N_syn, self.fs)

        ensemble = np.empty((n_members, N_syn))

        for i in range(n_members):
            # Draw random phases uniformly in [-π, π]
            phases = rng.uniform(-np.pi, np.pi, size=n_freqs)

            # Pin DC and Nyquist phases to zero for a real-valued output
            phases[0] = 0.0
            if N_syn % 2 == 0:
                phases[-1] = 0.0

            # Build complex spectrum and transform to time domain
            Z = amps * np.exp(1j * phases)
            g_syn = np.fft.irfft(Z, n=N_syn)

            # Normalise to unit variance
            # This is the amplitude normalisation fix: corrects any variance
            # drift from PSD interpolation while preserving spectral shape.
            std = np.std(g_syn)
            if std < 1e-10:
                raise RuntimeError(
                    f"Member {i}: near-zero standard deviation ({std:.2e}) after irfft. "
                    "Check that the PSD is non-zero across the frequency range."
                )
            ensemble[i] = g_syn / std

        return ensemble

    def validate(
        self,
        n_check: int = 300,
        verbose: bool = True,
    ) -> dict:
        """
        Verify that the synthesizer produces output consistent with the
        fitted PSD.

        Generates a check ensemble, computes the ensemble-mean PSD, and
        reports the fraction of frequencies at which the mean PSD falls
        within the fitted confidence intervals.

        Parameters
        ----------
        n_check : int
            Number of members to generate for the check. Default 300.
            More members give a more stable mean PSD estimate.
        verbose : bool
            If True, print a formatted summary.

        Returns
        -------
        results : dict
            Keys:
                psd_coverage     : float — fraction of freqs where ensemble
                                   mean PSD falls within CI (target: ~1-ci_alpha)
                mean_variance    : float — mean variance across check members
                                   (target: ~1.0)
                std_variance     : float — std of variance across members
                                   (target: small, ~0.05)
                passed           : bool
        """
        self._check_fitted()

        rng = np.random.default_rng(0)
        ensemble = self.synthesize(
            n_years=max(100, self.n_obs * self.dt_years * 2),
            n_members=n_check,
            rng=rng,
        )

        # Ensemble-mean PSD
        member_psds = []
        for i in range(n_check):
            _, p = welch(ensemble[i], fs=self.fs, nperseg=self.nperseg)
            member_psds.append(p)

        member_psds = np.array(member_psds)
        mean_psd = member_psds.mean(axis=0)

        # Fraction of frequencies where mean PSD is within the fitted CI
        within = (mean_psd >= self.psd_ci_lower) & (mean_psd <= self.psd_ci_upper)
        psd_coverage = float(within.mean())

        # Variance of each member (should be ~1.0 after normalisation)
        variances = ensemble.var(axis=1)
        mean_variance = float(variances.mean())
        std_variance = float(variances.std())

        passed = (
            psd_coverage > 0.5        # majority of freqs within CI
            and abs(mean_variance - 1.0) < 0.05
            and std_variance < 0.1
        )

        results = {
            "psd_coverage": psd_coverage,
            "mean_variance": mean_variance,
            "std_variance": std_variance,
            "passed": passed,
        }

        if verbose:
            status = "PASSED ✓" if passed else "FAILED ✗"
            print(f"SpectralSynthesizer validation — {status}")
            print(f"  PSD coverage:   {psd_coverage:.3f}  (fraction of freqs within CI)")
            print(f"  Mean variance:  {mean_variance:.4f}  (target: ~1.0)")
            print(f"  Std variance:   {std_variance:.4f}  (target: small)")

        return results

    def __repr__(self) -> str:
        if self.is_fitted:
            return (
                f"SpectralSynthesizer(fitted=True, n_obs={self.n_obs}, "
                f"nperseg={self.nperseg}, n_segments={self.n_segments}, "
                f"dt_years={self.dt_years})"
            )
        return f"SpectralSynthesizer(fitted=False, nperseg={self.nperseg}, dt_years={self.dt_years})"