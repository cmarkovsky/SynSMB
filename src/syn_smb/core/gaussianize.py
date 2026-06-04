"""
gaussianize.py
==============
Rank-based Gaussian transform (normal quantile transform).

Provides a fully invertible mapping between an empirically distributed
1D time series and a standard normal N(0,1) series. Designed as Step 1
of the synthetic SMB generation pipeline.

The inverse transform enforces zero mean on the back-transformed result.
The observed SMB mean must be added back externally, exactly once, in
SMBGenerator.generate(). It is never touched here.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import norm, ks_1samp
from typing import Union

ArrayLike = Union[np.ndarray, xr.DataArray]


class GaussianTransform:
    """
    Rank-based Gaussian transform (normal quantile transform).

    Maps a 1D time series from its empirical distribution to N(0,1) and
    back, preserving rank structure. The empirical inverse CDF is learned
    at fit() time and reused for all subsequent inverse transforms.

    Parameters
    ----------
    None

    Attributes
    ----------
    x_sorted : np.ndarray
        Training data sorted ascending. Serves as the empirical inverse CDF.
    quantiles : np.ndarray
        Hazen plotting positions corresponding to x_sorted.
        Defined as (i - 0.5) / n for i = 1..n, strictly in (0, 1).
    n : int
        Number of training observations.
    is_fitted : bool
        Whether fit() has been called.

    Examples
    --------
    >>> gt = GaussianTransform()
    >>> gt.fit(residuals)
    >>> g = gt.transform(residuals)          # → N(0,1)
    >>> r = gt.inverse_transform(g_syn)      # → zero-mean physical residuals
    >>> gt.validate(residuals)               # round-trip and normality checks
    """

    def __init__(self) -> None:
        self.x_sorted: np.ndarray | None = None
        self.quantiles: np.ndarray | None = None
        self.n: int | None = None
        self.is_fitted: bool = False

        # Semi-parametric tail attributes — populated by fit()
        self.tail_loc: float | None = None      # fitted normal mean
        self.tail_scale: float | None = None    # fitted normal std
        self.lower_tail_offset: float | None = None  # continuity correction at lower splice
        self.upper_tail_offset: float | None = None  # continuity correction at upper splice

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "GaussianTransform is not fitted. Call fit() before "
                "transform() or inverse_transform()."
            )

    @staticmethod
    def _extract_values(data: ArrayLike) -> np.ndarray:
        """Extract a flat numpy array from either numpy or xarray input."""
        if isinstance(data, xr.DataArray):
            return data.values.ravel()
        return np.asarray(data).ravel()

    @staticmethod
    def _wrap_output(result: np.ndarray, template: ArrayLike) -> ArrayLike:
        """
        Return result in the same type as the original input.
        If input was an xarray DataArray, preserve coordinates and dims.
        """
        if isinstance(template, xr.DataArray):
            return xr.DataArray(
                result,
                coords=template.coords,
                dims=template.dims,
                name=template.name,
            )
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: ArrayLike) -> GaussianTransform:
        """
        Learn the empirical distribution from observed residuals.

        Stores the sorted training values and their Hazen plotting positions.
        These together define the empirical inverse CDF used in
        inverse_transform().

        Parameters
        ----------
        data : array-like, shape (n,)
            Detrended, deseasoned SMB residuals in physical units (m w.e. a⁻¹).
            Should already have trend and seasonal cycle removed — the
            Preprocessor handles this upstream.

        Returns
        -------
        self : GaussianTransform
            Returns self to allow method chaining: gt.fit(x).transform(x).
        """
        x = self._extract_values(data)

        if x.ndim != 1:
            raise ValueError(
                f"Expected 1D input after flattening, got shape {x.shape}. "
                "GaussianTransform operates on a single time series."
            )
        if len(x) < 10:
            raise ValueError(
                f"Too few observations ({len(x)}) to fit a reliable "
                "empirical distribution. Minimum is 10."
            )

        self.n = len(x)

        # Hazen plotting positions: (i - 0.5) / n for i in 1..n
        # Strictly in (0, 1) — avoids 0 and 1 which map to ±inf via norm.ppf.
        # This is the same formula used in transform(), ensuring consistency
        # between the forward and inverse mappings.
        self.quantiles = (np.arange(1, self.n + 1) - 0.5) / self.n

        # Sort the training data — this is the empirical inverse CDF.
        # x_sorted[i] is the value at quantile quantiles[i].
        self.x_sorted = np.sort(x)

        # --- Semi-parametric tail fitting ---
        # Fit a normal distribution to the full residual dataset.
        # Used only for extrapolation beyond the observed range — the
        # empirical CDF is used for all values within the training range.
        #
        # The splice point is at the exact empirical boundaries:
        #   lower: quantiles[0]  = 0.5 / n  (smallest observed quantile)
        #   upper: quantiles[-1] = (n-0.5)/n (largest observed quantile)
        #
        # Continuity offsets ensure Q_semi is continuous at the splice:
        #   offset = x_boundary - Q_normal(p_boundary)
        # so Q_semi(p_boundary) = Q_normal(p_boundary) + offset = x_boundary.
        self.tail_loc, self.tail_scale = norm.fit(x)

        p_low  = float(self.quantiles[0])
        p_high = float(self.quantiles[-1])

        self.lower_tail_offset = float(
            self.x_sorted[0] - norm.ppf(p_low, loc=self.tail_loc, scale=self.tail_scale)
        )
        self.upper_tail_offset = float(
            self.x_sorted[-1] - norm.ppf(p_high, loc=self.tail_loc, scale=self.tail_scale)
        )

        self.is_fitted = True
        return self

    def transform(self, data: ArrayLike) -> ArrayLike:
        """
        Forward transform: physical residuals → N(0, 1).

        Maps each value to its empirical quantile via rank transform, then
        converts to the corresponding standard normal quantile.

        Parameters
        ----------
        data : array-like, shape (n,)
            SMB residuals in physical units. Should come from the same
            distribution as the training data passed to fit().

        Returns
        -------
        g : same type as input, shape (n,)
            Gaussianized residuals approximately ~ N(0, 1).

        Notes
        -----
        If len(data) != self.n, Hazen positions are recomputed for the new
        length. This is correct when transforming data of a different size
        but drawn from the same distribution (e.g. a validation split).

        Tied values in the input receive consecutive ranks. For SMB data
        this is unlikely to occur but worth noting.
        """
        self._check_fitted()

        x = self._extract_values(data)
        n = len(x)

        # Recompute plotting positions if input length differs from training.
        # This handles the validation split case without error.
        u_positions = (np.arange(1, n + 1) - 0.5) / n

        # Double argsort: first argsort gives sorted indices,
        # second argsort maps each element back to its rank (0-indexed).
        # Adding 1 gives 1-indexed ranks in [1, n].
        ranks = np.argsort(np.argsort(x)) + 1

        # Map ranks to (0, 1) via Hazen plotting positions
        u = (ranks - 0.5) / n

        # Map uniform quantiles to standard normal via inverse CDF
        g = norm.ppf(u)

        return self._wrap_output(g, data)

    def inverse_transform(self, g: ArrayLike) -> ArrayLike:
        """
        Inverse transform: N(0, 1) → zero-mean physical residuals.

        Uses a semi-parametric approach:
          - Within the observed range [quantiles[0], quantiles[-1]]:
            empirical inverse CDF via linear interpolation (identical to
            the original approach — round-trip is exact for training data)
          - Beyond the observed range:
            fitted normal distribution + continuity offset, ensuring no
            jump at the splice point and allowing new extreme values

        This replaces the original purely empirical approach, which clipped
        all values to the observed min/max and caused variance saturation
        at high band scale factors (e.g. 5x, 10x).

        Parameters
        ----------
        g : array-like, shape (n,)
            Synthetic Gaussian series ~ N(0, 1). Typically the output of
            SpectralSynthesizer.synthesize().

        Returns
        -------
        r : same type as input, shape (n,)
            Synthetic SMB residuals in physical units (m w.e. a⁻¹).
            Zero mean enforced — the observed SMB mean is added back
            externally in SMBGenerator.generate().

        Notes
        -----
        Splice continuity: lower_tail_offset and upper_tail_offset are
        computed at fit() time so that the parametric tail exactly equals
        the empirical value at the boundary. Q_semi is continuous at both
        splice points by construction.

        Mean shift fix: the final mean subtraction ensures band-scaling
        experiments isolate variance structure rather than shifting the
        mean SMB forcing passed to Icepack.
        """
        self._check_fitted()

        g_vals = self._extract_values(g)

        # Step 1: N(0,1) → Uniform(0,1)
        # Clip to a safe range to avoid norm.ppf returning ±inf when
        # scaled Gaussian values are extreme (e.g. g*10 → norm.cdf = 1.0 exactly)
        u_syn = np.clip(norm.cdf(g_vals), 1e-15, 1 - 1e-15)

        p_low  = float(self.quantiles[0])
        p_high = float(self.quantiles[-1])

        # Step 2: semi-parametric inverse CDF
        #
        # np.where evaluates all branches before selecting — norm.ppf is
        # called on all u values. This is safe since norm.cdf returns
        # strictly (0,1) for all finite inputs.
        lower_tail = (norm.ppf(u_syn, loc=self.tail_loc, scale=self.tail_scale)
                      + self.lower_tail_offset)
        upper_tail = (norm.ppf(u_syn, loc=self.tail_loc, scale=self.tail_scale)
                      + self.upper_tail_offset)
        bulk       = np.interp(u_syn, self.quantiles, self.x_sorted)

        r_syn = np.where(u_syn < p_low,  lower_tail,
                np.where(u_syn > p_high, upper_tail, bulk))

        # Step 3: enforce zero mean (mean shift fix)
        r_syn = r_syn - np.mean(r_syn)

        return self._wrap_output(r_syn, g)

    def validate(
        self,
        data: ArrayLike,
        roundtrip_tolerance: float = 1e-6,
        ks_alpha: float = 0.05,
        verbose: bool = True,
    ) -> dict:
        """
        Sanity checks on fit quality and round-trip fidelity.

        Runs two checks:
        1. The output of transform() is approximately N(0,1), verified via
           a one-sample Kolmogorov-Smirnov test against the standard normal.
        2. inverse_transform(transform(data)) recovers the original centered
           data to numerical precision (round-trip fidelity).

        Call this after fit() before using the transform in the pipeline.
        A failure here means every downstream result is unreliable.

        Parameters
        ----------
        data : array-like
            The same data passed to fit().
        roundtrip_tolerance : float
            Maximum allowed pointwise absolute error for round-trip check.
            Default 1e-6 is achievable for float64 given the norm.ppf →
            norm.cdf composition.
        ks_alpha : float
            Significance level for the KS normality test. Default 0.05.
        verbose : bool
            If True, print a formatted summary to stdout.

        Returns
        -------
        results : dict
            Keys:
                ks_statistic      : float  — KS test statistic
                ks_pvalue         : float  — KS test p-value
                transform_mean    : float  — mean of transformed output (target: ~0)
                transform_std     : float  — std of transformed output (target: ~1)
                roundtrip_max_err : float  — max pointwise round-trip error
                passed            : bool   — True if all checks pass
        """
        self._check_fitted()

        x = self._extract_values(data)

        # --- Check 1: transform output is approximately N(0,1) ---
        g = self._extract_values(self.transform(data))

        ks_stat, ks_pvalue = ks_1samp(g, norm.cdf)
        transform_mean = float(np.mean(g))
        transform_std = float(np.std(g))

        # --- Check 2: round-trip fidelity ---
        # inverse_transform enforces zero mean, so compare against x_centered
        r_recovered = self._extract_values(self.inverse_transform(g))
        x_centered = x - np.mean(x)
        roundtrip_max_err = float(np.max(np.abs(r_recovered - x_centered)))

        # --- Check 3: tail continuity ---
        # At the splice points the semi-parametric function must be continuous.
        # Evaluate both branches at the boundary and confirm they agree.
        p_low  = float(self.quantiles[0])
        p_high = float(self.quantiles[-1])

        empirical_low  = float(self.x_sorted[0])
        empirical_high = float(self.x_sorted[-1])
        parametric_low  = float(norm.ppf(p_low,  loc=self.tail_loc, scale=self.tail_scale) + self.lower_tail_offset)
        parametric_high = float(norm.ppf(p_high, loc=self.tail_loc, scale=self.tail_scale) + self.upper_tail_offset)

        lower_splice_err = abs(empirical_low  - parametric_low)
        upper_splice_err = abs(empirical_high - parametric_high)

        passed = (
            ks_pvalue > ks_alpha
            and roundtrip_max_err < roundtrip_tolerance
            and abs(transform_mean) < 0.05
            and abs(transform_std - 1.0) < 0.05
            and lower_splice_err < 1e-10
            and upper_splice_err < 1e-10
        )

        results = {
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "transform_mean": transform_mean,
            "transform_std": transform_std,
            "roundtrip_max_err": roundtrip_max_err,
            "lower_splice_err": lower_splice_err,
            "upper_splice_err": upper_splice_err,
            "passed": passed,
        }

        if verbose:
            status = "PASSED ✓" if passed else "FAILED ✗"
            print(f"GaussianTransform validation — {status}")
            print(f"  KS statistic:        {ks_stat:.4f}")
            print(f"  KS p-value:          {ks_pvalue:.4f}  (threshold: {ks_alpha})")
            print(f"  Transform mean:      {transform_mean:+.4f}  (target: ~0)")
            print(f"  Transform std:       {transform_std:.4f}   (target: ~1)")
            print(f"  Round-trip error:    {roundtrip_max_err:.2e}  (tolerance: {roundtrip_tolerance:.0e})")
            print(f"  Lower splice error:  {lower_splice_err:.2e}  (target: ~0)")
            print(f"  Upper splice error:  {upper_splice_err:.2e}  (target: ~0)")

        return results

    def __repr__(self) -> str:
        if self.is_fitted:
            return (
                f"GaussianTransform(fitted=True, n={self.n}, "
                f"x_range=[{self.x_sorted[0]:.4f}, {self.x_sorted[-1]:.4f}], "
                f"tail=N({self.tail_loc:.4f}, {self.tail_scale:.4f}))"
            )
        return "GaussianTransform(fitted=False)"