"""
preprocessor.py
===============
Removes deterministic components (trend, seasonal cycle) from an SMB
time series, leaving stochastic residuals suitable for Gaussian
transformation and spectral synthesis.

This is Step 3 of the synthetic SMB generation pipeline:

    SMBDataLoader.load()
        ↓
    Preprocessor.fit_transform()   ← removes trend + seasonal cycle
        ↓
    GaussianTransform.transform()
        ↓
    SpectralSynthesizer.synthesize()
        ↓
    GaussianTransform.inverse_transform()
        ↓
    Preprocessor.inverse_transform()  ← adds seasonal cycle back
        ↓
    SMBGenerator.generate()           ← adds SMB mean back

Design note: Preprocessor requires xarray DataArray input with a 'time'
coordinate. Unlike GaussianTransform and SpectralSynthesizer (which
operate on unitless 1D arrays), Preprocessor is inherently time-aware —
it needs calendar metadata for seasonal cycle removal and trend
evaluation. This same implementation handles both 1D scalar time series
and 2D spatial fields: xarray's polyfit and groupby operations apply
along the time dimension regardless of additional spatial dimensions.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

class Preprocessor:
    """
    Removes and restores deterministic SMB components.

    Applies up to two deterministic removals in sequence:
      1. Linear trend  (via least-squares polynomial fit)
      2. Monthly climatology  (mean seasonal cycle)

    The residuals passed to GaussianTransform are therefore the
    stochastic component of SMB variability, free of the long-term
    trend and the repeating seasonal signal.

    Parameters
    ----------
    remove_trend : bool
        Whether to fit and remove a linear trend. Default True.
    remove_seasonal : bool
        Whether to compute and remove the monthly climatological mean.
        Default True. The climatology is computed on the detrended
        series so that the trend does not bias the seasonal estimates.

    Attributes
    ----------
    trend_coeffs : xr.DataArray or None
        Polynomial coefficients from xr.polyfit (degree 1).
        Shape: (2,) for 1D input; (2, ...) for spatial input.
    seasonal_cycle : xr.DataArray or None
        Monthly climatological means indexed by month (1–12).
        Computed from the detrended series.
    n_obs : int or None
        Number of time steps in the training data.
    is_fitted : bool
        Whether fit() has been called.

    Examples
    --------
    >>> pp = Preprocessor()
    >>> pp.fit(smb)
    >>> residuals = pp.transform(smb)            # → stochastic anomalies
    >>> pp.check_stationarity(residuals)
    >>> smb_reconstructed = pp.inverse_transform(synthetic_anomaly)
    """

    def __init__(
        self,
        remove_trend: bool = True,
        remove_seasonal: bool = True,
    ) -> None:
        self.remove_trend = remove_trend
        self.remove_seasonal = remove_seasonal

        # Fitted attributes
        self.trend_coeffs: xr.DataArray | None = None
        self.seasonal_cycle: xr.DataArray | None = None
        self.n_obs: int | None = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor is not fitted. Call fit() before "
                "transform() or inverse_transform()."
            )

    @staticmethod
    def _check_time_coord(data: xr.DataArray) -> None:
        """
        Verify the input has a 'time' coordinate with datetime-like values.
        Preprocessor needs calendar metadata for groupby operations.
        """
        if not isinstance(data, xr.DataArray):
            raise TypeError(
                f"Preprocessor requires xr.DataArray input, got {type(data).__name__}. "
                "Unlike GaussianTransform and SpectralSynthesizer, Preprocessor "
                "requires time metadata for trend evaluation and seasonal removal."
            )
        if 'time' not in data.coords:
            raise ValueError(
                "Input DataArray must have a 'time' coordinate. "
                "Ensure the time dimension is named 'time' and has "
                "datetime-like values (cftime or numpy datetime64)."
            )

    def _evaluate_trend(self, time_coord: xr.DataArray) -> xr.DataArray:
        """Evaluate stored polynomial trend at the given time coordinate."""
        return xr.polyval(time_coord, self.trend_coeffs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: xr.DataArray) -> Preprocessor:
        """
        Learn the linear trend and monthly climatology from training data.

        Parameters
        ----------
        data : xr.DataArray
            SMB time series with a 'time' coordinate. For 1D use, shape
            is (n_time,). For spatial use, any additional dimensions are
            handled automatically — the same operations are applied at
            every grid point.

        Returns
        -------
        self : Preprocessor
            Returns self for method chaining.

        Notes
        -----
        Order of operations: trend is fitted first, then the monthly
        climatology is computed from the detrended series. This prevents
        a long-term trend from biasing the seasonal mean estimates.

        For a 45-year record (n=540 monthly values), the linear trend
        captures a smooth background signal. Non-linear low-frequency
        variability remains in the residuals and is captured by the
        SpectralSynthesizer's decadal-band PSD estimate.
        """
        self._check_time_coord(data)
        self.n_obs = data.sizes['time']

        # --- Step 1: linear trend ---
        if self.remove_trend:
            fit_result = data.polyfit(dim='time', deg=1)
            # Store coefficients (not evaluated trend) so we can later
            # evaluate at arbitrary synthetic time coordinates.
            self.trend_coeffs = fit_result['polyfit_coefficients']

        # --- Step 2: monthly climatology ---
        # Computed on the detrended series to avoid trend contamination
        # of the seasonal estimates.
        if self.remove_seasonal:
            if self.remove_trend:
                trend_vals = self._evaluate_trend(data['time'])
                detrended = data - trend_vals
            else:
                detrended = data

            # Mean value for each calendar month (1–12)
            self.seasonal_cycle = detrended.groupby('time.month').mean()

        self.is_fitted = True
        return self

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """
        Remove deterministic components and return stochastic residuals.

        Applies the same removals learned at fit() time:
          1. Subtract evaluated linear trend
          2. Subtract monthly climatological mean

        Parameters
        ----------
        data : xr.DataArray
            SMB time series. Must have the same structure as the data
            passed to fit(), though it can be a different time slice
            (e.g. a validation split).

        Returns
        -------
        residuals : xr.DataArray
            Stochastic anomalies with trend and seasonal cycle removed.
            Approximately zero-mean and stationary.

        Notes
        -----
        The returned residuals are passed directly to
        GaussianTransform.fit() and GaussianTransform.transform().
        Their stationarity should be verified with check_stationarity()
        before proceeding.
        """
        self._check_fitted()
        self._check_time_coord(data)

        result = data.copy()

        if self.remove_trend and self.trend_coeffs is not None:
            trend_vals = self._evaluate_trend(data['time'])
            result = result - trend_vals

        if self.remove_seasonal and self.seasonal_cycle is not None:
            result = result.groupby('time.month') - self.seasonal_cycle

        return result

    def fit_transform(self, data: xr.DataArray) -> xr.DataArray:
        """
        Convenience method: fit() then transform() in one call.

        Parameters
        ----------
        data : xr.DataArray
            SMB time series.

        Returns
        -------
        residuals : xr.DataArray
            Stochastic anomalies with deterministic components removed.
        """
        return self.fit(data).transform(data)

    def inverse_transform(
        self,
        anomaly: xr.DataArray,
        add_trend: bool = False,
        seasonal_amp_scale: float = 1.0,
    ) -> xr.DataArray:
        """
        Add deterministic components back to a stochastic anomaly series.

        Parameters
        ----------
        anomaly : xr.DataArray
            Stochastic anomaly series with a 'time' coordinate.
            For synthetic output this will typically be a 1000-year
            series with a synthetic cftime coordinate.
        add_trend : bool
            Whether to add the linear trend back. Default False.
            For long synthetic series (e.g. 1000 years), the trend
            is deliberately not re-added because extrapolating a 45-year
            observed trend over 1000 years is physically unreasonable.
            Set to True only when reconstructing the training period for
            validation purposes.
        seasonal_amp_scale : float
            Amplitude multiplier on the deterministic seasonal cycle.
            Default 1.0 reproduces the observed climatology exactly. Pass
            Experiment.seasonal_amplitude_factor (= sqrt(seasonal_scale)) to
            amplify or suppress the annual-timescale forcing. Mean-safe: the
            twelve monthly anomalies sum to zero, so scaling them does not
            shift the record mean.

        Returns
        -------
        reconstructed : xr.DataArray
            Anomaly with the seasonal cycle (and optionally trend) added back.

        Notes
        -----
        The observed SMB mean is NOT added here. It is added once and
        only once in SMBGenerator.generate(), consistent with the
        architecture established in GaussianTransform.inverse_transform().

        The synthetic DataArray must have a datetime-like time coordinate
        so that groupby('time.month') can identify the calendar month of
        each time step and apply the correct seasonal correction.
        """
        self._check_fitted()
        self._check_time_coord(anomaly)

        result = anomaly.copy()

        # Add seasonal cycle first (before trend, to match removal order).
        # seasonal_amp_scale amplifies/suppresses the deterministic cycle;
        # 1.0 is the observed climatology. Mean-safe (anomalies sum to zero).
        if self.remove_seasonal and self.seasonal_cycle is not None:
            result = result.groupby('time.month') + seasonal_amp_scale * self.seasonal_cycle

        # Optionally add trend — off by default for synthetic series
        if add_trend and self.remove_trend and self.trend_coeffs is not None:
            trend_vals = self._evaluate_trend(anomaly['time'])
            result = result + trend_vals

        return result

    def check_stationarity(
        self,
        data: xr.DataArray,
        verbose: bool = True,
    ) -> dict:
        """
        Run an Augmented Dickey-Fuller test for stationarity.

        Call this after transform() to confirm the residuals are
        stationary before passing to GaussianTransform. A non-stationary
        residual series violates the assumptions of the spectral synthesis
        method.

        Parameters
        ----------
        data : xr.DataArray
            1D time series to test (typically the output of transform()).
        verbose : bool
            If True, print a formatted summary.

        Returns
        -------
        results : dict
            Keys:
                adf_statistic : float — ADF test statistic
                p_value       : float — p-value (< 0.05 → reject unit root)
                n_lags        : int   — number of lags chosen by AIC
                is_stationary : bool  — True if p_value < 0.05
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            raise ImportError(
                "statsmodels is required for check_stationarity(). "
                "Install with: pip install statsmodels"
            )

        x = data.values.ravel()
        adf_result = adfuller(x, autolag='AIC')

        adf_stat     = float(adf_result[0])
        p_value      = float(adf_result[1])
        n_lags       = int(adf_result[2])
        is_stationary = p_value < 0.05

        results = {
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'n_lags': n_lags,
            'is_stationary': is_stationary,
        }

        if verbose:
            status = "STATIONARY ✓" if is_stationary else "NON-STATIONARY ✗"
            print(f"ADF stationarity test — {status}")
            print(f"  ADF statistic:  {adf_stat:.4f}")
            print(f"  p-value:        {p_value:.4f}  (threshold: 0.05)")
            print(f"  Lags used:      {n_lags}")
            if not is_stationary:
                print(
                    "  WARNING: residuals appear non-stationary. "
                    "Check that remove_trend=True is set and that the "
                    "data does not contain structural breaks."
                )

        return results

    def summarize(self) -> None:
        """Print a summary of the fitted deterministic components."""
        if not self.is_fitted:
            print("Preprocessor not fitted. Call fit() first.")
            return

        print("Preprocessor summary")
        print(f"  remove_trend:    {self.remove_trend}")
        print(f"  remove_seasonal: {self.remove_seasonal}")
        print(f"  n_obs:           {self.n_obs}")

        if self.remove_trend and self.trend_coeffs is not None:
            # Extract slope and intercept from stored coefficients
            coeffs = self.trend_coeffs.values
            if coeffs.ndim == 1:
                print(f"  trend slope:     {float(coeffs[0]):.4e}  (units/time-unit)")
                print(f"  trend intercept: {float(coeffs[1]):.4e}")

        if self.remove_seasonal and self.seasonal_cycle is not None:
            sc = self.seasonal_cycle.values
            if sc.ndim == 1:
                print(f"  seasonal range:  [{sc.min():.4f}, {sc.max():.4f}]  "
                      f"(peak-to-trough: {sc.max() - sc.min():.4f})")

    def __repr__(self) -> str:
        status = f"fitted=True, n_obs={self.n_obs}" if self.is_fitted else "fitted=False"
        return (
            f"Preprocessor({status}, "
            f"remove_trend={self.remove_trend}, "
            f"remove_seasonal={self.remove_seasonal})"
        )
    

    def plot_decomposition(self, smb, save_path=None):
        """
        Four-panel figure showing the effect of each preprocessing step.
        Only requires the raw SMB DataArray — all components are derived
        from self after fit() has been called.

        Panels
        ------
        Top-left  : Raw SMB time series with fitted trend overlaid
        Top-right : PSD comparison — raw vs detrended vs fully preprocessed
                    (shows the ~30x annual-band reduction from seasonal removal)
        Bottom-left : Monthly climatology bar chart (seasonal cycle removed)
        Bottom-right: Final stochastic residuals passed to GaussianTransform
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before plot_decomposition().")

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.signal import welch

        # ── Derive all components from self + smb ──
        trend      = self._evaluate_trend(smb)
        detrended  = smb - trend
        residuals  = self.transform(smb)          # removes trend + seasonal
        months     = np.arange(1, 13)
        fs         = 12.0                         # cycles per year

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Preprocessor decomposition", fontsize=12)

        # ── Top-left: raw SMB + trend ──
        ax = axes[0, 0]
        ax.plot(smb.time, smb.values,   color="tab:blue",   lw=0.8,
                label="Raw SMB", alpha=0.8)
        ax.plot(smb.time, trend.values, color="tab:red",    lw=1.5,
                linestyle="--", label="Linear trend")
        ax.set_ylabel(f"SMB ({smb.attrs.get('units', 'm w.e.')})")
        ax.set_title("Raw SMB with trend")
        ax.legend(fontsize=8)

        # ── Top-right: PSD comparison ──
        ax = axes[0, 1]
        for series, label, color, lw in [
            (smb.values,        "Raw",         "tab:gray",   0.8),
            (detrended.values,  "Detrended",   "tab:orange", 1.2),
            (residuals.values,  "Residuals\n(trend + seasonal removed)",
                                            "tab:blue",   1.8),
        ]:
            f, p = welch(series, fs=fs, nperseg=min(60, len(series)))
            ax.loglog(f[1:], p[1:], label=label, color=color, lw=lw)
        ax.axvline(1.0, color="gray", lw=0.8, linestyle=":", alpha=0.6)
        ax.set_xlabel("Frequency (cycles yr⁻¹)")
        ax.set_ylabel("PSD")
        ax.set_title("PSD at each preprocessing stage")
        ax.legend(fontsize=8)

        # ── Bottom-left: seasonal cycle bar chart ──
        ax = axes[1, 0]
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        colors = ["tab:blue" if v >= 0 else "tab:red"
                for v in self.seasonal_cycle.values]
        ax.bar(months, self.seasonal_cycle.values, color=colors, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_ylabel(f"SMB anomaly ({smb.attrs.get('units', 'm w.e.')})")
        ax.set_title("Seasonal cycle (monthly climatology)")

        # ── Bottom-right: final residuals ──
        ax = axes[1, 1]
        ax.plot(smb.time, residuals.values, color="tab:blue", lw=0.8, alpha=0.8)
        ax.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_ylabel(f"Residual ({smb.attrs.get('units', 'm w.e.')})")
        ax.set_title("Stochastic residuals → GaussianTransform")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()