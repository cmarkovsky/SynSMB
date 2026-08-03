"""
spatial_preprocessor.py
=======================
Applies the same detrend + seasonal cycle removal as the 1D Preprocessor,
but independently at every RACMO grid point.

Operates on xr.DataArray (time, rlat, rlon) — the output of
SMBFieldLoader.load() — and returns the stochastic residual field ready
for EOFDecomposer.

Why spatial rather than basin-mean preprocessing
-------------------------------------------------
The seasonal cycle of SMB is spatially variable across a glacier basin:
accumulation peaks at different calendar months in different parts of the
domain depending on orography, distance from the coast, and prevailing
wind direction. Removing the seasonal cycle in physical space (before EOF
decomposition) prevents the first EOF from being dominated by the spatial
pattern of the seasonal cycle rather than the stochastic variability of
interest.

Pipeline position
-----------------
    SMBFieldLoader.load()        → (time, rlat, rlon), units m w.e.
         ↓
    SpatialPreprocessor.fit_transform()   ← this class
         ↓
    EOFDecomposer.fit()          → n_modes PC time series

Relationship with 1D Preprocessor
-----------------------------------
SpatialPreprocessor stores the same two deterministic components:
  - trend_coeffs  : per-grid-cell linear trend
  - seasonal_means: per-grid-cell monthly climatological anomalies

The field_mean (time-mean at each grid point) is stored separately and
is added back by SMBFieldGenerator.generate(), analogous to smb_mean in
the 1D SMBGenerator. The trend is deliberately NOT restored in
inverse_transform() for the same reason as the 1D case.
"""

from __future__ import annotations

import warnings
import numpy as np
import xarray as xr


class SpatialPreprocessor:
    """
    Per-grid-cell linear detrending and seasonal cycle removal.

    Parameters
    ----------
    deg : int
        Degree of the detrending polynomial. Default 1 (linear).

    Attributes (available after fit())
    ------------------------------------
    field_mean    : xr.DataArray (rlat, rlon)
        Time-mean SMB at each grid point. Stored separately and NOT
        restored in inverse_transform(); SMBFieldGenerator handles this.
    trend_coeffs  : xr.DataArray (degree+1, rlat, rlon)
        Polynomial coefficients from xr.polyfit.
    seasonal_means : xr.DataArray (month=12, rlat, rlon)
        Monthly climatological anomalies (relative to field_mean and
        trend) — zero mean across the 12 months at each grid point.
    spatial_dims  : list[str]
        Names of the spatial dimensions (e.g. ['rlat', 'rlon']).
    n_time        : int
        Length of the training time series.
    """

    def __init__(self, deg: int = 1) -> None:
        self.deg = deg
        self._is_fitted       = False
        self._field_mean:     xr.DataArray | None = None
        self._trend_coeffs:   xr.DataArray | None = None
        self._seasonal_means: xr.DataArray | None = None
        self._spatial_dims:   list[str]   = []
        self._n_time:         int         = 0
        self._time_coord:     xr.DataArray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, field: xr.DataArray) -> SpatialPreprocessor:
        """
        Learn the trend and seasonal cycle from the training field.

        Parameters
        ----------
        field : xr.DataArray, shape (time, rlat, rlon)
            Basin-masked SMB field from SMBFieldLoader (NaN outside basin).

        Returns
        -------
        self
        """
        self._validate_input(field)

        self._spatial_dims = [d for d in field.dims if d != "time"]
        self._n_time       = field.sizes["time"]

        # Build a numeric time axis (integer index) for polyfit.
        # Using an integer index rather than the raw time values avoids
        # issues with cftime coordinates and gives dimensionless
        # polynomial coefficients that are easy to interpret.
        t_idx = self._make_time_index(field)

        # 1. Store time-mean field (used by SMBFieldGenerator, not here)
        self._field_mean = field.mean(dim="time")

        # 2. Mean-center so the trend is estimated around zero
        field_centered = field - self._field_mean

        # 3. Fit per-grid-cell polynomial trend
        self._trend_coeffs = self._fit_trend(field_centered, t_idx)

        # 4. Evaluate and subtract the trend
        trend      = self._eval_trend(t_idx)
        detrended  = field_centered - trend

        # 5. Monthly climatological means on the detrended, centred field.
        #    These are anomalies relative to the annual mean, so their
        #    mean across the 12 months is ≈ 0 at each grid point.
        self._seasonal_means = (
            detrended
            .groupby("time.month")
            .mean("time")
        )

        self._time_coord = t_idx
        self._is_fitted  = True
        return self

    def transform(self, field: xr.DataArray) -> xr.DataArray:
        """
        Remove the trend and seasonal cycle from ``field``.

        Parameters
        ----------
        field : xr.DataArray, shape (time, rlat, rlon)
            Must have the same spatial dimensions as the training field.
            Can be a different length in time (e.g. for validation).

        Returns
        -------
        residuals : xr.DataArray, shape (time, rlat, rlon)
            Stochastic residuals. NaN cells are preserved.
            Mean across time ≈ 0 at each valid grid point.
        """
        self._check_fitted()
        self._validate_input(field)

        t_idx     = self._make_time_index(field)
        trend     = self._eval_trend(t_idx)
        centered  = field - self._field_mean
        detrended = centered - trend
        residuals = detrended.groupby("time.month") - self._seasonal_means

        residuals.attrs.update({
            "long_name": "Stochastic SMB residuals (trend + seasonal removed)",
            "units":     field.attrs.get("units", "m w.e."),
        })
        return residuals

    def fit_transform(self, field: xr.DataArray) -> xr.DataArray:
        """Fit on ``field`` and return the residuals in one call."""
        return self.fit(field).transform(field)

    def inverse_transform(
        self,
        residuals: xr.DataArray,
        add_mean: bool = False,
        seasonal_amp_scale: float = 1.0,
    ) -> xr.DataArray:
        """
        Restore the seasonal cycle from stochastic residuals.

        Note: the linear trend is NOT restored — extrapolating a 45-year
        trend over 1000 years is physically unreasonable.
        Note: field_mean is NOT added here — SMBFieldGenerator handles
        this step to keep the mean addition in one place.

        Parameters
        ----------
        residuals : xr.DataArray, shape (time, rlat, rlon)
            Stochastic residuals in physical units.
        add_mean : bool
            If True, also add field_mean back. Default False (let
            SMBFieldGenerator handle the mean). Set True only for
            round-trip validation.
        seasonal_amp_scale : float
            Multiplier on the seasonal-cycle amplitude (an *amplitude*
            factor; pass sqrt(seasonal_variance_scale)). Default 1.0
            reproduces the observed climatology exactly. Values >1 amplify
            the deterministic seasonal cycle to represent a stronger annual
            forcing, mirroring the 1-D Preprocessor. Mean-safe: the seasonal
            anomalies sum to zero over the twelve months, so scaling them
            leaves the record mean unchanged.

        Returns
        -------
        reconstructed : xr.DataArray, shape (time, rlat, rlon)
        """
        self._check_fitted()

        # Restore seasonal anomalies (optionally amplitude-scaled)
        reconstructed = (
            residuals.groupby("time.month")
            + seasonal_amp_scale * self._seasonal_means
        )

        # Optionally restore the field mean
        if add_mean:
            reconstructed = reconstructed + self._field_mean

        return reconstructed

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def field_mean(self) -> xr.DataArray:
        self._check_fitted()
        return self._field_mean  # type: ignore

    @property
    def seasonal_means(self) -> xr.DataArray:
        self._check_fitted()
        return self._seasonal_means  # type: ignore

    @property
    def trend_coeffs(self) -> xr.DataArray:
        self._check_fitted()
        return self._trend_coeffs  # type: ignore

    @property
    def spatial_dims(self) -> list[str]:
        return self._spatial_dims

    @property
    def n_time(self) -> int:
        return self._n_time

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def seasonal_rmse(self) -> float:
        """
        Root-mean-squared seasonal-mean of the residuals over the
        training data. Should be near zero after fit_transform().
        Useful as a quick sanity check.
        """
        self._check_fitted()
        field     = self._field_mean   # placeholder — call transform() externally
        warnings.warn(
            "Call transform() on the training data and check "
            "residuals.groupby('time.month').mean() directly.",
            stacklevel=2,
        )
        return float("nan")

    def variance_map(self, residuals: xr.DataArray) -> xr.DataArray:
        """
        Per-grid-cell variance of the stochastic residuals.
        Useful for inspecting the spatial structure before EOF decomposition.
        """
        return residuals.var(dim="time")

    def plot_decomposition(
        self,
        field: xr.DataArray,
        save_path: str | None = None,
    ) -> None:
        """
        Four-panel diagnostic figure:
          Top-left  : Time-mean SMB field (field_mean)
          Top-right : Variance of stochastic residuals (variance map)
          Bottom-left : Basin-mean seasonal cycle (12 monthly means)
          Bottom-right: Basin-mean residual time series
        """
        self._check_fitted()
        import matplotlib.pyplot as plt

        residuals = self.transform(field)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("SpatialPreprocessor decomposition", fontsize=12)

        # ── Top-left: time-mean field ──
        ax = axes[0, 0]
        pcm = ax.pcolormesh(
            self._field_mean.values, cmap="Blues", shading="auto"
        )
        plt.colorbar(pcm, ax=ax, label="SMB (m w.e.)")
        ax.set_title("Time-mean SMB field")

        # ── Top-right: residual variance map ──
        ax = axes[0, 1]
        vmap = self.variance_map(residuals)
        pcm  = ax.pcolormesh(vmap.values, cmap="YlOrRd", shading="auto")
        plt.colorbar(pcm, ax=ax, label="Variance (m w.e.)²")
        ax.set_title("Stochastic residual variance\n(target for EOF decomposition)")

        # ── Bottom-left: basin-mean seasonal cycle ──
        ax = axes[1, 0]
        basin_seasonal = self._seasonal_means.mean(
            dim=self._spatial_dims, skipna=True
        )
        months = np.arange(1, 13)
        colors = [
            "tab:blue" if v >= 0 else "tab:red"
            for v in basin_seasonal.values
        ]
        ax.bar(months, basin_seasonal.values, color=colors, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        month_labels = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_ylabel("SMB anomaly (m w.e.)")
        ax.set_title("Basin-mean seasonal cycle")

        # ── Bottom-right: basin-mean residual time series ──
        ax = axes[1, 1]
        basin_resid = residuals.mean(dim=self._spatial_dims, skipna=True)
        ax.plot(basin_resid.values, color="tab:blue", lw=0.8, alpha=0.8)
        ax.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_xlabel("Time step (months)")
        ax.set_ylabel("Residual SMB (m w.e.)")
        ax.set_title("Basin-mean stochastic residuals\n→ EOFDecomposer")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_input(self, field: xr.DataArray) -> None:
        if not isinstance(field, xr.DataArray):
            raise TypeError(
                f"Expected xr.DataArray, got {type(field).__name__}."
            )
        if "time" not in field.dims:
            raise ValueError(
                f"Field must have a 'time' dimension. Got dims: {list(field.dims)}"
            )
        if field.ndim < 2:
            raise ValueError(
                "Field must be at least 2D (time + at least one spatial dim). "
                f"Got {field.ndim}D."
            )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "SpatialPreprocessor is not fitted. Call fit() first."
            )

    def _make_time_index(self, field: xr.DataArray) -> xr.DataArray:
        """
        Build an integer time index 0, 1, ..., n-1 as a DataArray
        with the same 'time' coordinate as ``field``.

        Using a simple integer index rather than actual datetime values
        avoids issues with cftime coordinates in xr.polyfit and makes
        the polynomial coefficients dimensionless.
        """
        return xr.DataArray(
            np.arange(field.sizes["time"], dtype=float),
            coords={"time": field.time},
            dims=["time"],
        )

    def _fit_trend(
        self,
        field: xr.DataArray,
        t_idx: xr.DataArray,
    ) -> xr.DataArray:
        """
        Fit a degree-self.deg polynomial at each grid point.

        Returns the polyfit_coefficients DataArray
        (degree+1, rlat, rlon), where index 0 = highest degree.
        """
        # Replace actual time coord with integer index for polyfit
        field_numeric = field.assign_coords(time=t_idx)
        result = field_numeric.polyfit(dim="time", deg=self.deg, skipna=True)
        return result["polyfit_coefficients"]

    def _eval_trend(self, t_idx: xr.DataArray) -> xr.DataArray:
        """Evaluate the fitted polynomial at the time points in t_idx."""
        return xr.polyval(t_idx, self._trend_coeffs)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._is_fitted:
            return f"SpatialPreprocessor(deg={self.deg}, fitted=False)"
        ny = self._field_mean.sizes.get(self._spatial_dims[0], "?")
        nx = self._field_mean.sizes.get(self._spatial_dims[1], "?") \
             if len(self._spatial_dims) > 1 else "?"
        return (
            f"SpatialPreprocessor("
            f"deg={self.deg}, "
            f"n_time={self._n_time}, "
            f"grid=({ny}×{nx}))"
        )