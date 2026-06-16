"""
eof_decomposer.py
=================
SVD-based EOF (Empirical Orthogonal Function) decomposition of the
stochastic residual field produced by SpatialPreprocessor.

Role in the 2D pipeline
-----------------------
EOFDecomposer sits between SpatialPreprocessor and the 1D synthesis
components. It converts the 2D residual field into a set of PC time
series that the existing GaussianTransform + SpectralSynthesizer
can handle without modification, and provides the reconstruction
step that converts synthetic PCs back into a 2D spatial field.

    SpatialPreprocessor.fit_transform()   → (time, rlat, rlon)
         ↓
    EOFDecomposer.fit_transform()         → (time, n_modes)  PC time series
         ↓
    [GaussianTransform + SpectralSynthesizer per PC — existing 1D code]
         ↓
    EOFDecomposer.inverse_transform()     → (time, rlat, rlon) synthetic field

Mathematical background
-----------------------
Let X ∈ ℝ^{T×P} be the residual field flattened to (time, n_valid_cells),
optionally area-weighted. The truncated SVD X ≈ U S Vᵀ gives:

    EOFs   = Vᵀ  ∈ ℝ^{n_modes × P}  (spatial patterns, orthonormal)
    PCs    = U S  ∈ ℝ^{T × n_modes}  (temporal coefficients)
    expvar = S²  / ‖S²‖₁              (fraction of variance per mode)

Reconstruction: X̂ = PCs_syn @ EOFs, then un-weight → spatial field.

Area weighting
--------------
Each cell is weighted by √cos(lat) before SVD so that the decomposition
minimises area-weighted variance rather than giving equal weight to all
grid points regardless of their physical size. For RACMO's rotated
lat-lon grid this correction is small but non-negligible near the ice
sheet margins.
"""

from __future__ import annotations

import warnings
import numpy as np
import xarray as xr


class EOFDecomposer:
    """
    EOF decomposition of a 2D stochastic SMB residual field.

    Parameters
    ----------
    n_modes : int
        Number of EOF modes to retain. Default 10. Use
        suggest_n_modes() to choose this data-adaptively.

    Attributes (after fit())
    ------------------------
    pcs            : np.ndarray (n_time, n_modes)
        PC time series of the training data. These are passed to the
        1D GaussianTransform + SpectralSynthesizer per mode.
    explained_variance_ratio : np.ndarray (n_modes,)
        Fraction of total variance explained by each mode.
    singular_values : np.ndarray (n_modes,)
        Singular values from the SVD.
    n_valid_cells : int
        Number of non-NaN grid cells inside the basin.
    """

    def __init__(self, n_modes: int = 10) -> None:
        self.n_modes       = n_modes
        self._is_fitted    = False

        # Set during fit()
        self._eofs_weighted: np.ndarray | None = None  # (n_modes, n_valid)
        self._pcs:           np.ndarray | None = None  # (n_time, n_modes)
        self._singular_vals: np.ndarray | None = None  # (n_modes,)
        self._expvar:        np.ndarray | None = None  # (n_modes,)
        self._weights:       np.ndarray | None = None  # (n_valid,)
        self._valid_mask:    np.ndarray | None = None  # (rlat*rlon,) bool
        self._field_shape:   tuple        = ()          # (rlat, rlon)
        self._spatial_dims:  list[str]   = []
        self._n_time:        int         = 0
        self._n_valid:       int         = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        residuals: xr.DataArray,
        lat:       xr.DataArray | None = None,
    ) -> EOFDecomposer:
        """
        Fit the EOF decomposition on the residual field.

        Parameters
        ----------
        residuals : xr.DataArray, shape (time, rlat, rlon)
            Output of SpatialPreprocessor.fit_transform().
            NaN outside the basin mask.
        lat : xr.DataArray (rlat, rlon) or None
            Latitude in degrees for area weighting. If None, all cells
            are weighted equally (not recommended for large domains).

        Returns
        -------
        self
        """
        self._validate_input(residuals)

        self._spatial_dims = [d for d in residuals.dims if d != "time"]
        self._field_shape  = tuple(
            residuals.sizes[d] for d in self._spatial_dims
        )
        n_time   = residuals.sizes["time"]
        self._n_time = n_time

        # 1. Flatten to (time, n_cells) and find valid (non-NaN) cells
        flat  = residuals.values.reshape(n_time, -1)   # (T, rlat*rlon)
        valid = np.all(np.isfinite(flat), axis=0)       # (rlat*rlon,)
        if not valid.any():
            raise ValueError(
                "No valid (non-NaN) cells found in residuals. "
                "Ensure SMBFieldLoader and SpatialPreprocessor ran correctly."
            )
        self._valid_mask = valid
        self._n_valid    = int(valid.sum())
        X = flat[:, valid]                              # (T, n_valid)

        # 2. Area weights √cos(lat) for valid cells
        if lat is not None:
            lat_flat = lat.values.flatten()[valid]
            w = np.sqrt(np.cos(np.deg2rad(np.abs(lat_flat))))
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        else:
            w = np.ones(self._n_valid)
        self._weights = w

        # 3. Apply weights
        X_w = X * w[np.newaxis, :]                     # (T, n_valid)

        # 4. Truncated SVD — economy form avoids huge U matrix
        n_modes_safe = min(self.n_modes, n_time - 1, self._n_valid - 1)
        if n_modes_safe < self.n_modes:
            warnings.warn(
                f"Requested n_modes={self.n_modes} exceeds the maximum "
                f"({n_modes_safe}). Using {n_modes_safe}.",
                stacklevel=2,
            )
            self.n_modes = n_modes_safe

        U, s, Vt = np.linalg.svd(X_w, full_matrices=False)
        # U:  (T, min(T, n_valid))
        # s:  (min(T, n_valid),)
        # Vt: (min(T, n_valid), n_valid)

        # 5. Retain n_modes
        self._eofs_weighted = Vt[:self.n_modes, :]    # (n_modes, n_valid)
        self._pcs           = U[:, :self.n_modes] * s[:self.n_modes]
        self._singular_vals = s[:self.n_modes]
        self._expvar        = (s ** 2) / (s ** 2).sum()

        # 6. Standardise EOF signs: first non-zero loading positive.
        #    This makes the PCs more interpretable (positive PC →
        #    above-average SMB) and keeps signs consistent across
        #    calls with the same data.
        for i in range(self.n_modes):
            first_nonzero = self._eofs_weighted[i].flat[
                next((j for j, v in enumerate(self._eofs_weighted[i])
                      if v != 0), 0)
            ]
            if first_nonzero < 0:
                self._eofs_weighted[i] *= -1
                self._pcs[:, i]        *= -1

        self._is_fitted = True
        print(
            f"EOFDecomposer fitted: {self.n_modes} modes, "
            f"{self._n_valid} valid cells, "
            f"cumulative variance = "
            f"{self._expvar[:self.n_modes].sum():.1%}"
        )
        return self

    def transform(self, residuals: xr.DataArray) -> np.ndarray:
        """
        Project a residual field onto the fitted EOF patterns.

        Can be called on data not used for fitting (e.g. validation
        period or held-out data) to get out-of-sample PCs.

        Parameters
        ----------
        residuals : xr.DataArray, shape (time, rlat, rlon)
            Must have the same spatial structure as the training field.

        Returns
        -------
        pcs : np.ndarray, shape (time, n_modes)
        """
        self._check_fitted()
        self._validate_input(residuals)

        n_time = residuals.sizes["time"]
        flat   = residuals.values.reshape(n_time, -1)
        X      = flat[:, self._valid_mask]          # (time, n_valid)
        X_w    = X * self._weights[np.newaxis, :]  # weighted
        # Project onto EOF patterns
        pcs = X_w @ self._eofs_weighted.T           # (time, n_modes)
        return pcs

    def fit_transform(
        self,
        residuals: xr.DataArray,
        lat: xr.DataArray | None = None,
    ) -> np.ndarray:
        """Fit and return training PCs in one call."""
        return self.fit(residuals, lat=lat).pcs

    def inverse_transform(
        self,
        pcs: np.ndarray,
        time_coord=None,
    ) -> xr.DataArray:
        """
        Reconstruct the spatial field from PC time series.

        Called during synthetic generation after the 1D pipeline has
        produced synthetic PCs for each mode.

        Parameters
        ----------
        pcs : np.ndarray, shape (time_syn, n_modes)
            Synthetic PC time series — one column per EOF mode.
        time_coord : array-like or None
            Time coordinate for the output DataArray. If None, an
            integer index is used.

        Returns
        -------
        field : xr.DataArray, shape (time_syn, rlat, rlon)
            Reconstructed spatial field. NaN outside the basin mask.
        """
        self._check_fitted()
        if pcs.shape[1] != self.n_modes:
            raise ValueError(
                f"pcs has {pcs.shape[1]} columns but n_modes={self.n_modes}."
            )

        n_time_syn = pcs.shape[0]

        # Reconstruct weighted field: (time_syn, n_valid)
        X_w_syn = pcs @ self._eofs_weighted

        # Un-weight to recover physical units
        X_syn = X_w_syn / self._weights[np.newaxis, :]

        # Map back into (time_syn, rlat, rlon) with NaN outside basin
        field_flat = np.full((n_time_syn, self._valid_mask.size), np.nan)
        field_flat[:, self._valid_mask] = X_syn

        ny, nx   = self._field_shape
        field_3d = field_flat.reshape(n_time_syn, ny, nx)

        # Build time coordinate
        if time_coord is None:
            time_coord = np.arange(n_time_syn)

        return xr.DataArray(
            field_3d,
            dims=["time"] + self._spatial_dims,
            coords={"time": time_coord},
            attrs={"long_name": "Reconstructed SMB residuals from EOF synthesis"},
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def suggest_n_modes(self, threshold: float = 0.95) -> int:
        """
        Return the minimum number of modes needed to explain at least
        ``threshold`` fraction of the total variance.

        Parameters
        ----------
        threshold : float
            Target cumulative explained variance fraction. Default 0.95.

        Returns
        -------
        n : int
        """
        self._check_fitted()
        cumvar = np.cumsum(self._expvar)
        # Find first index where cumulative variance exceeds threshold
        hits = np.where(cumvar >= threshold)[0]
        if len(hits) == 0:
            warnings.warn(
                f"The fitted {len(self._expvar)} singular values only "
                f"explain {cumvar[-1]:.1%} of total variance — less than "
                f"the {threshold:.0%} threshold. Re-fit with more modes "
                f"or lower the threshold."
            )
            return len(self._expvar)
        return int(hits[0]) + 1

    def eofs_as_field(self) -> xr.DataArray:
        """
        Return EOFs reshaped to (n_modes, rlat, rlon) with NaN outside
        the basin — convenient for plotting.

        Returns
        -------
        eofs : xr.DataArray, shape (n_modes, rlat, rlon)
        """
        self._check_fitted()
        ny, nx = self._field_shape
        n      = self.n_modes

        flat = np.full((n, self._valid_mask.size), np.nan)
        flat[:, self._valid_mask] = self._eofs_weighted / self._weights
        eofs_3d = flat.reshape(n, ny, nx)

        return xr.DataArray(
            eofs_3d,
            dims=["mode"] + self._spatial_dims,
            coords={"mode": np.arange(1, n + 1)},
            attrs={"long_name": "EOF spatial patterns (un-weighted)"},
        )

    def plot_eofs(
        self,
        n: int = 4,
        save_path: str | None = None,
    ) -> None:
        """
        Plot the first ``n`` EOF spatial patterns.

        Each panel shows one EOF on the basin grid with a diverging
        colormap centred at zero. The title reports the explained
        variance for that mode.
        """
        self._check_fitted()
        import matplotlib.pyplot as plt

        n     = min(n, self.n_modes)
        eofs  = self.eofs_as_field()
        ncols = min(n, 4)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4 * ncols, 3.5 * nrows),
            squeeze=False,
        )
        fig.suptitle("EOF spatial patterns", fontsize=12)

        for i in range(n):
            ax = axes[i // ncols, i % ncols]
            eof_vals = eofs.isel(mode=i).values
            vmax = np.nanpercentile(np.abs(eof_vals), 98)
            pcm  = ax.pcolormesh(
                eof_vals,
                cmap="RdBu_r",
                vmin=-vmax, vmax=vmax,
                shading="auto",
            )
            expv = self._expvar[i]
            ax.set_title(
                f"EOF {i+1}  ({expv:.1%} var)",
                fontsize=9,
            )
            plt.colorbar(pcm, ax=ax, fraction=0.04)
            ax.set_xticks([]); ax.set_yticks([])

        # Hide unused axes
        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    def plot_variance(self, save_path: str | None = None) -> None:
        """
        Two-panel variance scree figure:
          Left:  Per-mode explained variance (bar chart)
          Right: Cumulative explained variance with 80% / 95% thresholds

        Use this to justify the choice of n_modes in the paper.
        """
        self._check_fitted()
        import matplotlib.pyplot as plt

        n_all  = len(self._expvar)
        modes  = np.arange(1, n_all + 1)
        expvar = self._expvar
        cumvar = np.cumsum(expvar)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle("EOF variance decomposition", fontsize=12)

        # ── Left: per-mode bar chart ──
        ax = axes[0]
        colors = [
            "tab:blue" if i < self.n_modes else "tab:gray"
            for i in range(n_all)
        ]
        ax.bar(modes, expvar * 100, color=colors, alpha=0.85)
        ax.set_xlabel("EOF mode")
        ax.set_ylabel("Explained variance (%)")
        ax.set_title("Per-mode explained variance\n"
                     "(blue = retained modes)")
        ax.axvline(self.n_modes + 0.5, color="black",
                   lw=1.5, linestyle="--", label=f"n_modes={self.n_modes}")
        ax.legend(fontsize=9)

        # ── Right: cumulative variance ──
        ax = axes[1]
        ax.plot(modes, cumvar * 100, "o-",
                color="tab:blue", lw=2, markersize=5)
        for thresh, color, label in [
            (80, "#E69F00", "80%"),
            (95, "#D55E00", "95%"),
        ]:
            ax.axhline(thresh, color=color, lw=1.2, linestyle="--",
                       label=label)
            # Annotate crossing point
            cross = np.searchsorted(cumvar * 100, thresh)
            if cross < n_all:
                ax.plot(modes[cross], cumvar[cross] * 100,
                        "s", color=color, markersize=8, zorder=5)
        ax.axvline(self.n_modes + 0.5, color="black",
                   lw=1.5, linestyle="--")
        ax.set_xlabel("EOF mode")
        ax.set_ylabel("Cumulative explained variance (%)")
        ax.set_title("Cumulative explained variance\n"
                     "(dashed = 80% and 95% thresholds)")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_xlim(0.5, n_all + 0.5)
        

        # Find where cumvar first exceeds 99%
        cutoff = int(np.searchsorted(cumvar, 0.99)) + 5
        axes[0].set_xlim(0.5, cutoff)
        axes[1].set_xlim(0.5, cutoff)


        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    def plot_pcs(
        self,
        n: int = 4,
        save_path: str | None = None,
    ) -> None:
        """Plot the first ``n`` PC time series."""
        self._check_fitted()
        import matplotlib.pyplot as plt

        n    = min(n, self.n_modes)
        fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
        if n == 1:
            axes = [axes]
        fig.suptitle("Principal component time series", fontsize=12)

        for i, ax in enumerate(axes):
            ax.plot(self._pcs[:, i], color="tab:blue", lw=0.8, alpha=0.85)
            ax.axhline(0, color="gray", lw=0.7, linestyle="--")
            expv = self._expvar[i]
            ax.set_ylabel(f"PC {i+1}", fontsize=9)
            ax.set_title(f"PC {i+1}  ({expv:.1%} var)", fontsize=9)

        axes[-1].set_xlabel("Time step (months)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def pcs(self) -> np.ndarray:
        self._check_fitted()
        return self._pcs  # type: ignore

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        self._check_fitted()
        return self._expvar[:self.n_modes]

    @property
    def singular_values(self) -> np.ndarray:
        self._check_fitted()
        return self._singular_vals  # type: ignore

    @property
    def n_valid_cells(self) -> int:
        self._check_fitted()
        return self._n_valid

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _validate_input(self, residuals: xr.DataArray) -> None:
        if not isinstance(residuals, xr.DataArray):
            raise TypeError(
                f"Expected xr.DataArray, got {type(residuals).__name__}."
            )
        if "time" not in residuals.dims:
            raise ValueError(
                f"Residuals must have a 'time' dimension. "
                f"Got: {list(residuals.dims)}"
            )
        if residuals.ndim < 3:
            raise ValueError(
                f"Residuals must be 3D (time, rlat, rlon), got {residuals.ndim}D."
            )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "EOFDecomposer is not fitted. Call fit() first."
            )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a compact variance decomposition table."""
        self._check_fitted()
        cumvar = np.cumsum(self._expvar)
        print(f"\nEOFDecomposer summary")
        print(f"  n_modes retained : {self.n_modes}")
        print(f"  n_valid cells    : {self._n_valid}")
        print(f"  n_time           : {self._n_time}")
        print()
        print(f"  {'Mode':>5}  {'Var (%)':>8}  {'Cumul (%)':>10}  "
              f"{'Singular val':>13}")
        print(f"  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*13}")
        for i in range(self.n_modes):
            marker = " ← suggest_n_modes(0.95)" \
                if abs(cumvar[i] - 0.95) < 0.02 else ""
            print(f"  {i+1:>5}  {self._expvar[i]*100:>7.2f}%"
                  f"  {cumvar[i]*100:>9.2f}%"
                  f"  {self._singular_vals[i]:>13.4f}"
                  f"{marker}")
        print()
        n95 = self.suggest_n_modes(0.95)
        n80 = self.suggest_n_modes(0.80)
        print(f"  Modes to explain 80%: {n80}")
        print(f"  Modes to explain 95%: {n95}")

    def __repr__(self) -> str:
        if not self._is_fitted:
            return f"EOFDecomposer(n_modes={self.n_modes}, fitted=False)"
        cumv = self._expvar[:self.n_modes].sum()
        return (
            f"EOFDecomposer("
            f"n_modes={self.n_modes}, "
            f"n_valid={self._n_valid}, "
            f"cumvar={cumv:.1%})"
        )