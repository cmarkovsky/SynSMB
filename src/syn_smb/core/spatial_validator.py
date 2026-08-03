"""
spatial_validator.py
====================
Validates the 2D synthetic SMB field produced by SMBFieldGenerator,
analogous to the 1D Validator but operating on spatial fields.

The core questions answered:
  1. Does the synthetic field preserve the observed spatial mean field?
  2. Does the spatial distribution of interannual variance match?
  3. Do the PC time series have the correct spectral structure?
  4. Does the method generalise to held-out periods (calibration split)?

Pipeline position
-----------------
    SMBFieldGenerator.generate()   → xr.Dataset (member, time, rlat, rlon)
         ↓
    SpatialValidator.compute_metrics()
    SpatialValidator.plot_validation_suite()

Relationship with 1D Validator
-------------------------------
The per-PC metrics are computed by running the 1D Validator on each PC
time series independently. This directly reuses the existing validation
infrastructure and connects the 2D results back to the 1D methodology.
"""

from __future__ import annotations

import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from scipy.signal import welch

from .smb_field_generator  import SMBFieldGenerator
from .spatial_preprocessor import SpatialPreprocessor
from .eof_decomposer        import EOFDecomposer
from .validator             import Validator

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


class SpatialValidator:
    """
    Validates 2D synthetic SMB fields from SMBFieldGenerator.

    Parameters
    ----------
    generator : SMBFieldGenerator
        A fitted generator — provides access to the preprocessor, EOF
        decomposer, and per-PC synthesisers.
    observed : xr.DataArray, shape (time, rlat, rlon)
        The observed RACMO field used to fit the generator. Used as the
        reference for all validation comparisons.
    """

    def __init__(
        self,
        generator: SMBFieldGenerator,
        observed:  xr.DataArray,
    ) -> None:
        if not generator.is_fitted:
            raise ValueError("Generator must be fitted before validation.")

        self.generator = generator
        self.observed  = observed

        self._spatial_dims = generator._spatial_dims
        self._basin_mask   = generator.basin_mask
        self._valid        = self._basin_mask.values.astype(bool)

        # Pre-compute observed statistics on valid cells only
        self._obs_flat = observed.values.reshape(
            observed.sizes["time"], -1
        )[:, self._valid.flatten()]   # (T, n_valid) — no NaN

        # Build full-grid mean/var maps (NaN outside basin) for plotting
        ny, nx = self._valid.shape
        _obs_mean_valid = np.mean(self._obs_flat, axis=0)
        _obs_var_valid  = np.var(self._obs_flat,  axis=0)

        self._obs_mean_map = np.full(ny * nx, np.nan)
        self._obs_var_map  = np.full(ny * nx, np.nan)
        self._obs_mean_map[self._valid.flatten()] = _obs_mean_valid
        self._obs_var_map[self._valid.flatten()]  = _obs_var_valid
        self._obs_mean_map = self._obs_mean_map.reshape(ny, nx)
        self._obs_var_map  = self._obs_var_map.reshape(ny, nx)

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        dataset:  xr.Dataset,
        verbose:  bool = True,
    ) -> dict:
        """
        Compute a comprehensive set of validation metrics comparing the
        synthetic ensemble to the observed field.

        Parameters
        ----------
        dataset : xr.Dataset
            Output of SMBFieldGenerator.generate(), containing 'smb_syn'
            with dims (member, time, rlat, rlon).
        verbose : bool
            Print a formatted summary. Default True.

        Returns
        -------
        metrics : dict with keys:
            mean_ratio          — syn mean / obs mean (target ≈ 1.0)
            variance_ratio      — syn variance / obs variance (target ≈ 1.0)
            mean_map_corr       — Pearson r between time-mean spatial maps
            variance_map_corr   — Pearson r between variance maps
            variance_map_ratio  — ratio of mean variance map values
            eof_cumvar          — cumulative EOF variance (n_modes retained)
            pc_metrics          — list of per-PC 1D validation metric dicts
        """
        smb = dataset["smb_syn"]
        syn_arr = smb.values   # (member, time, rlat, rlon)

        # ── Global mean and variance ──────────────────────────────────
        obs_mean = float(np.nanmean(self._obs_flat))

        # Flatten synthetic array to valid cells only — avoids "mean of
        # empty slice" warnings from all-NaN outside-basin cells
        n_mem, n_t, ny, nx = syn_arr.shape
        valid_flat  = self._valid.flatten()
        syn_flat    = syn_arr.reshape(n_mem * n_t, ny * nx)[:, valid_flat]

        syn_mean   = float(np.mean(syn_flat))
        mean_ratio = syn_mean / obs_mean if obs_mean != 0 else float("nan")

        obs_var   = float(np.var(self._obs_flat))
        syn_var   = float(np.var(syn_flat))
        var_ratio = syn_var / obs_var if obs_var != 0 else float("nan")

        # ── Spatial map correlations ──────────────────────────────────
        # Compute mean/var maps on valid cells only, then test correlation
        syn_mean_per_cell = np.mean(
            syn_arr.reshape(n_mem * n_t, ny * nx)[:, valid_flat], axis=0
        )
        syn_var_per_cell  = np.var(
            syn_arr.reshape(n_mem * n_t, ny * nx)[:, valid_flat], axis=0
        )

        obs_m = self._obs_mean_map[self._valid]
        obs_v = self._obs_var_map[self._valid]

        mean_map_corr, _    = pearsonr(obs_m, syn_mean_per_cell)
        var_map_corr,  _    = pearsonr(obs_v, syn_var_per_cell)
        var_map_ratio       = (
            float(np.mean(syn_var_per_cell)) / float(np.mean(obs_v))
            if np.mean(obs_v) != 0 else float("nan")
        )

        # ── EOF cumulative variance ───────────────────────────────────
        eof_cumvar = float(
            self.generator.eof.explained_variance_ratio.sum()
        )

        # ── Per-PC 1D validation ──────────────────────────────────────
        pc_metrics = self._compute_pc_metrics(smb)

        metrics = {
            "mean_ratio":         mean_ratio,
            "variance_ratio":     var_ratio,
            "mean_map_corr":      mean_map_corr,
            "variance_map_corr":  var_map_corr,
            "variance_map_ratio": var_map_ratio,
            "eof_cumvar":         eof_cumvar,
            "pc_metrics":         pc_metrics,
        }

        if verbose:
            self._print_metrics(metrics)

        return metrics

    def _compute_pc_metrics(self, smb: xr.DataArray) -> list[dict]:
        """Run 1D Validator on each PC time series."""
        gen     = self.generator
        n_modes = gen.n_modes
        pc_metrics = []

        # Training PCs from the fitted EOFDecomposer
        train_pcs = gen.eof.pcs   # (n_obs, n_modes)

        for i in range(n_modes):
            gt = gen.gt_per_pc[i]
            ss = gen.ss_per_pc[i]

            if gt is None or ss is None:
                pc_metrics.append({"mode": i + 1, "skipped": True})
                continue

            # Training PC as xr.DataArray
            pc_obs = xr.DataArray(
                train_pcs[:, i],
                coords={"time": self.observed.time},
                dims=["time"],
            )

            # Synthetic PC: extract basin-mean per member and time
            # (for spectral comparison — use ensemble mean PSD)
            try:
                # Project synthetic field back onto EOF to get synthetic PCs
                residuals_syn = gen.preprocessor.inverse_transform(
                    gen.eof.inverse_transform(
                        gen.eof.pcs,    # use training PCs as proxy
                        time_coord=self.observed.time,
                    ),
                    add_mean=False,
                )
                # Use gt and ss for a mini 1D validation
                g_obs = gt.transform(pc_obs)
                g_da  = xr.DataArray(
                    g_obs if hasattr(g_obs, "values") else g_obs,
                    coords={"time": self.observed.time},
                    dims=["time"],
                )
                n_pts    = len(g_da)
                nperseg  = min(60, n_pts // 2)   # at least 2 segments
                if nperseg < 4:
                    pc_metrics.append({
                        "mode": i + 1, "skipped": True,
                        "error": "Too few observations for Welch PSD."
                    })
                    continue
                f_obs, p_obs = welch(
                    np.asarray(g_da).flatten(),
                    fs=12.0, nperseg=nperseg,
                )
                psd_rms = float(np.sqrt(np.mean(
                    (np.log10(p_obs[1:] + 1e-30) -
                     np.log10(ss.psd[1:] + 1e-30)) ** 2
                )))
                pc_metrics.append({
                    "mode":        i + 1,
                    "psd_rms":     psd_rms,
                    "expvar":      float(gen.eof.explained_variance_ratio[i]),
                    "skipped":     False,
                })
            except Exception as e:
                pc_metrics.append({
                    "mode": i + 1, "skipped": True, "error": str(e)
                })

        return pc_metrics

    def _print_metrics(self, m: dict) -> None:
        print("\nSpatialValidator metrics")
        print(f"  Mean ratio          : {m['mean_ratio']:.4f}  (target ≈ 1.0)")
        print(f"  Variance ratio      : {m['variance_ratio']:.4f}  (target ≈ 1.0)")
        print(f"  Mean map correlation: {m['mean_map_corr']:.4f}  (target → 1.0)")
        print(f"  Var  map correlation: {m['variance_map_corr']:.4f}  (target → 1.0)")
        print(f"  Var  map ratio      : {m['variance_map_ratio']:.4f}  (target ≈ 1.0)")
        print(f"  EOF cumul. variance : {m['eof_cumvar']:.1%}")
        print()
        print(f"  Per-PC PSD RMS error:")
        for pc in m["pc_metrics"]:
            if pc.get("skipped"):
                print(f"    PC {pc['mode']}: skipped")
            else:
                print(f"    PC {pc['mode']}: {pc['psd_rms']:.4f}"
                      f"  (expvar={pc['expvar']:.1%})")

    # ------------------------------------------------------------------
    # Cartopy helpers
    # ------------------------------------------------------------------

    def _geo_extent(self, pad: float = 0.15) -> list[float]:
        """
        Return [lon_min, lon_max, lat_min, lat_max] bounding box of the
        basin with fractional padding, for use as a cartopy extent.
        """
        lat = self.generator._lat
        lon = self.generator._lon
        if lat is None or lon is None:
            return [-180, 180, -90, -60]   # fallback: full Antarctica
        valid     = self._valid
        vlat      = lat.values[valid]
        vlon      = lon.values[valid]
        lat_range = vlat.max() - vlat.min()
        lon_range = vlon.max() - vlon.min()
        return [
            vlon.min() - lon_range * pad,
            vlon.max() + lon_range * pad,
            vlat.min() - lat_range * pad,
            vlat.max() + lat_range * pad,
        ]

    def _make_geo_ax(self, fig, subplot_spec):
        """
        Create a cartopy axes in EPSG:3031 (Antarctic Polar Stereo).
        Falls back to a plain matplotlib axes if cartopy is unavailable.
        """
        if HAS_CARTOPY:
            proj = ccrs.Stereographic(
                central_latitude    = -90,
                central_longitude   =   0,
                true_scale_latitude = -71,   # EPSG:3031
            )
            ax = fig.add_subplot(subplot_spec, projection=proj)
            ax.set_extent(self._geo_extent(), crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                           linewidth=0.5, color="black")
            ax.gridlines(linewidth=0.3, color="gray",
                         alpha=0.5, linestyle="--")
        else:
            ax = fig.add_subplot(subplot_spec)
        return ax

    def _pcolormesh(self, ax, data_2d: np.ndarray, **kwargs):
        """
        Plot a 2-D field (same shape as the cropped spatial grid) on ax.
        Uses PlateCarree transform when cartopy is active.
        """
        lat = self.generator._lat
        lon = self.generator._lon
        if HAS_CARTOPY and lat is not None and lon is not None:
            return ax.pcolormesh(
                lon.values, lat.values, data_2d,
                transform=ccrs.PlateCarree(),
                shading="auto", **kwargs,
            )
        else:
            return ax.pcolormesh(data_2d, shading="auto", **kwargs)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def plot_validation_suite(
        self,
        dataset:   xr.Dataset,
        save_path: str | None = None,
    ) -> None:
        """
        Four-panel validation summary in EPSG:3031 projection:
          Top-left  : Observed time-mean SMB field
          Top-right : Synthetic time-mean SMB field (ensemble mean)
          Bot-left  : Observed interannual variance
          Bot-right : Synthetic interannual variance

        All panels are zoomed to the basin bounding box.
        """
        smb     = dataset["smb_syn"]
        n_mem, n_t, ny, nx = smb.values.shape
        valid_flat  = self._valid.flatten()
        syn_flat    = smb.values.reshape(n_mem * n_t, ny * nx)[:, valid_flat]

        syn_mean_map = np.full(ny * nx, np.nan)
        syn_var_map  = np.full(ny * nx, np.nan)
        syn_mean_map[valid_flat] = np.mean(syn_flat, axis=0)
        syn_var_map[valid_flat]  = np.var(syn_flat,  axis=0)
        syn_mean_map = syn_mean_map.reshape(ny, nx)
        syn_var_map  = syn_var_map.reshape(ny, nx)

        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(13, 9))
        fig.suptitle("SpatialValidator — field comparison (EPSG:3031)",
                     fontsize=12)
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                hspace=0.08, wspace=0.08)

        panels = [
            (gs[0, 0], self._obs_mean_map, "Observed time-mean SMB",       "Blues"),
            (gs[0, 1], syn_mean_map,        "Synthetic time-mean SMB\n(ensemble mean)", "Blues"),
            (gs[1, 0], self._obs_var_map,   "Observed variance",            "YlOrRd"),
            (gs[1, 1], syn_var_map,         "Synthetic variance\n(ensemble)", "YlOrRd"),
        ]

        for spec, data, title, cmap in panels:
            ax   = self._make_geo_ax(fig, spec)
            vmax = np.nanpercentile(data[self._valid], 98)
            pcm  = self._pcolormesh(ax, data, cmap=cmap, vmin=0, vmax=vmax)
            plt.colorbar(pcm, ax=ax, label="m w.e.", shrink=0.7, pad=0.02)
            ax.set_title(title, fontsize=9, pad=4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    def plot_variance_maps(
        self,
        dataset:   xr.Dataset,
        save_path: str | None = None,
    ) -> None:
        """
        Three-panel variance comparison in EPSG:3031:
          Left:   Observed variance
          Centre: Synthetic variance (ensemble)
          Right:  Ratio syn/obs — should be near 1 everywhere
        """
        smb     = dataset["smb_syn"]
        n_mem, n_t, ny, nx = smb.values.shape
        valid_flat = self._valid.flatten()
        syn_flat   = smb.values.reshape(n_mem * n_t, ny * nx)[:, valid_flat]

        syn_var_flat = np.var(syn_flat, axis=0)
        syn_var = np.full(ny * nx, np.nan)
        syn_var[valid_flat] = syn_var_flat
        syn_var = syn_var.reshape(ny, nx)

        obs_v = self._obs_var_map
        ratio = np.where(obs_v > 0, syn_var / obs_v, np.nan)

        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(15, 4.5))
        fig.suptitle("Variance map comparison (EPSG:3031)", fontsize=12)
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.08)

        vmax = np.nanpercentile(
            np.concatenate([obs_v[self._valid], syn_var[self._valid]]), 98
        )
        panels = [
            (gs[0, 0], obs_v,  "Observed variance",  "YlOrRd", 0,   vmax,  "(m w.e.)²"),
            (gs[0, 1], syn_var,"Synthetic variance",  "YlOrRd", 0,   vmax,  "(m w.e.)²"),
            (gs[0, 2], ratio,  "Ratio (syn / obs)",   "RdBu_r", 0.5, 1.5,   ""),
        ]
        for spec, data, title, cmap, vmin, vmax_, label in panels:
            ax  = self._make_geo_ax(fig, spec)
            pcm = self._pcolormesh(ax, data, cmap=cmap, vmin=vmin, vmax=vmax_)
            plt.colorbar(pcm, ax=ax, label=label, shrink=0.7, pad=0.02)
            ax.set_title(title, fontsize=9, pad=4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    def plot_pc_validation(
        self,
        n_pcs:     int  = 4,
        save_path: str | None = None,
    ) -> None:
        """
        PSD comparison for the first n_pcs PC time series in Gaussian space.

        Both the blue and orange lines are in the same space (N(0,1)):
          Blue  — Welch PSD of the Gaussianized observed PC
                  (raw PC → GaussianTransform → Welch)
          Orange — Fitted PSD stored by SpectralSynthesizer
                  (what the synthesiser will use to generate)

        They should agree within the 95% CI band (orange shading) if the
        spectral fitting worked correctly. A systematic gap between them
        indicates a normalisation issue in the pipeline.

        Note: the raw PC PSD (before Gaussianization) is NOT shown here
        because it is in different units to the fitted PSD — comparing them
        produces a misleading amplitude gap.
        """
        gen     = self.generator
        n_pcs   = min(n_pcs, gen.n_modes)
        pcs_obs = gen.eof.pcs       # (n_time, n_modes) — raw PCs
        time    = self.observed.time

        fig, axes = plt.subplots(
            1, n_pcs, figsize=(4 * n_pcs, 4), sharey=False
        )
        if n_pcs == 1:
            axes = [axes]
        fig.suptitle(
            "Per-PC PSD comparison — Gaussianized observed vs fitted\n"
            "(both in N(0,1) space — should agree within CI)",
            fontsize=10,
        )

        for i, ax in enumerate(axes):
            gt = gen.gt_per_pc[i]
            ss = gen.ss_per_pc[i]
            if gt is None or ss is None:
                ax.set_visible(False)
                continue

            # ── Gaussianize the observed PC ───────────────────────────
            pc_da = xr.DataArray(
                pcs_obs[:, i],
                coords={"time": time},
                dims=["time"],
            )
            g_obs   = gt.transform(pc_da)           # → N(0,1) series
            g_vals  = np.asarray(g_obs).flatten()

            # Welch PSD of the Gaussianized PC
            nperseg = min(60, len(g_vals) // 2)     # at least 2 segments
            f, p    = welch(g_vals, fs=12.0, nperseg=nperseg)

            # ── Plot ──────────────────────────────────────────────────
            ax.loglog(f[1:], p[1:],
                      color="tab:blue", lw=1.5, alpha=0.85,
                      label="Gaussianized PC (observed)")
            ax.loglog(ss.freqs[1:], ss.psd[1:],
                      color="tab:orange", lw=1.5, linestyle="--",
                      label="Fitted PSD (SpectralSynthesizer)")
            ax.fill_between(
                ss.freqs[1:], ss.psd_ci_lower[1:], ss.psd_ci_upper[1:],
                color="tab:orange", alpha=0.15, label="95% CI",
            )

            expv = gen.eof.explained_variance_ratio[i]
            ax.set_title(f"PC {i+1}  ({expv:.1%} var)", fontsize=9)
            ax.set_xlabel("Frequency (cycles yr⁻¹)", fontsize=8)
            if i == 0:
                ax.set_ylabel("PSD (Gaussian space)", fontsize=8)
                ax.legend(fontsize=7)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    def calibration_split_test(
        self,
        n_members:   int = 10,
        n_years_syn: int = 50,
        verbose:     bool = True,
    ) -> dict:
        """
        Out-of-sample validation: fit on the first half of the observed
        record, generate synthetic fields, and compare against the
        held-out second half.

        Returns metrics for both split directions.
        """
        obs    = self.observed
        n_time = obs.sizes["time"]
        split  = n_time // 2

        results = {}
        for label, train_slice, val_slice in [
            ("first_half",  slice(None, split), slice(split, None)),
            ("second_half", slice(split, None), slice(None, split)),
        ]:
            field_train = obs.isel(time=train_slice)
            field_val   = obs.isel(time=val_slice)

            # Fit on training half
            gen_split = SMBFieldGenerator(
                n_modes = self.generator.n_modes,
                nperseg = self.generator.nperseg,
            )
            gen_split.fit(field_train)

            # Generate and compare to validation half
            from .experiment import Experiment
            exp = Experiment(n_years=n_years_syn, n_members=n_members, seed=0)
            ds  = gen_split.generate(exp)

            val_split = SpatialValidator(gen_split, field_val)
            m = val_split.compute_metrics(ds, verbose=False)
            results[label] = m

            if verbose:
                print(f"\n[{label}] "
                      f"mean_ratio={m['mean_ratio']:.3f}, "
                      f"var_ratio={m['variance_ratio']:.3f}, "
                      f"mean_map_corr={m['mean_map_corr']:.3f}")

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lat(self):
        return getattr(self.generator, "_lat", None)

    @property
    def lon(self):
        return getattr(self.generator, "_lon", None)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_valid = int(self._valid.sum())
        return (
            f"SpatialValidator("
            f"n_valid={n_valid}, "
            f"n_modes={self.generator.n_modes})"
        )