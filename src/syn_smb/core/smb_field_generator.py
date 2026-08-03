"""
smb_field_generator.py
=======================
Orchestrator for the full 2D synthetic SMB generation pipeline.

Coordinates SpatialPreprocessor → EOFDecomposer → per-PC GaussianTransform
+ SpectralSynthesizer → reconstruction into a synthetic 2D SMB field.

The 1D components (GaussianTransform, SpectralSynthesizer, Experiment) are
reused unchanged from syn_smb.core. The per-PC loop in generate() is the
only new logic; everything else is orchestration.

Typical usage
-------------
    from syn_smb.core.smb_field_loader     import SMBFieldLoader
    from syn_smb.core.smb_field_generator  import SMBFieldGenerator
    from syn_smb.core.experiment           import Experiment

    # Fit
    gen = SMBFieldGenerator.from_path(
        racmo_path  = "./data/RACMO2.4p1_ANT11.nc",
        shp_path    = "./data/IceBoundaries_Antarctica_V2.shp",
        basin_name  = "PineIsland",
    )

    # Generate
    ds = gen.generate(Experiment.baseline())

    # Or the full standard suite
    suite = gen.generate_suite(Experiment.standard_suite())
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from .spatial_preprocessor import SpatialPreprocessor
from .eof_decomposer        import EOFDecomposer
from .gaussianize           import GaussianTransform
from .spectral              import SpectralSynthesizer
from .experiment            import Experiment


class SMBFieldGenerator:
    """
    2D synthetic SMB field generator.

    Fit on a basin-masked RACMO spatial field (time, rlat, rlon),
    generate arbitrarily long synthetic ensembles with the same
    spatial covariance structure and temporal spectral content.

    Parameters
    ----------
    n_modes : int
        Number of EOF modes to retain. Use EOFDecomposer.suggest_n_modes()
        after a trial fit to choose this. Default 10.
    nperseg : int
        Welch segment length (months) passed to SpectralSynthesizer.
        Default 60 (5 years), same as 1D SMBGenerator.

    Attributes (after fit())
    ------------------------
    preprocessor   : SpatialPreprocessor
    eof            : EOFDecomposer
    gt_per_pc      : list of GaussianTransform, length n_modes
    ss_per_pc      : list of SpectralSynthesizer, length n_modes
    field_mean     : xr.DataArray (rlat, rlon) — basin mean SMB per grid cell
    basin_mask     : xr.DataArray (rlat, rlon) — True inside basin
    n_obs          : int — number of training time steps
    """

    def __init__(self, n_modes: int = 10, nperseg: int = 60) -> None:
        self.n_modes  = n_modes
        self.nperseg  = nperseg

        self.preprocessor: SpatialPreprocessor | None = None
        self.eof:           EOFDecomposer        | None = None
        self.gt_per_pc:     list[GaussianTransform]    = []
        self.ss_per_pc:     list[SpectralSynthesizer]  = []

        self._field_mean:  xr.DataArray | None = None
        self._lat:         xr.DataArray | None = None
        self._lon:         xr.DataArray | None = None
        self._basin_mask:  xr.DataArray | None = None
        self._spatial_dims: list[str]          = []
        self._n_obs:        int                = 0
        self._is_fitted:    bool               = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        field: xr.DataArray,
        lat:   xr.DataArray | None = None,
        lon:   xr.DataArray | None = None,
    ) -> SMBFieldGenerator:
        """
        Fit the full 2D pipeline on a basin-masked SMB field.

        Parameters
        ----------
        field : xr.DataArray, shape (time, rlat, rlon)
            Output of SMBFieldLoader.load(). NaN outside the basin.
        lat : xr.DataArray (rlat, rlon) or None
            Latitude for EOF area-weighting. Strongly recommended.

        Returns
        -------
        self
        """
        print("SMBFieldGenerator.fit()")
        print(f"  Field shape : {dict(field.sizes)}")

        self._spatial_dims = [d for d in field.dims if d != "time"]
        self._n_obs        = field.sizes["time"]

        # Infer basin mask from NaN pattern and store coordinates for plotting
        self._basin_mask = xr.DataArray(
            np.all(np.isfinite(field.values), axis=0),
            dims=self._spatial_dims,
        )
        self._lat = lat
        self._lon = lon
        print(f"  Valid cells : {int(self._basin_mask.sum())}")

        # ── Step 2: SpatialPreprocessor ──────────────────────────────
        print("  [1/4] SpatialPreprocessor...")
        self.preprocessor = SpatialPreprocessor()
        residuals = self.preprocessor.fit_transform(field)

        # Store spatial mean for final reconstruction
        self._field_mean = self.preprocessor.field_mean

        # ── Step 3: EOFDecomposer ────────────────────────────────────
        print(f"  [2/4] EOFDecomposer (n_modes={self.n_modes})...")
        self.eof = EOFDecomposer(n_modes=self.n_modes)
        pcs      = self.eof.fit_transform(residuals, lat=lat)
        # pcs: (n_obs, n_modes) — training PC time series

        # Update n_modes in case EOFDecomposer capped it
        self.n_modes = self.eof.n_modes

        # ── Step 4: per-PC GaussianTransform + SpectralSynthesizer ───
        print(f"  [3/4] Per-PC 1D pipeline ({self.n_modes} modes)...")
        self.gt_per_pc = []
        self.ss_per_pc = []

        time_coord = field.time

        for i in range(self.n_modes):
            pc_da = xr.DataArray(
                pcs[:, i],
                coords={"time": time_coord},
                dims=["time"],
                name=f"PC{i+1}",
            )

            # Skip near-zero-variance modes (below 0.1% of total variance)
            pc_var = float(pc_da.var())
            if pc_var < 1e-10:
                warnings.warn(
                    f"PC {i+1} has near-zero variance ({pc_var:.2e}). "
                    f"Replacing with a trivial transform."
                )
                self.gt_per_pc.append(None)
                self.ss_per_pc.append(None)
                continue

            # Gaussian rank transform — transform() fits and applies in one call
            gt   = GaussianTransform()
            g_pc = gt.transform(pc_da)
            self.gt_per_pc.append(gt)

            # Spectral synthesis
            ss = SpectralSynthesizer(nperseg=self.nperseg)
            ss.fit(g_pc)
            self.ss_per_pc.append(ss)

        print(f"  [4/4] Fit complete.")
        self.eof.summary()

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, experiment: Experiment) -> xr.Dataset:
        """
        Generate a synthetic SMB field ensemble.

        Parameters
        ----------
        experiment : Experiment
            Controls n_years, n_members, seed, and band_scales.
            Passed unchanged to each SpectralSynthesizer.synthesize().

        Returns
        -------
        ds : xr.Dataset
            Variables:
              smb_syn   (member, time, rlat, rlon) — synthetic SMB, m w.e.
              basin_mask (rlat, rlon)               — True inside basin
        """
        self._check_fitted()

        n_time_syn = experiment.n_years * 12
        time_syn   = xr.cftime_range(
            "1979-01", periods=n_time_syn, freq="MS"
        )

        members = []

        for m in range(experiment.n_members):
            pcs_syn = np.zeros((n_time_syn, self.n_modes))

            for i in range(self.n_modes):
                gt = self.gt_per_pc[i]
                ss = self.ss_per_pc[i]

                if gt is None or ss is None:
                    pcs_syn[:, i] = 0.0
                    continue

                # Each (member, mode) pair gets an independent RNG seeded
                # by a combination of experiment seed + member + mode index,
                # ensuring all members and all modes are statistically
                # independent while remaining reproducible.
                rng = np.random.default_rng(
                    experiment.seed + m + i * (experiment.n_members + 1)
                )

                # Synthesise one member in Gaussian space
                g_syn_raw = ss.synthesize(
                    n_years     = experiment.n_years,
                    n_members   = 1,
                    band_scales = experiment.band_scales,
                    rng         = rng,
                )
                # synthesize() may return (1, T) or (T,) — flatten to 1D
                g_syn = np.asarray(g_syn_raw).flatten()

                # Back-transform to physical PC units
                pc_syn = gt.inverse_transform(g_syn)
                pcs_syn[:, i] = np.asarray(pc_syn).flatten()

            # Reconstruct spatial residual field from synthetic PCs
            residuals_syn = self.eof.inverse_transform(
                pcs_syn, time_coord=time_syn
            )

            # Restore seasonal cycle (add_mean=False — we add mean below).
            # seasonal_amp_scale couples the annual-band experiments: a
            # stronger seasonal cycle is applied as an amplitude factor on the
            # per-cell climatology. getattr fallback = 1.0 so this is safe even
            # if the 1-D Experiment seasonal-scaling fix is not yet applied
            # (Experiment.seasonal_amplitude_factor == sqrt(seasonal_scale)).
            seas_amp = getattr(experiment, "seasonal_amplitude_factor", 1.0)
            field_syn = self.preprocessor.inverse_transform(
                residuals_syn, add_mean=False,
                seasonal_amp_scale=seas_amp,
            )

            # Restore per-grid-cell climatological mean
            field_syn = field_syn + self._field_mean

            members.append(field_syn)

        # Stack members → (member, time, rlat, rlon)
        smb_syn = xr.concat(members, dim="member")
        smb_syn = smb_syn.assign_coords(
            member=np.arange(experiment.n_members)
        )
        smb_syn.attrs.update({
            "long_name":    "Synthetic basin SMB field",
            "units":        "m w.e.",
            "experiment":   experiment.name,
            "n_modes":      self.n_modes,
            "n_years":      experiment.n_years,
            "n_members":    experiment.n_members,
        })

        ds = xr.Dataset(
            {
                "smb_syn":    smb_syn,
                "basin_mask": self._basin_mask,
            }
        )
        ds.attrs["experiment"] = experiment.name
        return ds

    def generate_suite(
        self,
        suite: list[Experiment] | None = None,
    ) -> dict[str, xr.Dataset]:
        """
        Generate the full standard suite of experiments.

        Parameters
        ----------
        suite : list of Experiment or None
            Experiments to run. Defaults to Experiment.standard_suite().

        Returns
        -------
        datasets : dict[str, xr.Dataset]
            {experiment.name: xr.Dataset}
        """
        if suite is None:
            suite = Experiment.standard_suite()

        datasets = {}
        for exp in suite:
            print(f"\nGenerating: {exp.name}")
            datasets[exp.name] = self.generate(exp)

        return datasets

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_path(
        cls,
        racmo_path:  str,
        shp_path:    str,
        basin_name:  str,
        name_col:    str = "NAME",
        n_modes:     int = 10,
        nperseg:     int = 60,
    ) -> SMBFieldGenerator:
        """
        Fit directly from RACMO + shapefile paths.

        Parameters
        ----------
        racmo_path  : path to full-domain RACMO NetCDF
        shp_path    : path to basin shapefile
        basin_name  : value of name_col in the shapefile
        name_col    : shapefile column with basin names (default 'NAME')
        n_modes     : number of EOF modes to retain
        nperseg     : Welch segment length for SpectralSynthesizer

        Returns
        -------
        gen : SMBFieldGenerator (fitted)
        """
        from .smb_field_loader import SMBFieldLoader

        loader = SMBFieldLoader(
            racmo_path = racmo_path,
            shp_path   = shp_path,
            basin_name = basin_name,
            name_col   = name_col,
        )
        field = loader.load()
        gen   = cls(n_modes=n_modes, nperseg=nperseg)
        gen.fit(field, lat=loader.lat, lon=loader.lon)
        return gen

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
    def basin_mask(self) -> xr.DataArray:
        self._check_fitted()
        return self._basin_mask  # type: ignore

    @property
    def n_obs(self) -> int:
        return self._n_obs

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "SMBFieldGenerator is not fitted. "
                "Call fit() or from_path() first."
            )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._is_fitted:
            return (
                f"SMBFieldGenerator("
                f"n_modes={self.n_modes}, nperseg={self.nperseg}, "
                f"fitted=False)"
            )
        n_valid = int(self._basin_mask.sum()) if self._basin_mask is not None else "?"
        return (
            f"SMBFieldGenerator("
            f"n_modes={self.n_modes}, "
            f"n_obs={self._n_obs}, "
            f"n_valid_cells={n_valid})"
        )