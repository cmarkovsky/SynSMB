"""
generator.py
============
SMBGenerator: orchestrates the complete 1D synthetic SMB pipeline.

Wraps Preprocessor, GaussianTransform, SpectralSynthesizer, and
Experiment into a single fit/generate interface. This is the primary
user-facing object for synthetic SMB production.

Pipeline (forward — fit time):
    SMB DataArray
        → Preprocessor.fit_transform()    removes trend + seasonal cycle
        → GaussianTransform.fit()          learns empirical distribution
        → GaussianTransform.transform()    maps residuals to N(0,1)
        → SpectralSynthesizer.fit()        estimates PSD

Pipeline (inverse — generate time):
    SpectralSynthesizer.synthesize()      phase-randomised N(0,1) ensemble
        → GaussianTransform.inverse_transform()  back to zero-mean residuals
        → + smb_mean                             restores observed mean
        → Preprocessor.inverse_transform()       restores seasonal cycle
        → xr.Dataset                             ready for Icepack
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import xarray as xr

from syn_smb.core.preprocessor import Preprocessor
from syn_smb.core.gaussianize import GaussianTransform
from syn_smb.core.spectral import SpectralSynthesizer
from syn_smb.core.experiment import Experiment
from syn_smb.core.data_loader import SMBDataLoader


class SMBGenerator:
    """
    Orchestrates the complete 1D synthetic SMB generation pipeline.

    Parameters
    ----------
    nperseg : int
        Welch segment length for PSD estimation. Default 60 (5-year
        segments for monthly data).
    dt_years : float
        Sampling interval in years. Default 1/12 for monthly data.
    remove_trend : bool
        Whether Preprocessor removes the linear trend. Default True.
    remove_seasonal : bool
        Whether Preprocessor removes the monthly climatology. Default True.

    Attributes
    ----------
    preprocessor : Preprocessor
    gaussian_transform : GaussianTransform
    spectral_synthesizer : SpectralSynthesizer
    smb_mean : float
        Observed time-mean SMB. Added back once in generate().
    smb_start_year : int
        First year of the observed record. Used to anchor synthetic
        time coordinates.
    n_obs : int
        Number of observed time steps.
    is_fitted : bool

    Examples
    --------
    >>> gen = SMBGenerator()
    >>> gen.fit(smb_dataarray)
    >>> dataset = gen.generate(Experiment.baseline())
    >>> datasets = gen.generate_suite(Experiment.standard_suite())
    """

    def __init__(
        self,
        nperseg: int = 60,
        dt_years: float = 1 / 12,
        remove_trend: bool = True,
        remove_seasonal: bool = True,
    ) -> None:
        self.nperseg = nperseg
        self.dt_years = dt_years
        self.remove_trend = remove_trend
        self.remove_seasonal = remove_seasonal

        # Component objects — instantiated at fit() time
        self.preprocessor: Preprocessor | None = None
        self.gaussian_transform: GaussianTransform | None = None
        self.spectral_synthesizer: SpectralSynthesizer | None = None

        # Fitted scalars
        self.smb_mean: float | None = None
        self.smb_start_year: int | None = None
        self.n_obs: int | None = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "SMBGenerator is not fitted. Call fit() or from_path() first."
            )

    def _make_synthetic_time(self, n_months: int) -> xr.CFTimeIndex:
        """
        Build a monthly cftime coordinate of length n_months, starting
        from the first year of the observed record.
        """
        return xr.cftime_range(
            start=f"{self.smb_start_year}-01-01",
            periods=n_months,
            freq="MS",
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def freqs(self) -> np.ndarray | None:
        """Frequency grid from SpectralSynthesizer (cycles/year)."""
        return self.spectral_synthesizer.freqs if self.is_fitted else None

    @property
    def psd(self) -> np.ndarray | None:
        """Fitted PSD from SpectralSynthesizer."""
        return self.spectral_synthesizer.psd if self.is_fitted else None

    @property
    def psd_ci_lower(self) -> np.ndarray | None:
        return self.spectral_synthesizer.psd_ci_lower if self.is_fitted else None

    @property
    def psd_ci_upper(self) -> np.ndarray | None:
        return self.spectral_synthesizer.psd_ci_upper if self.is_fitted else None

    @property
    def seasonal_cycle(self) -> xr.DataArray | None:
        """Monthly climatology from Preprocessor."""
        return self.preprocessor.seasonal_cycle if self.is_fitted else None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, smb: xr.DataArray) -> SMBGenerator:
        """
        Fit the full pipeline to an observed SMB time series.

        Steps (in order):
          1. Store observed mean — the only place it lives
          2. Preprocessor.fit_transform() — remove trend + seasonal cycle
          3. GaussianTransform.fit() — learn empirical distribution
          4. GaussianTransform.transform() — map residuals to N(0,1)
          5. SpectralSynthesizer.fit() — estimate PSD + confidence intervals

        Parameters
        ----------
        smb : xr.DataArray
            Observed monthly SMB with a 'time' coordinate. Units should be
            m w.e. a⁻¹ (SMBDataLoader handles unit conversion).

        Returns
        -------
        self : SMBGenerator
        """
        if 'time' not in smb.coords:
            raise ValueError("smb must have a 'time' coordinate.")

        self.n_obs = smb.sizes['time']

        # Step 1: store the observed mean — added back in generate(), once only
        self.smb_mean = float(smb.mean())
        self.smb_start_year = int(str(smb['time'].values[0])[:4])

        # Step 2: preprocessing — removes trend and seasonal cycle
        self.preprocessor = Preprocessor(
            remove_trend=self.remove_trend,
            remove_seasonal=self.remove_seasonal,
        )
        residuals = self.preprocessor.fit_transform(smb)

        # Step 3 + 4: Gaussianize residuals
        self.gaussian_transform = GaussianTransform()
        self.gaussian_transform.fit(residuals)
        g_resid = self.gaussian_transform.transform(residuals)

        # Step 5: fit spectral model to Gaussianized residuals
        g_vals = g_resid.values if isinstance(g_resid, xr.DataArray) else g_resid
        self.spectral_synthesizer = SpectralSynthesizer(
            nperseg=self.nperseg,
            dt_years=self.dt_years,
        )
        self.spectral_synthesizer.fit(g_vals)

        # Store residuals for use in validate() — lightweight for 1D scalar
        self._residuals = residuals

        self.is_fitted = True
        return self

    @classmethod
    def from_path(
        cls,
        path: str,
        var: str = "smbgl",
        **kwargs,
    ) -> SMBGenerator:
        """
        Convenience: load data from a NetCDF file and fit in one call.

        Parameters
        ----------
        path : str
            Path to RACMO NetCDF file.
        var : str
            SMB variable name. Default 'smbgl'.
        **kwargs
            Passed to SMBGenerator.__init__() (nperseg, dt_years, etc.).

        Returns
        -------
        gen : SMBGenerator
            Fitted generator ready for generate().
        """
        smb = SMBDataLoader(path, var=var).load()
        gen = cls(**kwargs)
        gen.fit(smb)
        return gen

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, experiment: Experiment) -> xr.Dataset:
        """
        Generate a synthetic SMB ensemble for a given Experiment.

        Runs the inverse pipeline for all ensemble members:
          1. SpectralSynthesizer.synthesize() — phase-randomised Gaussian ensemble
          2. GaussianTransform.inverse_transform() — back to zero-mean residuals
          3. Add smb_mean — observed mean restored once, here, only here
          4. Preprocessor.inverse_transform() — seasonal cycle restored

        Parameters
        ----------
        experiment : Experiment
            Configuration object defining n_years, n_members, seed, and
            optional band_scales.

        Returns
        -------
        ds : xr.Dataset
            Dataset with dimensions (member, time) and variables:
              smb_syn   — full reconstructed SMB in m w.e. a⁻¹
              resid_syn — zero-mean stochastic residuals before mean + seasonal
              g_syn     — Gaussianized residuals (unit variance, for spectral QC)
        """
        self._check_fitted()

        N_syn = experiment.n_months
        syn_time = self._make_synthetic_time(N_syn)

        # --- Phase-randomise in Gaussian space for all members at once ---
        ensemble_g = self.spectral_synthesizer.synthesize(
            n_years=experiment.n_years,
            n_members=experiment.n_members,
            band_scales=experiment.band_scales,
            rng=experiment.rng,
        )
        # ensemble_g shape: (n_members, N_syn)

        smb_list   = []
        resid_list = []
        g_list     = []

        for i in range(experiment.n_members):
            g_da = xr.DataArray(
                ensemble_g[i],
                coords={"time": syn_time},
                dims=["time"],
            )

            # Gaussian → zero-mean stochastic residual
            resid = self.gaussian_transform.inverse_transform(g_da)

            # Add observed mean back — once, here, only here
            smb_with_mean = resid + self.smb_mean

            # Restore seasonal cycle (trend deliberately excluded for
            # synthetic series — see Preprocessor methodology notes)
            smb_full = self.preprocessor.inverse_transform(
                smb_with_mean, add_trend=False
            )

            smb_list.append(smb_full.expand_dims(member=[i]))
            resid_list.append(resid.expand_dims(member=[i]))
            g_list.append(g_da.expand_dims(member=[i]))

        ds = xr.Dataset(
            {
                "smb_syn":   xr.concat(smb_list,   dim="member"),
                "resid_syn": xr.concat(resid_list, dim="member"),
                "g_syn":     xr.concat(g_list,     dim="member"),
            }
        )

        # Store experiment metadata as Dataset attributes
        ds.attrs.update({
            "experiment_name":        experiment.name,
            "experiment_description": experiment.description,
            "n_years":                experiment.n_years,
            "n_members":              experiment.n_members,
            "seed":                   experiment.seed,
            "band_scales":            str(experiment.band_scales),
            "smb_mean":               self.smb_mean,
            "smb_start_year":         self.smb_start_year,
        })

        return ds

    def generate_suite(
        self,
        suite: list[Experiment],
    ) -> dict[str, xr.Dataset]:
        """
        Run a list of Experiments, returning one Dataset per experiment.

        Parameters
        ----------
        suite : list of Experiment
            Typically from Experiment.standard_suite().

        Returns
        -------
        results : dict[str, xr.Dataset]
            Keys are experiment names; values are the ensemble Datasets.
        """
        self._check_fitted()
        return {exp.name: self.generate(exp) for exp in suite}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, verbose: bool = True) -> dict:
        """
        Run component validation checks on the fitted pipeline.

        Checks:
          1. GaussianTransform — round-trip fidelity and normality
          2. SpectralSynthesizer — PSD fidelity against confidence intervals
          3. Stationarity — ADF test on preprocessed residuals

        Parameters
        ----------
        verbose : bool
            If True, print formatted results.

        Returns
        -------
        results : dict
            Keys: 'gaussian_transform', 'spectral_synthesizer',
                  'stationarity', 'passed'.
        """
        self._check_fitted()

        if verbose:
            print("SMBGenerator validation")
            print("=" * 40)

        # --- 1. GaussianTransform ---
        if verbose:
            print("\n[1] GaussianTransform:")
        gt_results = self.gaussian_transform.validate(
            self._residuals, verbose=verbose
        )

        # --- 2. SpectralSynthesizer ---
        if verbose:
            print("\n[2] SpectralSynthesizer:")
        ss_results = self.spectral_synthesizer.validate(
            n_check=100, verbose=verbose
        )

        # --- 3. Stationarity of preprocessed residuals ---
        if verbose:
            print("\n[3] Stationarity (ADF test on preprocessed residuals):")
        stat_results = self.preprocessor.check_stationarity(
            self._residuals, verbose=verbose
        )

        overall_passed = (
            gt_results["passed"]
            and ss_results["passed"]
            and stat_results["is_stationary"]
        )

        results = {
            "gaussian_transform":   gt_results,
            "spectral_synthesizer": ss_results,
            "stationarity":         stat_results,
            "passed":               overall_passed,
        }

        if verbose:
            print(f"\nOverall: {'PASSED ✓' if overall_passed else 'FAILED ✗'}")

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        dataset: xr.Dataset,
        filepath: str,
        experiment: Experiment | None = None,
    ) -> None:
        """
        Save an ensemble Dataset to a NetCDF file.

        Experiment metadata is stored as global attributes so the file
        is self-describing. If experiment is None, only the data is saved.

        Parameters
        ----------
        dataset : xr.Dataset
            Output of generate().
        filepath : str
            Output path. Should end in .nc.
        experiment : Experiment or None
            If provided, metadata is written as global attributes.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if experiment is not None:
            attrs = experiment.to_dict()
            # NetCDF attributes do not support multi-dimensional arrays,
            # lists of tuples, or None. Convert band_scales to a string
            # representation in all cases. str(None) → "None", which is
            # readable and recoverable via ast.literal_eval() if needed.
            attrs["band_scales"] = str(attrs["band_scales"])
            dataset.attrs.update(attrs)

        dataset.to_netcdf(path)
        print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summarize(self) -> None:
        """Print a structured summary of the fitted pipeline."""
        if not self.is_fitted:
            print("SMBGenerator not fitted.")
            return

        print("SMBGenerator summary")
        print(f"  n_obs:         {self.n_obs}")
        print(f"  smb_mean:      {self.smb_mean:.5f}  m w.e. a⁻¹")
        print(f"  start_year:    {self.smb_start_year}")
        print(f"  nperseg:       {self.nperseg}")
        print(f"  remove_trend:  {self.remove_trend}")
        print(f"  remove_seasonal: {self.remove_seasonal}")

        if self.preprocessor and self.preprocessor.is_fitted:
            print("\n  Preprocessor:")
            self.preprocessor.summarize()

        if self.spectral_synthesizer and self.spectral_synthesizer.is_fitted:
            ss = self.spectral_synthesizer
            print(f"\n  SpectralSynthesizer:")
            print(f"    n_segments:  {ss.n_segments}")
            print(f"    CI width:    {(ss.psd_ci_upper[1:] / ss.psd_ci_lower[1:]).mean():.2f}x")
            annual_idx = int(np.argmin(np.abs(ss.freqs - 1.0)))
            decadal_idx = int(np.argmin(np.abs(ss.freqs - 0.1)))
            print(f"    PSD at f=1.0 (annual):  {ss.psd[annual_idx]:.4e}")
            print(f"    PSD at f=0.1 (decadal): {ss.psd[decadal_idx]:.4e}")

    def __repr__(self) -> str:
        if self.is_fitted:
            return (
                f"SMBGenerator(fitted=True, n_obs={self.n_obs}, "
                f"smb_mean={self.smb_mean:.4f}, "
                f"start_year={self.smb_start_year})"
            )
        return "SMBGenerator(fitted=False)"