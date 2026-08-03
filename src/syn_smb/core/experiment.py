"""
experiment.py
=============
Configuration dataclass for synthetic SMB generation experiments.

An Experiment holds all parameters needed to produce one synthetic ensemble:
generation length, ensemble size, random seed, and optional band-selective
PSD scaling. It replaces the ad-hoc band_factors dict used in the original
pipeline, making experiments self-documenting and reproducible.

Usage
-----
# Use a preset:
exp = Experiment.annual_enhanced(factor=10.0)

# Or build a custom experiment:
exp = Experiment(
    name="decadal_sweep",
    n_years=1000,
    n_members=30,
    seed=42,
    band_scales=[(8.0, 20.0, 5.0)],
    description="5x decadal variance, custom configuration",
)

# Pass directly to SpectralSynthesizer:
ensemble = ss.synthesize(
    n_years=exp.n_years,
    n_members=exp.n_members,
    band_scales=exp.band_scales,
    rng=exp.rng,
)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# Standard period band definitions (years) — consistent with AGU 2025 presentation
_ANNUAL_BAND:  tuple[float, float] = (0.8, 1.5)    # ~1 year period
_DECADAL_BAND: tuple[float, float] = (8.0, 20.0)   # ~8–20 year periods


@dataclass
class Experiment:
    """
    Configuration for one synthetic SMB generation experiment.

    Holds all parameters needed to produce a reproducible ensemble:
    generation length, ensemble size, random seed, and optional
    band-selective PSD scaling factors.

    Parameters
    ----------
    n_years : int
        Length of each synthetic series in years. Default 1000.
    n_members : int
        Number of ensemble members. Default 30.
    seed : int
        Random seed for the numpy Generator. Using the same seed with
        the same Experiment produces identical output. Default 0.
    band_scales : list of (pmin, pmax, factor) or None
        PSD scaling within specified period bands. Each tuple specifies:
          pmin  — shortest period in the band (years)
          pmax  — longest period in the band (years)
          factor — multiplicative scale applied to PSD in [1/pmax, 1/pmin]
        A factor > 1 amplifies variance in the band; < 1 suppresses it.
        None means no scaling (baseline experiment). Default None.
    name : str
        Short identifier used in filenames and plot titles. Default "baseline".
    description : str
        Longer human-readable description for documentation. Default "".

    Examples
    --------
    >>> exp = Experiment.baseline()
    >>> exp = Experiment.annual_enhanced(factor=10.0)
    >>> exp = Experiment.decadal_suppressed(factor=0.1, n_members=50)
    >>> exp = Experiment(name="custom", band_scales=[(5.0, 15.0, 3.0)])
    """

    n_years:     int                                     = 1000
    n_members:   int                                     = 30
    seed:        int                                     = 0
    band_scales: list[tuple[float, float, float]] | None = None
    name:        str                                     = "baseline"
    description: str                                     = ""
    seasonal_scale: float                                = 1.0
    # seasonal_scale is a VARIANCE factor on the deterministic seasonal cycle
    # (default 1.0 = observed climatology). It is applied at reconstruction as
    # an amplitude multiplier sqrt(seasonal_scale); see seasonal_amplitude_factor.
    # Values > 1 amplify the annual-timescale forcing (a stronger seasonal
    # cycle); it is mean-safe because the monthly anomalies sum to zero.

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self.n_years <= 0:
            raise ValueError(f"n_years must be positive, got {self.n_years}.")
        if self.n_members <= 0:
            raise ValueError(f"n_members must be positive, got {self.n_members}.")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}.")

        if self.band_scales is not None:
            if not isinstance(self.band_scales, list):
                raise TypeError(
                    f"band_scales must be a list of (pmin, pmax, factor) tuples, "
                    f"got {type(self.band_scales).__name__}."
                )
            for i, entry in enumerate(self.band_scales):
                if len(entry) != 3:
                    raise ValueError(
                        f"band_scales[{i}] must be a 3-tuple (pmin, pmax, factor), "
                        f"got length {len(entry)}."
                    )
                pmin, pmax, factor = entry
                if pmin <= 0 or pmax <= 0:
                    raise ValueError(
                        f"band_scales[{i}]: period bounds must be positive, "
                        f"got ({pmin}, {pmax})."
                    )
                if pmin >= pmax:
                    raise ValueError(
                        f"band_scales[{i}]: pmin must be strictly less than pmax, "
                        f"got pmin={pmin}, pmax={pmax}."
                    )
                if factor < 0:
                    raise ValueError(
                        f"band_scales[{i}]: scale factor must be non-negative, "
                        f"got {factor}. Use 0.0 to fully suppress a band."
                    )

        if self.seasonal_scale < 0:
            raise ValueError(
                f"seasonal_scale must be non-negative, got {self.seasonal_scale}. "
                f"Use 1.0 for the observed seasonal cycle, 0.0 to remove it."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rng(self) -> np.random.Generator:
        """
        Return a seeded numpy Generator for this experiment.

        A new Generator is created each time this property is accessed,
        always starting from self.seed. This guarantees that calling
        exp.rng twice produces the same sequence — no shared state.
        """
        return np.random.default_rng(self.seed)

    @property
    def n_months(self) -> int:
        """Total number of monthly time steps (n_years × 12)."""
        return self.n_years * 12

    @property
    def has_band_scaling(self) -> bool:
        """True if any band scaling is applied."""
        return self.band_scales is not None and len(self.band_scales) > 0

    @property
    def seasonal_amplitude_factor(self) -> float:
        """
        Amplitude multiplier for the deterministic seasonal cycle.

        seasonal_scale is a variance factor; the seasonal cycle enters the
        reconstruction as sqrt(seasonal_scale) * s(m). Pass this value to
        Preprocessor.inverse_transform(..., seasonal_amp_scale=...).
        """
        return float(np.sqrt(self.seasonal_scale))

    # ------------------------------------------------------------------
    # Preset constructors
    # ------------------------------------------------------------------

    @classmethod
    def baseline(
        cls,
        n_years: int = 1000,
        n_members: int = 30,
        seed: int = 0,
    ) -> Experiment:
        """
        Baseline experiment: observed spectral structure, no band scaling.
        All variance comes from phase randomisation of the observed PSD.
        """
        return cls(
            n_years=n_years,
            n_members=n_members,
            seed=seed,
            band_scales=None,
            name="baseline",
            description="No band scaling — observed spectral structure preserved.",
        )

    @classmethod
    def annual_enhanced(
        cls,
        factor: float = 10.0,
        n_years: int = 1000,
        n_members: int = 30,
        seed: int = 0,
    ) -> Experiment:
        """
        Enhanced annual variability: PSD in the annual band scaled up by factor.
        Corresponds to the 'Annual Enhanced' experiment in Markovsky et al. (AGU 2025).
        Annual band: 0.8–1.5 year periods.
        """
        pmin, pmax = _ANNUAL_BAND
        return cls(
            n_years=n_years,
            n_members=n_members,
            seed=seed,
            band_scales=[(pmin, pmax, factor)],
            seasonal_scale=factor,
            name=f"annual_enhanced_{factor}x",
            description=(
                f"Annual band ({pmin}–{pmax} yr) PSD scaled by {factor}x and "
                f"seasonal-cycle variance scaled by {factor}x. Amplifies the "
                f"annual-timescale forcing (stochastic band + deterministic cycle)."
            ),
        )

    @classmethod
    def annual_suppressed(
        cls,
        factor: float = 0.1,
        n_years: int = 1000,
        n_members: int = 30,
        seed: int = 0,
    ) -> Experiment:
        """
        Suppressed annual variability: PSD in the annual band scaled down by factor.
        Annual band: 0.8–1.5 year periods.
        """
        pmin, pmax = _ANNUAL_BAND
        return cls(
            n_years=n_years,
            n_members=n_members,
            seed=seed,
            band_scales=[(pmin, pmax, factor)],
            seasonal_scale=factor,
            name=f"annual_suppressed_{factor}x",
            description=(
                f"Annual band ({pmin}–{pmax} yr) PSD scaled by {factor}x and "
                f"seasonal-cycle variance scaled by {factor}x. Suppresses the "
                f"annual-timescale forcing (stochastic band + deterministic cycle)."
            ),
        )

    @classmethod
    def decadal_enhanced(
        cls,
        factor: float = 10.0,
        n_years: int = 1000,
        n_members: int = 30,
        seed: int = 0,
    ) -> Experiment:
        """
        Enhanced decadal variability: PSD in the decadal band scaled up by factor.
        Corresponds to the 'Decadal Enhanced' experiment in Markovsky et al. (AGU 2025).
        Decadal band: 8–20 year periods.
        """
        pmin, pmax = _DECADAL_BAND
        return cls(
            n_years=n_years,
            n_members=n_members,
            seed=seed,
            band_scales=[(pmin, pmax, factor)],
            name=f"decadal_enhanced_{factor}x",
            description=(
                f"Decadal band ({pmin}–{pmax} yr) PSD scaled by {factor}x. "
                f"Variance in the decadal band amplified by {factor}x."
            ),
        )

    @classmethod
    def decadal_suppressed(
        cls,
        factor: float = 0.1,
        n_years: int = 1000,
        n_members: int = 30,
        seed: int = 0,
    ) -> Experiment:
        """
        Suppressed decadal variability: PSD in the decadal band scaled down by factor.
        Decadal band: 8–20 year periods.
        """
        pmin, pmax = _DECADAL_BAND
        return cls(
            n_years=n_years,
            n_members=n_members,
            seed=seed,
            band_scales=[(pmin, pmax, factor)],
            name=f"decadal_suppressed_{factor}x",
            description=(
                f"Decadal band ({pmin}–{pmax} yr) PSD scaled by {factor}x. "
                f"Variance in the decadal band reduced by {1/factor:.0f}x."
            ),
        )

    @classmethod
    def standard_suite(
        cls,
        n_years: int = 1000,
        n_members: int = 30,
        seed: int = 0,
    ) -> list[Experiment]:
        """
        Return the standard set of five experiments used in Markovsky et al.

        Includes baseline, annual enhanced/suppressed, and decadal
        enhanced/suppressed — the full experiment matrix from the AGU 2025
        presentation. All experiments share the same n_years, n_members,
        and seed for direct comparison.

        Returns
        -------
        experiments : list of Experiment
            [baseline, annual_enhanced, annual_suppressed,
             decadal_enhanced, decadal_suppressed]
        """
        kwargs = dict(n_years=n_years, n_members=n_members, seed=seed)
        return [
            cls.baseline(**kwargs),
            cls.annual_enhanced(factor=10.0, **kwargs),
            cls.annual_suppressed(factor=0.1, **kwargs),
            cls.decadal_enhanced(factor=10.0, **kwargs),
            cls.decadal_suppressed(factor=0.1, **kwargs),
        ]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise to a plain dictionary.
        Suitable for saving alongside NetCDF ensemble output as metadata.
        """
        return {
            "name":        self.name,
            "description": self.description,
            "n_years":     self.n_years,
            "n_members":   self.n_members,
            "seed":        self.seed,
            "band_scales": self.band_scales,
            "seasonal_scale": self.seasonal_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Experiment:
        """
        Reconstruct an Experiment from a dictionary produced by to_dict().
        """
        band_scales = d.get("band_scales")
        if band_scales is not None:
            band_scales = [tuple(b) for b in band_scales]
        return cls(
            name        = d.get("name", "unnamed"),
            description = d.get("description", ""),
            n_years     = d["n_years"],
            n_members   = d["n_members"],
            seed        = d["seed"],
            band_scales = band_scales,
            seasonal_scale = d.get("seasonal_scale", 1.0),
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable summary of this experiment."""
        print(f"Experiment: {self.name}")
        if self.description:
            print(f"  {self.description}")
        print(f"  n_years:   {self.n_years}")
        print(f"  n_members: {self.n_members}")
        print(f"  seed:      {self.seed}")
        if self.band_scales:
            print(f"  band_scales:")
            for pmin, pmax, factor in self.band_scales:
                fmin = 1.0 / pmax
                fmax = 1.0 / pmin
                print(
                    f"    periods [{pmin}–{pmax} yr]  "
                    f"freqs [{fmin:.3f}–{fmax:.3f} cycles/yr]  "
                    f"factor={factor}x"
                )
        else:
            print("  band_scales: None (baseline)")
        if self.seasonal_scale != 1.0:
            print(f"  seasonal_scale: {self.seasonal_scale} "
                  f"(amplitude x{self.seasonal_amplitude_factor:.3f})")

    def __repr__(self) -> str:
        bands = f", band_scales={self.band_scales}" if self.band_scales else ""
        return (
            f"Experiment(name='{self.name}', n_years={self.n_years}, "
            f"n_members={self.n_members}, seed={self.seed}{bands})"
        )