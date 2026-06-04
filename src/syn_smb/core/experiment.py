"""
Experiment configuration for synthetic SMB perturbation experiments.

Classes
-------
Experiment
    Configuration for a single perturbation experiment.
ExperimentSuite
    Ordered collection of experiments for batch generation.

Built-in Presets
----------------
BASELINE
    Observed noise at all frequencies.
MEAN_ONLY
    No noise — deterministic mean forcing only.
ANNUAL_ENHANCED
    Annual band amplitude scaled by 10x.
ANNUAL_SUPPRESSED
    Annual band amplitude scaled by 0.1x.
DECADAL_ENHANCED
    Decadal band amplitude scaled by 10x.
DECADAL_SUPPRESSED
    Decadal band amplitude scaled by 0.1x.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Experiment:
    """
    Configuration for a single synthetic SMB perturbation experiment.

    Defines how the amplitude spectrum of each PC is scaled before
    phase randomisation. Frequencies not covered by any band retain a
    scale of 1.0 (observed amplitude). The DC component (zero frequency)
    is always zeroed regardless of band definitions, ensuring the
    synthetic series remains zero-mean.

    Parameters
    ----------
    name : str
        Human-readable label used in output Dataset attributes and
        ExperimentSuite keys. Default: ``"baseline"``.
    band_scales : dict, optional
        Maps band label to a ``(f_low, f_high, scale_factor)`` tuple.
        ``f_low`` and ``f_high`` are in cycles per year (inclusive).
        ``scale_factor`` multiplies the amplitude spectrum, so power
        scales by ``scale_factor ** 2``. Example::

            {"annual": (0.8, 1.2, 10.0),
             "decadal": (0.05, 0.15, 0.1)}

        Defaults to an empty dict (no scaling applied).
    include_noise : bool
        If ``False``, all variability is suppressed and the generator
        produces a deterministic mean-only forcing. Default: ``True``.

    Examples
    --------
    Standard baseline — no perturbation:

    >>> exp = Experiment()

    Amplify the annual band by 10x, suppress the decadal band:

    >>> exp = Experiment(
    ...     name="annual_up_decadal_down",
    ...     band_scales={
    ...         "annual":  (0.8, 1.2,  10.0),
    ...         "decadal": (0.05, 0.15, 0.1),
    ...     },
    ... )

    Mean-only forcing (no noise):

    >>> exp = Experiment(name="mean_only", include_noise=False)
    """

    name: str = "baseline"
    band_scales: dict = field(default_factory=dict)
    include_noise: bool = True

    def build_scale_profile(self, freqs: np.ndarray) -> np.ndarray:
        """
        Build a per-frequency amplitude scale profile.

        Parameters
        ----------
        freqs : np.ndarray, shape (n_freqs,)
            One-sided frequency array in cycles per year, as returned by
            ``numpy.fft.rfftfreq`` multiplied by the sampling frequency.
            The first element must be 0.0 (DC component).

        Returns
        -------
        np.ndarray, shape (n_freqs,)
            Multiplicative scale factor at each frequency.
            DC component is always 0.0.
            Frequencies outside all defined bands are 1.0.
            Frequencies inside a band take that band's scale factor.

        Notes
        -----
        When two bands overlap, the last matching band in
        ``band_scales`` takes precedence (dict iteration order).
        """
        if not self.include_noise:
            return np.zeros_like(freqs, dtype=float)

        profile = np.ones_like(freqs, dtype=float)
        profile[0] = 0.0  # DC always zero — protects zero-mean property

        for _band_name, (f_low, f_high, scale) in self.band_scales.items():
            mask = (freqs >= f_low) & (freqs <= f_high)
            profile[mask] = scale

        return profile


# ---------------------------------------------------------------------------
# Built-in experiment presets
# ---------------------------------------------------------------------------

BASELINE = Experiment(
    name="baseline",
    band_scales={},
)
"""Baseline experiment: observed amplitude at all frequencies."""

MEAN_ONLY = Experiment(
    name="mean_only",
    include_noise=False,
)
"""Mean-only experiment: deterministic forcing, no variability."""

ANNUAL_ENHANCED = Experiment(
    name="annual_enhanced_10x",
    band_scales={"annual": (0.8, 1.2, 10.0)},
)
"""Annual band amplitude scaled by 10x (power by 100x)."""

ANNUAL_SUPPRESSED = Experiment(
    name="annual_suppressed_0.1x",
    band_scales={"annual": (0.8, 1.2, 0.1)},
)
"""Annual band amplitude scaled by 0.1x (power by 0.01x)."""

DECADAL_ENHANCED = Experiment(
    name="decadal_enhanced_10x",
    band_scales={"decadal": (0.05, 0.15, 10.0)},
)
"""Decadal band amplitude scaled by 10x (power by 100x)."""

DECADAL_SUPPRESSED = Experiment(
    name="decadal_suppressed_0.1x",
    band_scales={"decadal": (0.05, 0.15, 0.1)},
)
"""Decadal band amplitude scaled by 0.1x (power by 0.01x)."""


# ---------------------------------------------------------------------------
# ExperimentSuite
# ---------------------------------------------------------------------------

class ExperimentSuite:
    """
    Ordered collection of :class:`Experiment` objects for batch generation.

    Iterating over an ``ExperimentSuite`` yields each :class:`Experiment`
    in insertion order.

    Parameters
    ----------
    experiments : list of Experiment
        Experiments to include.

    Examples
    --------
    >>> from syn_smb import ExperimentSuite, BASELINE, ANNUAL_ENHANCED
    >>> suite = ExperimentSuite([BASELINE, ANNUAL_ENHANCED])
    >>> for exp in suite:
    ...     print(exp.name)
    baseline
    annual_enhanced_10x

    Use the built-in default suite of six standard experiments:

    >>> suite = ExperimentSuite.DEFAULT
    """

    #: Default suite containing all six built-in presets.
    DEFAULT: ExperimentSuite  # assigned after class definition

    def __init__(self, experiments: list[Experiment]) -> None:
        self.experiments = list(experiments)

    def add(self, experiment: Experiment) -> None:
        """
        Append an experiment to the suite.

        Parameters
        ----------
        experiment : Experiment
            Experiment to add.
        """
        self.experiments.append(experiment)

    def __iter__(self):
        return iter(self.experiments)

    def __len__(self) -> int:
        return len(self.experiments)

    def __repr__(self) -> str:
        names = [e.name for e in self.experiments]
        return f"ExperimentSuite({names})"


ExperimentSuite.DEFAULT = ExperimentSuite([
    BASELINE,
    MEAN_ONLY,
    ANNUAL_ENHANCED,
    ANNUAL_SUPPRESSED,
    DECADAL_ENHANCED,
    DECADAL_SUPPRESSED,
])