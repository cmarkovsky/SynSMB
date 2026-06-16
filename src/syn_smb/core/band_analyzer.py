"""
band_analyser.py
================
Data-driven identification of dominant frequency bands in basin-mean SMB.

Operates on a fitted SpectralSynthesizer and answers the question:
"Which frequency bands carry the most variance, and should therefore be
the target of controlled Experiment band-scaling?"

The approach is the frequency-domain analogue of PCA:
  - PCA partitions total variance across spatial/temporal modes
  - BandAnalyser partitions total variance across period bands
  - The cumulative variance curve (like a PCA scree plot) shows the
    minimum set of bands needed to explain a target fraction of variance

Typical usage
-------------
    from syn_smb import SMBGenerator, BandAnalyser

    gen = SMBGenerator.from_path("./data/PIG_smb.nc")
    ba  = BandAnalyser(gen.spectral_synthesizer)

    # Automatic peak detection
    peaks = ba.find_peaks()
    for p in peaks:
        print(p)

    # Variance decomposition into default period bands
    df = ba.variance_by_band()

    # Suggested Experiment.band_scales for the top-2 bands
    scales = ba.suggest_band_scales(n_bands=2)

    # Diagnostic figure
    ba.plot(save_path="pig_band_analysis.png")

    # Multi-basin comparison
    from band_analyser import compare_basin_bands
    compare_basin_bands(results)   # results from multi_basin_run()
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import find_peaks as sp_find_peaks
from scipy.ndimage import uniform_filter1d


# ======================================================================
# BandResult — one detected spectral peak / band
# ======================================================================

@dataclass
class BandResult:
    """
    A single detected frequency band and its variance statistics.

    Attributes
    ----------
    label       : human-readable label (e.g. 'annual', 'decadal', 'peak_3')
    pmin, pmax  : period bounds of the band (years)
    period_peak : period at the PSD maximum within the band (years)
    freq_peak   : frequency at the PSD maximum (cycles yr⁻¹)
    psd_peak    : PSD value at the peak
    variance    : absolute variance explained by the band
    variance_frac : fraction of total SMB variance in this band (0–1)
    rank        : rank by variance_frac (1 = largest)
    """
    label:         str
    pmin:          float
    pmax:          float
    period_peak:   float
    freq_peak:     float
    psd_peak:      float
    variance:      float
    variance_frac: float
    rank:          int = 0

    def as_band_scale(
        self,
        factor: float = 10.0,
    ) -> tuple[float, float, float]:
        """
        Return an ``Experiment.band_scales`` tuple (pmin, pmax, factor).
        """
        return (self.pmin, self.pmax, factor)

    def __str__(self) -> str:
        return (
            f"BandResult(rank={self.rank}, label='{self.label}', "
            f"peak={self.period_peak:.1f} yr [{self.pmin:.1f}–{self.pmax:.1f} yr], "
            f"var_frac={self.variance_frac:.1%})"
        )


# ======================================================================
# BandAnalyser
# ======================================================================

# Default period bin edges (years) for the fixed variance decomposition.
# Chosen to bracket physically meaningful timescales in Antarctic SMB:
#   sub-annual  : T < 0.8 yr   — intra-seasonal noise
#   annual      : 0.8 – 1.5 yr — seasonal accumulation variability
#   inter-annual: 1.5 – 8  yr  — ENSO, SAM interannual modes
#   decadal     : 8  – 20  yr  — decadal SAM, PDO teleconnections
#   multi-decadal: > 20 yr     — long-term trend residuals
DEFAULT_PERIOD_EDGES = [0.1, 0.8, 1.5, 8.0, 20.0, 100.0]
DEFAULT_BAND_LABELS  = [
    "sub-annual (<0.8 yr)",
    "annual (0.8–1.5 yr)",
    "inter-annual (1.5–8 yr)",
    "decadal (8–20 yr)",
    "multi-decadal (>20 yr)",
]

# Colorblind-safe band colours (Wong 2011)
_BAND_COLORS = ["#56B4E9", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


class BandAnalyser:
    """
    Data-driven identification of dominant SMB frequency bands.

    Parameters
    ----------
    synthesizer : SpectralSynthesizer
        Must be fitted (synthesizer.fit() already called). All analysis
        operates on synthesizer.freqs and synthesizer.psd.
    basin_name : str or None
        Used for plot titles and labels.
    """

    def __init__(self, synthesizer, basin_name: str | None = None):
        self.synth      = synthesizer
        self.basin_name = basin_name or "basin"

        # Exclude DC component (index 0, f=0)
        self._freqs   = synthesizer.freqs[1:].copy()
        self._psd     = synthesizer.psd[1:].copy()
        self._psd_lo  = synthesizer.psd_ci_lower[1:].copy()
        self._psd_hi  = synthesizer.psd_ci_upper[1:].copy()
        self._periods = 1.0 / self._freqs      # convert to years

        # Variance spectrum: variance per unit frequency (PSD × Δf)
        df = np.diff(synthesizer.freqs)
        df = np.append(df, df[-1])
        self._df        = df[1:]
        self._var_spec  = self._psd * self._df   # variance per bin
        self._total_var = float(self._var_spec.sum())

    # ------------------------------------------------------------------
    # 1. Fixed-band variance decomposition
    # ------------------------------------------------------------------

    def variance_by_band(
        self,
        period_edges: list[float] | None = None,
        labels: list[str] | None = None,
    ) -> list[dict]:
        """
        Partition total variance into fixed period bands and compute the
        fraction of total variance in each.

        This is the direct analogue of a PCA variance table, with
        period bands instead of spatial modes.

        Parameters
        ----------
        period_edges : list of float or None
            Monotonically decreasing period boundaries in years.
            Default: ``DEFAULT_PERIOD_EDGES``.
        labels : list of str or None
            Band labels. Must have length ``len(period_edges) - 1``.

        Returns
        -------
        bands : list of dict
            Each dict has keys: label, pmin, pmax, variance, variance_frac,
            cumulative_frac.
        """
        if period_edges is None:
            period_edges = DEFAULT_PERIOD_EDGES
        if labels is None:
            labels = DEFAULT_BAND_LABELS

        # Sort edges high→low (large period = low frequency)
        edges = sorted(period_edges, reverse=True)

        bands = []
        cumvar = 0.0
        for i in range(len(edges) - 1):
            pmax = edges[i]
            pmin = edges[i + 1]
            fmin = 1.0 / pmax
            fmax = 1.0 / pmin
            mask = (self._freqs >= fmin) & (self._freqs <= fmax)
            var  = float(self._var_spec[mask].sum())
            cumvar += var
            label = labels[i] if i < len(labels) else f"{pmin:.1f}–{pmax:.1f} yr"
            bands.append({
                "label":          label,
                "pmin":           pmin,
                "pmax":           pmax,
                "variance":       var,
                "variance_frac":  var / self._total_var if self._total_var > 0 else 0.0,
                "cumulative_frac":cumvar / self._total_var if self._total_var > 0 else 0.0,
            })

        return bands

    def explained_variance_at(
        self,
        threshold: float = 0.80,
        period_edges: list[float] | None = None,
    ) -> tuple[int, float]:
        """
        How many of the default bands (ordered by period, not variance)
        are needed to cumulatively explain ``threshold`` fraction of
        total variance?

        Returns (n_bands, cumulative_fraction).
        """
        bands = self.variance_by_band(period_edges)
        # Sort by variance fraction descending (like PCA scree)
        sorted_bands = sorted(bands, key=lambda b: b["variance_frac"], reverse=True)
        cumvar = 0.0
        for n, b in enumerate(sorted_bands, start=1):
            cumvar += b["variance_frac"]
            if cumvar >= threshold:
                return n, cumvar
        return len(sorted_bands), cumvar

    # ------------------------------------------------------------------
    # 2. Automatic peak detection
    # ------------------------------------------------------------------

    def find_peaks(
        self,
        prominence: float = 0.3,
        min_period: float = 0.5,
        max_period: float = 50.0,
        smooth_window: int = 3,
    ) -> list[BandResult]:
        """
        Automatically detect peaks in the PSD and return a BandResult
        for each, with period bounds set by the half-power bandwidth.

        This is the data-driven, basin-specific alternative to manually
        specifying annual/decadal bands.

        Parameters
        ----------
        prominence : float
            Minimum peak prominence in log10(PSD) units. Controls how
            distinct a peak must be above its surroundings. Default 0.3
            (~2× above background). Increase to find only the sharpest
            peaks; decrease to capture broader spectral features.
        min_period : float
            Ignore peaks at periods shorter than this (years). Default 0.5.
        max_period : float
            Ignore peaks at periods longer than this (years). Default 50.
        smooth_window : int
            Width of pre-smoothing filter (in frequency bins) applied
            before peak detection to reduce noise. Default 3.

        Returns
        -------
        peaks : list of BandResult
            Sorted by variance_frac descending (most important first).
        """
        # Work in log10(PSD) vs log10(period) space — peaks are more
        # symmetric and prominence is scale-invariant
        log_psd     = np.log10(self._psd)
        log_periods = np.log10(self._periods)

        # Smooth to reduce noise artefacts in peak detection
        if smooth_window > 1:
            log_psd_smooth = uniform_filter1d(log_psd, size=smooth_window)
        else:
            log_psd_smooth = log_psd

        # Restrict to the requested period range
        period_mask = (
            (self._periods >= min_period)
            & (self._periods <= max_period)
        )
        if not period_mask.any():
            warnings.warn(
                f"No frequencies in period range [{min_period}, {max_period}] yr."
            )
            return []

        # Find peaks on the smoothed log-PSD within the period mask
        # Note: find_peaks works on regularly-spaced arrays — periods are
        # NOT regularly spaced, so we work in frequency-index space.
        masked_indices = np.where(period_mask)[0]
        sub_psd = log_psd_smooth[masked_indices]

        peak_locs, props = sp_find_peaks(
            sub_psd,
            prominence=prominence,
        )

        if len(peak_locs) == 0:
            warnings.warn(
                f"No peaks found with prominence ≥ {prominence}. "
                f"Try lowering the prominence threshold."
            )
            return []

        # Map back to global indices
        global_peak_idx = masked_indices[peak_locs]

        results = []
        for idx in global_peak_idx:
            peak_period = self._periods[idx]
            peak_freq   = self._freqs[idx]
            peak_psd    = self._psd[idx]
            log_peak    = log_psd[idx]

            # Half-power bandwidth: find where PSD drops by 3 dB
            # (i.e. log_psd drops by log10(2) ≈ 0.301 on each side)
            half_power = log_peak - np.log10(2)

            # Search left (toward lower frequencies = longer periods)
            left = idx
            while left > 0 and log_psd[left - 1] >= half_power:
                left -= 1

            # Search right (toward higher frequencies = shorter periods)
            right = idx
            while right < len(log_psd) - 1 and log_psd[right + 1] >= half_power:
                right += 1

            # Extend slightly to ensure the band captures the full peak
            left  = max(0, left - 1)
            right = min(len(self._freqs) - 1, right + 1)

            # Period bounds (note: higher frequency = shorter period)
            pmin = float(self._periods[right])  # higher freq → shorter period
            pmax = float(self._periods[left])   # lower  freq → longer  period

            # Variance in this band
            band_mask = (self._freqs >= self._freqs[right]) & \
                        (self._freqs <= self._freqs[left])
            var = float(self._var_spec[band_mask].sum())

            results.append(BandResult(
                label        = _auto_label(peak_period),
                pmin         = round(pmin, 2),
                pmax         = round(pmax, 2),
                period_peak  = round(peak_period, 2),
                freq_peak    = round(peak_freq, 4),
                psd_peak     = float(peak_psd),
                variance     = var,
                variance_frac= var / self._total_var if self._total_var > 0 else 0.0,
            ))

        # Sort by variance fraction and assign ranks
        results.sort(key=lambda r: r.variance_frac, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    # ------------------------------------------------------------------
    # 3. Suggest Experiment band_scales
    # ------------------------------------------------------------------

    def suggest_band_scales(
        self,
        n_bands: int = 2,
        factor_enhance:  float = 10.0,
        factor_suppress: float = 0.1,
        prominence:      float = 0.3,
        min_period:      float = 0.5,
        max_period:      float = 50.0,
    ) -> list[tuple[float, float, float]]:
        """
        Return a list of ``Experiment.band_scales`` tuples for the top
        ``n_bands`` peaks by variance fraction.

        Each tuple is (pmin, pmax, factor) compatible with::

            Experiment(band_scales=[(0.8, 1.5, 10.0), (8.0, 20.0, 10.0)])

        Both enhanced and suppressed versions are returned for each band.

        Parameters
        ----------
        n_bands : int
            Number of dominant bands to suggest. Default 2.
        factor_enhance : float
            PSD scale factor for enhanced experiments. Default 10.0.
        factor_suppress : float
            PSD scale factor for suppressed experiments. Default 0.1.

        Returns
        -------
        scales : list of (pmin, pmax, factor) tuples
        """
        peaks = self.find_peaks(
            prominence=prominence,
            min_period=min_period,
            max_period=max_period,
        )

        if not peaks:
            warnings.warn("No peaks detected. Cannot suggest band_scales.")
            return []

        top = peaks[:n_bands]
        scales = []
        for band in top:
            scales.append((band.pmin, band.pmax, factor_enhance))
            scales.append((band.pmin, band.pmax, factor_suppress))

        return scales

    # ------------------------------------------------------------------
    # 4. Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        period_edges: list[float] | None = None,
        prominence:   float = 0.3,
        save_path: str | None = None,
    ) -> None:
        """
        Three-panel diagnostic figure:

        Left:   PSD (log-log) with detected peaks marked and half-power
                bandwidths shaded. Observed 95% CI shown.
        Middle: Bar chart of variance fraction by fixed period band.
                Equivalent to a PCA variance-explained bar chart.
        Right:  Cumulative variance fraction vs log(period).
                The 50%, 80%, 95% thresholds are marked — shows the
                minimum set of bands needed to explain most variance.

        Parameters
        ----------
        period_edges : list of float or None
            Edges for the fixed-band decomposition (middle panel).
        prominence : float
            Peak detection prominence threshold (left panel).
        save_path : str or None
        """
        peaks = self.find_peaks(prominence=prominence)
        bands = self.variance_by_band(period_edges)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"SMB spectral band analysis — {self.basin_name}",
            fontsize=12,
        )

        self._plot_psd_with_peaks(axes[0], peaks)
        self._plot_variance_bars(axes[1], bands)
        self._plot_cumulative_variance(axes[2], bands)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    def _plot_psd_with_peaks(self, ax, peaks: list[BandResult]) -> None:
        """Left panel: PSD with detected peaks and bandwidths."""
        # Observed PSD with CI
        ax.loglog(self._periods, self._psd,
                  color="tab:blue", lw=2, label="Fitted PSD", zorder=4)
        ax.fill_between(self._periods, self._psd_lo, self._psd_hi,
                        color="tab:blue", alpha=0.12, label="95% CI")

        # Shade detected bands and mark peaks
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(peaks), 1)))
        for band, color in zip(peaks, colors):
            ax.axvspan(band.pmin, band.pmax,
                       alpha=0.18, color=color,
                       label=f"Peak {band.rank}: {band.period_peak:.1f} yr "
                             f"({band.variance_frac:.1%})")
            ax.axvline(band.period_peak, color=color, lw=1.5,
                       linestyle="--", alpha=0.8)
            ax.plot(band.period_peak, band.psd_peak,
                    "o", color=color, markersize=7, zorder=5)

        ax.axvline(1.0, color="gray", lw=0.8, linestyle=":", alpha=0.6)
        ax.axvline(10.0, color="gray", lw=0.8, linestyle=":", alpha=0.6)
        ax.set_xlabel("Period (years)")
        ax.set_ylabel("PSD (Gaussianized space)")
        ax.set_title("PSD with detected peaks\n(shading = half-power bandwidth)")
        ax.legend(fontsize=7, loc="lower left")
        ax.invert_xaxis()   # longer periods on the left

    def _plot_variance_bars(self, ax, bands: list[dict]) -> None:
        """Middle panel: variance fraction by fixed period band."""
        labels = [b["label"].replace(" ", "\n") for b in bands]
        fracs  = [b["variance_frac"] for b in bands]
        colors = _BAND_COLORS[:len(bands)]

        x = np.arange(len(bands))
        bars = ax.bar(x, fracs, color=colors, alpha=0.85, edgecolor="white")

        # Annotate each bar with its percentage
        for bar, f in zip(bars, fracs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{f:.1%}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Fraction of total variance")
        ax.set_ylim(0, max(fracs) * 1.25)
        ax.set_title("Variance by period band\n(analogue of PCA scree plot)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    def _plot_cumulative_variance(self, ax, bands: list[dict]) -> None:
        """Right panel: cumulative variance vs log-period."""
        # Sort by period (longest first) for a cumulative plot that reads
        # left→right from low-frequency to high-frequency
        sorted_by_period = sorted(bands, key=lambda b: b["pmax"], reverse=True)

        period_mids = [np.sqrt(b["pmin"] * b["pmax"]) for b in sorted_by_period]
        cum_fracs   = [b["cumulative_frac"] for b in sorted_by_period]

        ax.semilogx(period_mids, cum_fracs,
                    "o-", color="tab:blue", lw=2, markersize=6)

        # Threshold lines
        for threshold, color, label in [
            (0.50, "#E69F00", "50%"),
            (0.80, "#D55E00", "80%"),
            (0.95, "#CC79A7", "95%"),
        ]:
            ax.axhline(threshold, color=color, lw=1, linestyle="--", alpha=0.8,
                       label=label)

        ax.set_xlabel("Period (years, log scale)")
        ax.set_ylabel("Cumulative variance fraction")
        ax.set_title("Cumulative variance vs period\n(longer → shorter)")
        ax.legend(fontsize=8, title="Thresholds")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.05)
        ax.invert_xaxis()   # longer periods on the left

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a compact summary of peak detection and variance decomposition."""
        peaks = self.find_peaks()
        bands = self.variance_by_band()

        print(f"\nBandAnalyser — {self.basin_name}")
        print(f"  Total variance (Gaussian space): {self._total_var:.4f}")
        print()

        print("  Detected peaks (by variance, highest first):")
        if not peaks:
            print("    (none detected)")
        for p in peaks:
            print(f"    {p}")

        print()
        print("  Variance by fixed period band:")
        for b in bands:
            bar = "█" * int(b["variance_frac"] * 30)
            print(f"    {b['label']:<30} {b['variance_frac']:>6.1%}  {bar}")

        n, cum = self.explained_variance_at(0.80)
        print(f"\n  Top {n} band(s) explain {cum:.1%} of total variance.")

        print()
        print("  Suggested Experiment.band_scales (top-2 peaks, 10× / 0.1×):")
        scales = self.suggest_band_scales(n_bands=2)
        for s in scales:
            print(f"    {s}")

    def __repr__(self) -> str:
        return f"BandAnalyser(basin='{self.basin_name}', n_freqs={len(self._freqs)})"


# ======================================================================
# Multi-basin comparison
# ======================================================================

def compare_basin_bands(
    results: dict[str, dict],
    period_edges: list[float] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Compare variance decomposition across all basins from multi_basin_run().

    Produces a stacked-bar figure where each bar is a basin and the
    colour segments show how much of the total SMB variance comes from
    each period band. Directly shows whether basins differ in their
    dominant timescales.

    Parameters
    ----------
    results : dict
        Output of multi_basin_run().
    period_edges : list of float or None
        Period bin edges (years). Default: DEFAULT_PERIOD_EDGES.
    save_path : str or None
    """
    if period_edges is None:
        period_edges = DEFAULT_PERIOD_EDGES

    basins = list(results.keys())
    n_basins = len(basins)

    # Compute variance fractions for each basin
    all_bands = []
    for basin in basins:
        gen = results[basin]["generator"]
        ba  = BandAnalyser(gen.spectral_synthesizer, basin_name=basin)
        all_bands.append(ba.variance_by_band(period_edges))

    n_bands = len(all_bands[0])
    labels  = [b["label"] for b in all_bands[0]]
    colors  = _BAND_COLORS[:n_bands]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SMB variance decomposition by basin", fontsize=12)

    # ── Left: stacked bar chart ──
    ax = axes[0]
    x  = np.arange(n_basins)
    bottoms = np.zeros(n_basins)

    for band_idx in range(n_bands):
        fracs = np.array([all_bands[i][band_idx]["variance_frac"]
                          for i in range(n_basins)])
        bars = ax.bar(x, fracs, bottom=bottoms,
                      color=colors[band_idx], label=labels[band_idx],
                      alpha=0.85, edgecolor="white")
        bottoms += fracs

    ax.set_xticks(x)
    ax.set_xticklabels(basins, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of total SMB variance")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=8, loc="upper right",
              title="Period band", ncol=1)
    ax.set_title("Variance by period band (stacked)")

    # ── Right: grouped bar for the two most scientifically relevant bands
    # (annual and decadal) to show basin-to-basin contrast directly ──
    ax = axes[1]
    # Find annual and decadal bands by label keywords
    annual_fracs  = []
    decadal_fracs = []
    for basin_bands in all_bands:
        for b in basin_bands:
            if "annual" in b["label"].lower() and "inter" not in b["label"].lower() \
                    and "multi" not in b["label"].lower():
                annual_fracs.append(b["variance_frac"])
            if "decadal" in b["label"].lower() and "multi" not in b["label"].lower():
                decadal_fracs.append(b["variance_frac"])

    if annual_fracs and decadal_fracs and len(annual_fracs) == n_basins:
        w = 0.35
        ax.bar(x - w/2, annual_fracs,  w, label="Annual",  color=_BAND_COLORS[1],
               alpha=0.85, edgecolor="white")
        ax.bar(x + w/2, decadal_fracs, w, label="Decadal", color=_BAND_COLORS[3],
               alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(basins, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Fraction of total SMB variance")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=9)
        ax.set_title("Annual vs decadal band variance\n(key Experiment targets)")
    else:
        ax.text(0.5, 0.5, "Annual/decadal bands\nnot found in period_edges",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ======================================================================
# Utilities
# ======================================================================

def _auto_label(period_yr: float) -> str:
    """Map a peak period to a human-readable label."""
    if period_yr < 0.5:
        return "sub-seasonal"
    if period_yr < 0.9:
        return "sub-annual"
    if period_yr < 1.5:
        return "annual"
    if period_yr < 4.0:
        return "biennial"
    if period_yr < 8.0:
        return "inter-annual"
    if period_yr < 20.0:
        return "decadal"
    return "multi-decadal"