"""
multi_basin.py
==============
Utilities for running the synthetic SMB pipeline across multiple
WAIS drainage basins simultaneously and producing comparison figures.

Typical usage
-------------
from syn_smb import SMBGenerator, Experiment
from multi_basin import multi_basin_run, plot_multibasin_psd, multibasin_metrics_table

basin_paths = {
    "PIG":     "./data/PIG_smb.nc",
    "Thwaites":"./data/Thwaites_smb.nc",
    "Getz":    "./data/Getz_smb.nc",
}

results  = multi_basin_run(basin_paths)
plot_multibasin_psd(results)
df       = multibasin_metrics_table(results)
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import welch

from .generator import SMBGenerator
from .experiment import Experiment
from .validator import Validator
from .data_loader import SMBDataLoader


# Colorblind-safe palette (Wong 2011) — up to 7 basins
_BASIN_COLORS = [
    "#0072B2",   # blue
    "#E69F00",   # orange
    "#009E73",   # green
    "#D55E00",   # vermillion
    "#CC79A7",   # pink
    "#56B4E9",   # sky blue
    "#F0E442",   # yellow
]


def multi_basin_run(
    basin_paths: dict[str, str],
    suite: list[Experiment] | None = None,
    var: str = "smbgl",
    verbose: bool = True,
    **generator_kwargs,
) -> dict[str, dict]:
    """
    Run the full pipeline across multiple WAIS drainage basins.

    For each basin: loads RACMO data, fits SMBGenerator, generates the
    standard experiment suite, and runs the validation suite.

    Parameters
    ----------
    basin_paths : dict[str, str]
        Mapping of {basin_name: path_to_racmo_netcdf}.
    suite : list of Experiment or None
        Experiments to run for each basin. Defaults to
        Experiment.standard_suite(n_years=1000, n_members=30).
    var : str
        RACMO variable name. Default 'smbgl'.
    verbose : bool
        Print progress. Default True.
    **generator_kwargs
        Passed to SMBGenerator.__init__() (e.g. nperseg, remove_seasonal).

    Returns
    -------
    results : dict[str, dict]
        One entry per basin. Each entry contains:
          'generator'  : fitted SMBGenerator
          'smb'        : observed xr.DataArray
          'datasets'   : {experiment_name: xr.Dataset}
          'validator'  : fitted Validator
          'metrics'    : {experiment_name: metrics_dict}
    """
    if suite is None:
        suite = Experiment.standard_suite(n_years=1000, n_members=30, seed=0)

    results = {}

    for basin, path in basin_paths.items():
        if verbose:
            print(f"\n{'─'*50}")
            print(f"Basin: {basin}  ({path})")
            print(f"{'─'*50}")

        # Load
        smb = SMBDataLoader(path, var=var).load()
        if verbose:
            print(f"  Loaded: n={smb.sizes['time']}, "
                  f"mean={float(smb.mean()):.4f}, "
                  f"std={float(smb.std()):.4f} m w.e. a⁻¹")

        # Fit
        gen = SMBGenerator(**generator_kwargs)
        gen.fit(smb)
        if verbose:
            print(f"  Fitted: {gen}")

        # Generate suite
        if verbose:
            print(f"  Generating {len(suite)} experiments...")
        datasets = gen.generate_suite(suite)

        # Validate and compute metrics
        val = Validator(gen, smb)
        metrics = {}
        for exp in suite:
            m = val.compute_metrics(datasets[exp.name], verbose=False)
            metrics[exp.name] = m
        if verbose:
            print(f"  Metrics computed for all {len(suite)} experiments.")

        results[basin] = {
            "generator": gen,
            "smb":       smb,
            "datasets":  datasets,
            "validator": val,
            "metrics":   metrics,
        }

    if verbose:
        print(f"\n{'─'*50}")
        print(f"Done. {len(results)} basins processed.")

    return results


def plot_multibasin_psd(
    results: dict[str, dict],
    experiment: str = "baseline",
    save_path: str | None = None,
) -> None:
    """
    Overlay power spectral densities for all basins on a single figure.

    This is the key figure demonstrating that each WAIS basin has a
    distinct SMB spectral character, motivating basin-specific synthesis.

    Parameters
    ----------
    results : dict
        Output of multi_basin_run().
    experiment : str
        Which experiment's g_syn to use for the PSD comparison.
        Default 'baseline'.
    save_path : str or None
    """
    n_basins = len(results)
    colors   = _BASIN_COLORS[:n_basins]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("WAIS basin SMB power spectral density comparison", fontsize=12)

    # ── left: all basin fitted PSDs with CIs ──
    ax = axes[0]
    for (basin, res), color in zip(results.items(), colors):
        gen = res["generator"]
        ss  = gen.spectral_synthesizer
        ax.loglog(ss.freqs[1:], ss.psd[1:],
                  color=color, lw=2, label=basin, zorder=3)
        ax.fill_between(ss.freqs[1:],
                        ss.psd_ci_lower[1:], ss.psd_ci_upper[1:],
                        color=color, alpha=0.10)

    ax.axvline(1.0, color="gray", linestyle=":", lw=1, alpha=0.6)
    ax.axvline(0.1, color="gray", linestyle=":", lw=1, alpha=0.6)
    ax.text(1.05, ax.get_ylim()[1] * 0.5, "Annual", fontsize=8,
            color="gray", va="center", rotation=90, alpha=0.7)
    ax.text(0.105, ax.get_ylim()[1] * 0.5, "Decadal", fontsize=8,
            color="gray", va="center", rotation=90, alpha=0.7)
    ax.set_xlabel("Frequency (cycles yr⁻¹)")
    ax.set_ylabel("PSD (Gaussianized residuals)")
    ax.set_title("Fitted PSD with 95% CI\n(Welch estimate, observed record)")
    ax.legend(fontsize=9)

    # ── right: annual peak and decadal band power per basin (bar chart) ──
    ax = axes[1]
    basin_names = list(results.keys())
    annual_psds = []
    decadal_psds = []

    for basin, res in results.items():
        ss  = res["generator"].spectral_synthesizer
        annual_idx  = int(np.argmin(np.abs(ss.freqs - 1.0)))
        decadal_idx = int(np.argmin(np.abs(ss.freqs - 0.1)))
        annual_psds.append(float(ss.psd[annual_idx]))
        decadal_psds.append(float(ss.psd[decadal_idx]))

    x = np.arange(n_basins)
    w = 0.35
    b1 = ax.bar(x - w/2, annual_psds, w, label="Annual (f=1.0)",
                color=colors, alpha=0.8, edgecolor="white")
    b2 = ax.bar(x + w/2, decadal_psds, w, label="Decadal (f=0.1)",
                color=colors, alpha=0.4, edgecolor=colors,
                linewidth=1.5, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(basin_names, rotation=20, ha="right")
    ax.set_ylabel("PSD at peak frequency")
    ax.set_yscale("log")
    ax.set_title("Annual vs decadal band power\nper basin")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def multibasin_metrics_table(
    results: dict[str, dict],
    experiment: str = "baseline",
    verbose: bool = True,
) -> dict:
    """
    Compute and display validation metrics for all basins in a formatted table.

    Parameters
    ----------
    results : dict
        Output of multi_basin_run().
    experiment : str
        Which experiment to compare metrics for. Default 'baseline'.
    verbose : bool
        Print the formatted table. Default True.

    Returns
    -------
    table : dict
        {basin_name: metrics_dict} for the specified experiment.
    """
    table = {}
    for basin, res in results.items():
        if experiment in res["metrics"]:
            table[basin] = res["metrics"][experiment]

    if verbose and table:
        metrics_keys = list(next(iter(table.values())).keys())
        # Header
        col_w = 12
        header = f"{'Basin':<15}" + "".join(f"{k:>{col_w}}" for k in metrics_keys)
        print(f"\nValidation metrics — experiment: {experiment}")
        print("─" * len(header))
        print(header)
        print("─" * len(header))
        for basin, m in table.items():
            row = f"{basin:<15}" + "".join(f"{m[k]:>{col_w}.4f}" for k in metrics_keys)
            print(row)
        print("─" * len(header))

    return table


def plot_multibasin_metrics(
    results: dict[str, dict],
    experiment: str = "baseline",
    save_path: str | None = None,
) -> None:
    """
    Heatmap-style figure showing validation metrics for all basins.

    Provides a compact visual summary of Table 2 in the paper.

    Parameters
    ----------
    results : dict
        Output of multi_basin_run().
    experiment : str
        Which experiment to show. Default 'baseline'.
    save_path : str or None
    """
    table   = multibasin_metrics_table(results, experiment, verbose=False)
    basins  = list(table.keys())
    metrics = [
        "mean_ratio", "variance_ratio", "ks_statistic",
        "psd_rms_error", "acf_lag1_error", "acf_lag12_error", "seasonal_rmse"
    ]
    labels  = [
        "Mean ratio", "Var. ratio", "KS stat.",
        "PSD RMS", "ACF lag-1", "ACF lag-12", "Seasonal RMSE"
    ]
    targets = [1.0, 1.0, None, None, 0.05, 0.05, None]

    data = np.array([[table[b][m] for m in metrics] for b in basins])

    fig, ax = plt.subplots(figsize=(11, max(3, 0.6 * len(basins) + 1.5)))
    fig.suptitle(f"Validation metrics — {experiment}", fontsize=11)

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(basins)))
    ax.set_yticklabels(basins, fontsize=9)

    # Annotate cells
    for i in range(len(basins)):
        for j in range(len(metrics)):
            val = data[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color="black")

    # Mark target reference below
    for j, (label, target) in enumerate(zip(labels, targets)):
        if target is not None:
            ax.text(j, len(basins) - 0.45, f"target≈{target}",
                    ha="center", va="bottom", fontsize=7, color="gray")

    plt.colorbar(im, ax=ax, fraction=0.02, label="Metric value")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_multibasin_split_test(
    results: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """
    Side-by-side calibration split test results for all basins.

    Runs calibration_split_test() for each basin and overlays the
    variance_ratio and psd_rms_error results, showing whether out-of-sample
    fidelity is consistent across the WAIS.

    Parameters
    ----------
    results : dict
        Output of multi_basin_run().
    save_path : str or None
    """
    n_basins = len(results)
    colors   = _BASIN_COLORS[:n_basins]

    split_results = {}
    for (basin, res), color in zip(results.items(), colors):
        print(f"Running calibration split test: {basin}...")
        val    = res["validator"]
        splits = val.calibration_split_test(
            n_members=10, n_years_syn=50, verbose=False
        )
        split_results[basin] = splits

    metrics_to_show = [
        ("variance_ratio",  "Variance ratio",   1.0),
        ("psd_rms_error",   "PSD RMS error",    None),
        ("acf_lag1_error",  "ACF lag-1 error",  0.05),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_show), figsize=(13, 5))
    fig.suptitle("Calibration split test — all basins", fontsize=11)

    for ax, (key, label, target) in zip(axes, metrics_to_show):
        for (basin, splits), color in zip(split_results.items(), colors):
            v1 = splits["first_half"][key]
            v2 = splits["second_half"][key]
            ax.plot([0, 1], [v1, v2], "o-", color=color,
                    lw=1.5, markersize=6, label=basin)

        if target is not None:
            ax.axhline(target, color="black", linestyle="--",
                       lw=1, label=f"Target={target}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["First half\n(train)", "Second half\n(train)"],
                           fontsize=9)
        ax.set_title(label)
        if ax is axes[0]:
            ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()