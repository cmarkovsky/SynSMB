"""
multi_basin_survey.py
=====================
Surveys the spectral structure of basin-integrated RACMO SMB across the
basins registered in a RACMOCatalog, to establish WHETHER the synthesis
machinery is necessary (i.e. whether basins depart from white noise) rather
than demonstrating it on Pine Island alone.

Uses the project's RACMOCatalog for basin discovery / metadata / filtering and
the pipeline's SMBDataLoader to load each basin exactly as the generator does,
then runs the verified decadal_diagnostic estimators on each series:
  load -> preprocess -> full-record multitaper band variance (NW=1.5)
       -> Monte Carlo significance vs white and AR(1) nulls
and classifies each basin as white / red (memory) / band-structured.

Outputs
-------
  results/multi_basin_survey.csv        one row per basin (+ region metadata)
  results/fig_multibasin_psd.pdf        variance-normalised PSD, one line/basin

Why normalise the PSD in the figure: whiteness is a statement about spectral
SHAPE, not magnitude. Dividing each PSD by its own variance makes a white
basin flat, a red basin slope down, and a cyclic basin show a low-frequency
bump. Magnitude (std, variance) lives in the table.

HOW TO RUN
----------
    poetry run python multi_basin_survey.py

Needs decadal_diagnostic.py importable (same directory) and the project
package (for RACMOCatalog + SMBDataLoader). Run once with LIST_ONLY = True to
print the catalog, then set filters. No shapefile/regionmask needed - the
catalog points at per-basin files.
"""

from __future__ import annotations
import os
import sys
import numpy as np

import decadal_diagnostic as dd    # verified estimators + MC test

# =====================================================================
#  CONFIG - EDIT THIS
# =====================================================================
DATA_DIR     = "./data/sectors_smb"      # dir of per-basin {basin}_smb.nc files
VAR          = "smbgl"
FILE_PATTERN = "{basin}_smb.nc"
OUTDIR       = "./results"

IMPORT_CANDIDATES = ["syn_smb.core", "syn_smb", "src.syn_smb.core", "core"]

# Set True once to print the catalog (regions / subregions / basins), then
# choose filters below.
LIST_ONLY = False

# Catalog filters (all optional, AND-combined). Leave as None to survey all
# available basins. Examples:
#   REGION = "West"          # partial, case-insensitive match on region
#   SUBREGION = "Ipp-J"
#   NAMES = ["PineIsland", "Thwaites", "Getz"]
REGION    = None
SUBREGION = None
NAMES     = None

# Extra basins to register beyond the catalog's auto-discovered KNOWN_BASINS,
# e.g. East Antarctic files you have. {key: path} or
# {key: {"path":..., "region":..., "subregion":..., "name":...}}
EXTRA_BASINS: dict = {}

ANNUAL_BAND  = (0.8, 1.5)
DECADAL_BAND = (8.0, 20.0)
FS           = 12.0
NW           = 1.5
N_SURR       = 500
SEED         = 0

PHI_RED    = 0.20      # |AR(1) phi| above this -> meaningful memory (redness)
SIG_PCTILE = 95.0      # Monte Carlo percentile for "significant excess"
EFFECT_RATIO = 2.0     # band variance must exceed this x the white-noise
                       # expectation to count as structure (guards against
                       # Monte-Carlo false positives from multiple comparisons)
REFERENCE_NAME = "PineIsland"   # sanity-gate basin (known white)
# =====================================================================

_COLORS = ["#000000", "#0072B2", "#D55E00", "#009E73",
           "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
_LS = ["-", "--", "-.", ":"]


def _import_project():
    """Locate RACMOCatalog and SMBDataLoader in the project package."""
    last = None
    for base in IMPORT_CANDIDATES:
        try:
            cat = __import__(f"{base}.racmo_catalog", fromlist=["RACMOCatalog"])
            ldr = __import__(f"{base}.data_loader", fromlist=["SMBDataLoader"])
            return cat.RACMOCatalog, ldr.SMBDataLoader, base
        except Exception as err:                     # noqa: BLE001
            last = err
    print("ERROR: could not import RACMOCatalog / SMBDataLoader.")
    print(f"Tried {IMPORT_CANDIDATES}; last error: {last}")
    print("Add your package path to IMPORT_CANDIDATES or run from the repo root.")
    sys.exit(1)


def load_series(SMBDataLoader, path):
    """Load a basin exactly as the generator does; collapse to a 1-D series."""
    smb = SMBDataLoader(str(path), var=VAR).load()
    if getattr(smb, "ndim", 1) > 1:
        dims = [d for d in smb.dims if d != "time"]
        smb = smb.mean(dim=dims)
    return np.asarray(smb.values, dtype=float).ravel()


def _white_expectation_pct(band):
    """Fraction of variance a white process puts in `band`, as a percent."""
    return 100.0 * (1.0 / band[0] - 1.0 / band[1]) / (FS / 2.0)


def classify(phi, dec_pct, dec_pw, dec_pa, ann_pct, ann_pw, ann_pa):
    """
    Tri-state verdict for method necessity (IID inadequate unless white).

    A band counts as an excess only if it is BOTH statistically significant
    (Monte Carlo percentile >= SIG_PCTILE) AND carries at least EFFECT_RATIO
    times the white-noise expectation for that band. The effect-size gate
    prevents a near-white basin from being mislabelled 'structured' when
    multiple-comparison noise pushes one percentile over the threshold.
    """
    wdec = _white_expectation_pct(DECADAL_BAND)
    wann = _white_expectation_pct(ANNUAL_BAND)
    dec_big = dec_pct >= EFFECT_RATIO * wdec
    ann_big = ann_pct >= EFFECT_RATIO * wann
    if (dec_big and dec_pa >= SIG_PCTILE) or (ann_big and ann_pa >= SIG_PCTILE):
        return "band-structured"
    if (abs(phi) >= PHI_RED
            or (dec_big and dec_pw >= SIG_PCTILE)
            or (ann_big and ann_pw >= SIG_PCTILE)):
        return "red (memory)"
    return "white"


def survey_series(series, entry):
    """Diagnose one basin series. Returns (row_dict, (freqs, psd_norm))."""
    resid, _ = dd.preprocess(series, verbose=False)
    f_mt, p_mt = dd.multitaper_psd(resid, fs=FS, NW=NW)
    v_dec, _ = dd.band_variance(f_mt, p_mt, *DECADAL_BAND)
    v_ann, _ = dd.band_variance(f_mt, p_mt, *ANNUAL_BAND)
    rv = float(resid.var())

    phi, _ = dd._ar1_fit(resid)
    efold = (-1.0 / np.log(abs(phi))) if 0 < abs(phi) < 1 else 0.0

    mc_dec = dd.monte_carlo_band_test(resid, *DECADAL_BAND, fs=FS, NW=NW,
                                      n_surr=N_SURR, seed=SEED)
    mc_ann = dd.monte_carlo_band_test(resid, *ANNUAL_BAND, fs=FS, NW=NW,
                                      n_surr=N_SURR, seed=SEED)
    dpw = mc_dec["nulls"]["white noise"]["percentile_of_obs"]
    dpa = mc_dec["nulls"]["AR(1)"]["percentile_of_obs"]
    apw = mc_ann["nulls"]["white noise"]["percentile_of_obs"]
    apa = mc_ann["nulls"]["AR(1)"]["percentile_of_obs"]

    from scipy.signal import welch
    fw, pw = welch(resid, fs=FS, nperseg=min(60, resid.size), detrend="constant")
    parseval = 100.0 * float(np.sum(pw) * (fw[1] - fw[0])) / rv if rv else np.nan

    dec_pct = 100.0 * v_dec / rv if rv else float("nan")
    ann_pct = 100.0 * v_ann / rv if rv else float("nan")
    verdict = classify(phi, dec_pct, dpw, dpa, ann_pct, apw, apa)

    row = dict(
        name=entry.name, subregion=entry.subregion, region=entry.region,
        n_obs=series.size, mean=float(np.mean(series)), std=float(np.std(series)),
        resid_var=rv, ar1_phi=phi, efold_months=efold,
        decadal_pct=dec_pct,
        decadal_p_white=dpw, decadal_p_ar1=dpa,
        annual_pct=ann_pct,
        annual_p_white=apw, annual_p_ar1=apa,
        parseval60_pct=parseval, verdict=verdict,
    )
    df = f_mt[1] - f_mt[0]
    return row, (f_mt, p_mt / (np.sum(p_mt) * df))


def make_figure(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.7, 3.2))
    for i, (label, is_ref, f, p) in enumerate(curves):
        m = f > 0
        ax.plot(f[m], p[m], color=_COLORS[i % len(_COLORS)],
                ls=_LS[(i // len(_COLORS)) % len(_LS)],
                lw=1.8 if is_ref else 1.0, label=label, alpha=0.9)
    for band, lab in ((ANNUAL_BAND, "annual"), (DECADAL_BAND, "decadal")):
        ax.axvspan(1.0 / band[1], 1.0 / band[0], color="#999999",
                   alpha=0.13, lw=0)
        ax.text(np.sqrt((1.0 / band[1]) * (1.0 / band[0])), 2e-3, lab,
                fontsize=6, ha="center", color="#555555")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("frequency (cycles yr$^{-1}$)")
    ax.set_ylabel("variance-normalised PSD")
    ax.set_title("Spectral shape of basin-integrated SMB residual "
                 "(flat = white)", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  wrote {path}")


def main():
    RACMOCatalog, SMBDataLoader, base_pkg = _import_project()
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Project package: {base_pkg}")

    cat = RACMOCatalog(data_dir=DATA_DIR, var=VAR, file_pattern=FILE_PATTERN)
    if EXTRA_BASINS:
        cat.register_many(EXTRA_BASINS)

    if LIST_ONLY:
        cat.summarize()
        return

    paths = cat.paths(region=REGION, subregion=SUBREGION, names=NAMES,
                      require_exists=True)
    if not paths:
        print("No basins matched the filters / none of the files exist.")
        print(f"Catalog data_dir={DATA_DIR}, pattern={FILE_PATTERN}.")
        print("Run with LIST_ONLY = True to inspect the catalog.")
        sys.exit(1)

    # PIG-first ordering for the reference sanity gate
    keys = sorted(paths, key=lambda k: (cat.entry(k).name != REFERENCE_NAME,
                                        cat.entry(k).name or k))

    print("=" * 74)
    print(f"MULTI-BASIN WHITENESS SURVEY  ({len(keys)} basins)")
    print("=" * 74)
    rows, curves = [], []
    for i, key in enumerate(keys):
        entry = cat.entry(key)
        label = entry.name or key
        print(f"[{i+1}/{len(keys)}] {entry.display_name} ...")
        try:
            series = load_series(SMBDataLoader, paths[key])
            row, (f, p) = survey_series(series, entry)
        except Exception as e:                       # noqa: BLE001
            print(f"    FAILED: {e}")
            rows.append(dict(name=label, region=entry.region,
                             verdict=f"ERROR: {e}"))
            continue
        rows.append(row)
        curves.append((label, entry.name == REFERENCE_NAME, f, p))
        print(f"    phi={row['ar1_phi']:+.3f} (e-fold {row['efold_months']:.1f} mo), "
              f"decadal {row['decadal_pct']:.1f}% (p_w {row['decadal_p_white']:.0f}, "
              f"p_AR1 {row['decadal_p_ar1']:.0f}), Parseval60 "
              f"{row['parseval60_pct']:.1f}% -> {row['verdict']}")
        if entry.name == REFERENCE_NAME:
            ok = (abs(row["ar1_phi"]) < 0.2 and row["decadal_pct"] < 5
                  and row["parseval60_pct"] > 90)
            print(f"    [reference check: {'OK' if ok else 'UNEXPECTED - verify loader'}]")

    # table
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    hdr = (f"{'basin':16} {'subreg':>7} {'std':>8} {'phi':>6} {'dec%':>6} "
           f"{'p_ar1':>5} {'ann%':>6} {'pars60':>7} {'verdict':>16}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if "std" not in r:
            print(f"{(r.get('name') or '?'):16} {'--':>7} {'(failed)':>8}")
            continue
        print(f"{(r['name'] or '?'):16} {(r['subregion'] or '-'):>7} "
              f"{r['std']:8.4f} {r['ar1_phi']:+6.2f} {r['decadal_pct']:6.1f} "
              f"{r['decadal_p_ar1']:5.0f} {r['annual_pct']:6.1f} "
              f"{r['parseval60_pct']:7.1f} {r['verdict']:>16}")

    n_white = sum(1 for r in rows if r.get("verdict") == "white")
    n_struct = sum(1 for r in rows if r.get("verdict") in
                   ("red (memory)", "band-structured"))
    print(f"\n  {n_white} white, {n_struct} structured (of {len(curves)} surveyed)")
    if n_struct == 0:
        print("  -> ALL basins white: frame the method as a principled default")
        print("     for a spectrally-featureless field; band control is a")
        print("     counterfactual tool. (Still publishable.)")
    else:
        print("  -> structured basins exist: the synthesis machinery is")
        print("     NECESSARY (IID sampling cannot reproduce their spectra).")

    try:
        import pandas as pd
        csv = os.path.join(OUTDIR, "multi_basin_survey.csv")
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"\n  wrote {csv}")
    except Exception as e:                           # noqa: BLE001
        print(f"  (CSV skipped: {e})")
    if curves:
        make_figure(curves, os.path.join(OUTDIR, "fig_multibasin_psd.png"))
    print("\nDONE.")


if __name__ == "__main__":
    main()