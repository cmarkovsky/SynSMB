"""
spatial_whiteness_diagnostic.py
===============================
Tests whether basin-mean aggregation hides timescale structure that a
grid-cell / EOF (field) view would reveal, for ONE basin. Answers the scope
question: is the 2-D pipeline scientifically *necessary* (the field carries
spectral structure the basin mean cannot see) or merely a spatial-coherence
convenience (everything is white all the way down)?

Three diagnostics, increasing in what they reveal:

  1. Grid-cell-mean PSD vs basin-mean PSD.
     Average the per-cell PSDs (not the series) and compare to the PSD of the
     averaged series. If the shapes match, aggregation is frequency-neutral and
     basin-mean whiteness is real. If the basin-mean spectrum is redder,
     aggregation is a spatial low-pass filter suppressing incoherent
     high-frequency variance.

  2. Band-resolved effective spatial coherence.
     From Var(mean) = (sigma^2 / N)[1 + (N-1) rho_bar], the effective mean
     pairwise correlation in a band is
        rho_bar = (N * V_mean_band / Vbar_cell_band - 1) / (N - 1).
     Computed for the annual and decadal bands. rho_decadal >> rho_annual is
     the signature of the red-shift hypothesis (coherent low frequency,
     incoherent high frequency).

  3. Per-PC whiteness (the decisive test).
     Run the whiteness diagnostic on the leading principal-component series.
     The basin mean is BLIND to any EOF whose spatial pattern integrates to
     ~zero over the basin (dipoles, higher modes) — exactly where a coherent
     decadal signal would hide. A well-separated PC with significant decadal
     structure, while the basin mean is white, is proof the field is necessary.
     North's rule flags modes whose eigenvalue is not separable from its
     neighbour (mode-mixing noise — the boring explanation to rule out first).

Outputs
-------
  results/spatial_whiteness_<basin>.csv     per-PC table
  results/fig_spatial_whiteness_<basin>.pdf  three-panel figure

HOW TO RUN
----------
    poetry run python spatial_whiteness_diagnostic.py

Needs decadal_diagnostic.py importable and the project package (SMBFieldLoader,
SpatialPreprocessor, EOFDecomposer). Uses the FULL-domain RACMO file + a
shapefile (the field loader masks the basin itself).
"""

from __future__ import annotations
import os
import sys
import numpy as np

import decadal_diagnostic as dd

# =====================================================================
#  CONFIG - EDIT THIS
# =====================================================================
RACMO_PATH = "./data/RACMO2.4p1_ANT11_full.nc"   # FULL-DOMAIN field
SHP_PATH   = "./data/IceBoundaries_Antarctica_v02_dissolved.shp"
BASIN_NAME = "Ronne"
NAME_COL   = "NAME"
SMB_VAR    = "smbgl"
OUTDIR     = "./results"

IMPORT_CANDIDATES = ["syn_smb.core", "syn_smb", "src.syn_smb.core", "core"]

N_MODES        = 8         # PCs to decompose + test
MAX_CELLS_PSD  = 500       # subsample for the cell-mean PSD (converges fast)
ANNUAL_BAND    = (0.8, 1.5)
DECADAL_BAND   = (8.0, 20.0)
FS             = 12.0
NW             = 1.5
N_SURR         = 500
SEED           = 0

PHI_RED      = 0.20
SIG_PCTILE   = 95.0
EFFECT_RATIO = 2.0         # band must exceed this x white expectation
# =====================================================================

_COLORS = ["#000000", "#0072B2", "#D55E00", "#009E73",
           "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]


def _import_project():
    last = None
    for base in IMPORT_CANDIDATES:
        try:
            ld = __import__(f"{base}.smb_field_loader", fromlist=["SMBFieldLoader"])
            pp = __import__(f"{base}.spatial_preprocessor", fromlist=["SpatialPreprocessor"])
            eo = __import__(f"{base}.eof_decomposer", fromlist=["EOFDecomposer"])
            return ld.SMBFieldLoader, pp.SpatialPreprocessor, eo.EOFDecomposer, base
        except Exception as err:                     # noqa: BLE001
            last = err
    print("ERROR: could not import the 2-D pipeline classes.")
    print(f"Tried {IMPORT_CANDIDATES}; last error: {last}")
    sys.exit(1)


def _white_expectation_pct(band):
    return 100.0 * (1.0 / band[0] - 1.0 / band[1]) / (FS / 2.0)


def classify(phi, dec_pct, dec_pw, dec_pa, ann_pct, ann_pw, ann_pa):
    """Significance + effect-size gate (identical logic to the basin survey)."""
    wdec, wann = _white_expectation_pct(DECADAL_BAND), _white_expectation_pct(ANNUAL_BAND)
    dbig, abig = dec_pct >= EFFECT_RATIO * wdec, ann_pct >= EFFECT_RATIO * wann
    if (dbig and dec_pa >= SIG_PCTILE) or (abig and ann_pa >= SIG_PCTILE):
        return "band-structured"
    if abs(phi) >= PHI_RED or (dbig and dec_pw >= SIG_PCTILE) or (abig and ann_pw >= SIG_PCTILE):
        return "red (memory)"
    return "white"


def band_variance_series(x, band):
    f, p = dd.multitaper_psd(np.asarray(x, float), fs=FS, NW=NW)
    v, _ = dd.band_variance(f, p, *band)
    return v


def diagnostic_1_2(X, basin_mean):
    """Cell-vs-basin PSD (shape) and band-resolved effective coherence."""
    n_time, n_cell = X.shape

    # --- basin-mean PSD ---
    f_bm, p_bm = dd.multitaper_psd(basin_mean, fs=FS, NW=NW)

    # --- cell-mean PSD (subsample cells; the mean converges fast) ---
    rng = np.random.default_rng(SEED)
    idx = (rng.choice(n_cell, MAX_CELLS_PSD, replace=False)
           if n_cell > MAX_CELLS_PSD else np.arange(n_cell))
    p_cells = np.zeros_like(p_bm)
    for j in idx:
        _, pj = dd.multitaper_psd(X[:, j], fs=FS, NW=NW)
        p_cells += pj
    p_cells /= idx.size

    # normalise both to unit total variance -> shape comparison
    df = f_bm[1] - f_bm[0]
    pn_bm = p_bm / (np.sum(p_bm) * df)
    pn_cell = p_cells / (np.sum(p_cells) * df)

    # --- band-resolved effective coherence ---
    coh = {}
    for name, band in (("annual", ANNUAL_BAND), ("decadal", DECADAL_BAND)):
        v_mean = band_variance_series(basin_mean, band)
        vbar_cell = np.mean([band_variance_series(X[:, j], band) for j in idx])
        rho = ((n_cell * v_mean / vbar_cell - 1.0) / (n_cell - 1.0)
               if vbar_cell > 0 and n_cell > 1 else np.nan)
        coh[name] = dict(v_mean=v_mean, vbar_cell=vbar_cell,
                         suppression=v_mean / vbar_cell if vbar_cell else np.nan,
                         rho_eff=rho)
    return f_bm, pn_bm, pn_cell, coh


def diagnostic_3(pcs, expl_var, sing_vals, n_time):
    """Per-PC whiteness + North's-rule separability flag."""
    n_modes = pcs.shape[1]
    # covariance eigenvalues ~ s^2; North error dlambda = lambda*sqrt(2/N)
    lam = np.asarray(sing_vals, float) ** 2
    dlam = lam * np.sqrt(2.0 / n_time)
    rows = []
    for i in range(n_modes):
        pc = pcs[:, i]
        resid, _ = dd.preprocess(pc, verbose=False)   # PCs are already ~zero-mean, no seasonal
        rv = float(resid.var())
        phi, _ = dd._ar1_fit(resid)
        f, p = dd.multitaper_psd(resid, fs=FS, NW=NW)
        vdec, _ = dd.band_variance(f, p, *DECADAL_BAND)
        vann, _ = dd.band_variance(f, p, *ANNUAL_BAND)
        mc_d = dd.monte_carlo_band_test(resid, *DECADAL_BAND, fs=FS, NW=NW, n_surr=N_SURR, seed=SEED)
        mc_a = dd.monte_carlo_band_test(resid, *ANNUAL_BAND, fs=FS, NW=NW, n_surr=N_SURR, seed=SEED)
        dpw = mc_d["nulls"]["white noise"]["percentile_of_obs"]
        dpa = mc_d["nulls"]["AR(1)"]["percentile_of_obs"]
        apw = mc_a["nulls"]["white noise"]["percentile_of_obs"]
        apa = mc_a["nulls"]["AR(1)"]["percentile_of_obs"]
        dec_pct = 100 * vdec / rv if rv else np.nan
        ann_pct = 100 * vann / rv if rv else np.nan
        verdict = classify(phi, dec_pct, dpw, dpa, ann_pct, apw, apa)

        # North separability: gap to nearest neighbour vs this mode's error
        gaps = []
        if i > 0:            gaps.append(lam[i - 1] - lam[i])
        if i < n_modes - 1:  gaps.append(lam[i] - lam[i + 1])
        min_gap = min(gaps) if gaps else np.inf
        well_sep = min_gap > dlam[i]

        rows.append(dict(pc=i + 1, expl_var_pct=100 * expl_var[i],
                         ar1_phi=phi, decadal_pct=dec_pct, decadal_p_ar1=dpa,
                         annual_pct=ann_pct, verdict=verdict,
                         north_separable=well_sep,
                         pc_freqs=f, pc_psd=p / (np.sum(p) * (f[1] - f[0]))))
    return rows


def make_figure(f, pn_bm, pn_cell, coh, pc_rows, basin, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.0))

    # (a) cell-mean vs basin-mean PSD (shape)
    m = f > 0
    ax[0].loglog(f[m], pn_cell[m], color="#0072B2", lw=1.2, label="cell-mean PSD")
    ax[0].loglog(f[m], pn_bm[m], color="#000000", lw=1.4, label="basin-mean PSD")
    for band in (ANNUAL_BAND, DECADAL_BAND):
        ax[0].axvspan(1 / band[1], 1 / band[0], color="#999", alpha=0.13, lw=0)
    ax[0].set_xlabel("frequency (cycles yr$^{-1}$)")
    ax[0].set_ylabel("variance-normalised PSD")
    ax[0].legend(frameon=False, fontsize=6)
    ax[0].set_title("(a) aggregation filter", fontsize=8, loc="left")

    # (b) band-resolved effective coherence + suppression
    names = ["annual", "decadal"]
    rhos = [coh[n]["rho_eff"] for n in names]
    supp = [coh[n]["suppression"] for n in names]
    x = np.arange(2)
    ax[1].bar(x - 0.2, rhos, 0.4, color="#D55E00", label=r"$\bar\rho_{\rm eff}$")
    ax[1].bar(x + 0.2, supp, 0.4, color="#56B4E9", label="V_mean / V_cell")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].legend(frameon=False, fontsize=6)
    ax[1].set_title("(b) band coherence", fontsize=8, loc="left")

    # (c) per-PC PSDs (variance-normalised)
    for k, r in enumerate(pc_rows[:6]):
        fp = r["pc_freqs"]; mm = fp > 0
        lbl = f"PC{r['pc']} ({r['verdict'][:5]})"
        ax[2].loglog(fp[mm], r["pc_psd"][mm], color=_COLORS[k % len(_COLORS)],
                     lw=1.0, label=lbl, alpha=0.85)
    for band in (ANNUAL_BAND, DECADAL_BAND):
        ax[2].axvspan(1 / band[1], 1 / band[0], color="#999", alpha=0.13, lw=0)
    ax[2].set_xlabel("frequency (cycles yr$^{-1}$)")
    ax[2].set_ylabel("normalised PSD")
    ax[2].legend(frameon=False, fontsize=5, ncol=2)
    ax[2].set_title("(c) per-PC spectra", fontsize=8, loc="left")

    fig.suptitle(f"Spatial whiteness diagnostic — {basin}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  wrote {path}")


def main():
    SMBFieldLoader, SpatialPreprocessor, EOFDecomposer, base = _import_project()
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Project package: {base}")

    if not os.path.exists(RACMO_PATH):
        print(f"ERROR: full-domain RACMO not found: {RACMO_PATH}")
        sys.exit(1)

    print(f"\nLoading field for {BASIN_NAME} ...")
    loader = SMBFieldLoader(RACMO_PATH, SHP_PATH, BASIN_NAME,
                            smb_var=SMB_VAR, name_col=NAME_COL)
    field = loader.load(crop=True)                    # (time, rlat, rlon)
    lat = getattr(loader, "lat", None)
    if lat is None:
        lat = field["lat"] if "lat" in field.coords else None

    print("Preprocessing (detrend + deseasonalise per cell) ...")
    pre = SpatialPreprocessor()
    residuals = pre.fit_transform(field)              # (time, rlat, rlon)

    # stack to (time, cell), drop cells with any NaN
    spatial = [d for d in residuals.dims if d != "time"]
    stacked = residuals.stack(cell=spatial)
    valid = ~np.isnan(stacked).any("time")
    X = np.asarray(stacked.isel(cell=valid).values, float)   # (time, n_valid)
    n_time, n_cell = X.shape
    basin_mean = X.mean(axis=1)
    print(f"  {n_cell} valid cells, {n_time} months")

    print("\n[1+2] cell-vs-basin PSD and band coherence ...")
    f, pn_bm, pn_cell, coh = diagnostic_1_2(X, basin_mean)
    for name in ("annual", "decadal"):
        c = coh[name]
        print(f"  {name:8}: V_mean/V_cell = {c['suppression']:.3f}  "
              f"(rho_eff = {c['rho_eff']:+.3f})")
    redshift = (coh["decadal"]["rho_eff"] - coh["annual"]["rho_eff"])
    print(f"  decadal-minus-annual coherence = {redshift:+.3f} "
          f"({'RED-SHIFT: low-f more coherent' if redshift > 0.05 else 'no strong red-shift'})")

    print("\n[3] per-PC whiteness ...")
    eof = EOFDecomposer(n_modes=N_MODES).fit(residuals, lat=lat)
    pc_rows = diagnostic_3(eof.pcs, eof.explained_variance_ratio,
                           eof.singular_values, n_time)
    hdr = f"  {'PC':>3} {'expl%':>6} {'phi':>6} {'dec%':>6} {'p_ar1':>5} {'North?':>7} {'verdict':>16}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in pc_rows:
        print(f"  {r['pc']:>3} {r['expl_var_pct']:6.1f} {r['ar1_phi']:+6.2f} "
              f"{r['decadal_pct']:6.1f} {r['decadal_p_ar1']:5.0f} "
              f"{'sep' if r['north_separable'] else 'MIXED':>7} {r['verdict']:>16}")

    # ---- verdict ----
    hidden = [r for r in pc_rows if r["verdict"] != "white" and r["north_separable"]]
    print("\n" + "=" * 68)
    if hidden:
        pcs_str = ", ".join(f"PC{r['pc']}" for r in hidden)
        print(f"  {pcs_str} show structure the basin mean does not, and are")
        print("  North-separable -> the FIELD carries timescale structure that")
        print("  aggregation cancels. The 2-D pipeline is scientifically necessary.")
    else:
        mixed = [r for r in pc_rows if r["verdict"] != "white"]
        note = (" (the only non-white PCs fail North's rule = mode-mixing noise)"
                if mixed else "")
        print("  All North-separable PCs are white; basin-mean whiteness is not an")
        print(f"  aggregation artefact{note}. The 2-D pipeline is a spatial-coherence")
        print("  convenience, not a spectral necessity — safe to defer to a 2nd paper.")
    print("=" * 68)

    # ---- outputs ----
    try:
        import pandas as pd
        drop = ("pc_freqs", "pc_psd")
        rows_csv = [{k: v for k, v in r.items() if k not in drop} for r in pc_rows]
        for name in ("annual", "decadal"):
            for r in rows_csv:
                r[f"coh_{name}"] = coh[name]["rho_eff"]
        csv = os.path.join(OUTDIR, f"spatial_whiteness_{BASIN_NAME}.csv")
        pd.DataFrame(rows_csv).to_csv(csv, index=False)
        print(f"\n  wrote {csv}")
    except Exception as e:                            # noqa: BLE001
        print(f"  (CSV skipped: {e})")
    make_figure(f, pn_bm, pn_cell, coh, pc_rows, BASIN_NAME,
                os.path.join(OUTDIR, f"fig_spatial_whiteness_{BASIN_NAME}.png"))
    print("\nDONE.")


if __name__ == "__main__":
    main()