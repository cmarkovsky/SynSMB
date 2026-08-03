"""
run_experiment_suite.py
=======================
Runs the corrected experiment suite on the fitted 1-D pipeline, organised
around TWO axes that answer two different questions. All variance is measured
on the final physical series with a full-record periodogram (nperseg=60 cannot
resolve the decadal band).

AXIS 1 - TIMESCALE ISOLATION (total variance conserved)
    Does the *timescale* at which variance sits matter, holding total variance
    (and mean, and marginal) fixed? Both sides scale only a residual band and
    rely on unit-variance renormalisation to REDISTRIBUTE variance:
      - annual_redist : band-only scaling of the 0.8-1.5 yr residual band,
                        seasonal_scale = 1.0 (seasonal cycle NOT amplified).
      - decadal       : band-only scaling of the 8-20 yr residual band.
    Matched on absolute injected band variance. This is the clean, airtight
    comparison: the two experiments differ ONLY in which timescale holds the
    injected variance. Feasible up to the decadal ceiling (~3-4x for PIG).

AXIS 2 - SEASONAL AMPLIFICATION (total variance increases; physical)
    What if the seasonal cycle itself were stronger (a warmer, wetter climate)?
    This ADDS variance at the annual timescale by amplifying the deterministic
    seasonal cycle:
      - annual_seasonal : annual_enhanced (scales seasonal amplitude + band).
    Each seasonal experiment is paired with a WHITE CONTROL that adds the same
    total variance with a FLAT spectrum (baseline + spectrally-white noise), so
    that any difference in ice response between the seasonal experiment and its
    white control is attributable to the annual TIMESCALE, not merely to the
    extra total variance. (Caveat: the white control broadens the marginal
    slightly; the seasonal experiment preserves it. Noted in the paper.)

Plus: annual_redist suppression (0.5x, total-conserving lower bracket) and
decadal_max (strongest decadal forcing the method can deliver).

HOW TO RUN
----------
    poetry run python run_experiment_suite.py

Requires the seasonal-scaling fix (annual_enhanced must set seasonal_scale).
Edit the CONFIG block.
"""

from __future__ import annotations
import os
import sys
import numpy as np

# =====================================================================
#  CONFIG - EDIT THIS
# =====================================================================
RACMO_PATH = "./data/PIG_smb.nc"
SMB_VAR    = "smbgl"
OUTDIR     = "./results/suite"

IMPORT_CANDIDATES = ["syn_smb.core", "syn_smb", "src.syn_smb.core", "core"]

ANNUAL_BAND  = (0.8, 1.5)     # years
DECADAL_BAND = (8.0, 20.0)    # years
FS           = 12.0           # samples per year

# AXIS 1: timescale isolation (total-conserving). Delivered annual ratios.
REDIST_MATCHED = [2.0, 3.0]
# AXIS 1 lower bracket (annual redistribute suppression).
SUPPRESS_RATIO = 0.5
# AXIS 2: seasonal amplification (adds total variance). Delivered annual ratios.
SEASONAL_LADDER = [2.0, 5.0, 10.0]

# Ensemble sizes
CAL_MEMBERS,   CAL_YEARS   = 5,  1000
FINAL_MEMBERS, FINAL_YEARS = 30, 1000
SEED = 0

# Calibration gamma grids (extend if inversion reports out-of-grid)
ANNUAL_REDIST_GRID   = [0.4, 0.7, 1.0, 1.5, 2.5, 4.0, 7.0, 12.0, 25.0, 60.0, 200.0]
ANNUAL_SEASONAL_GRID = [0.4, 0.7, 1.0, 1.5, 2.5, 4.0, 6.0, 9.0, 13.0, 18.0, 26.0]
DECADAL_GRID         = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
# =====================================================================


def _import_pipeline():
    last = None
    for base in IMPORT_CANDIDATES:
        try:
            g = __import__(f"{base}.generator", fromlist=["SMBGenerator"])
            e = __import__(f"{base}.experiment", fromlist=["Experiment"])
            return g.SMBGenerator, e.Experiment, base
        except Exception as err:                     # noqa: BLE001
            last = err
    print("ERROR: could not import SMBGenerator / Experiment.")
    print(f"Tried {IMPORT_CANDIDATES}; last error: {last}")
    sys.exit(1)


# ---------- measurement (verified: recovers A^2/2 for a pure sinusoid) -------
def band_variance(series, band, fs=FS):
    x = np.asarray(series, dtype=float).ravel()
    x = x - x.mean()
    X = np.fft.rfft(x)
    p = (np.abs(X) ** 2) / (fs * x.size)
    p[1:-1] *= 2.0
    f = np.fft.rfftfreq(x.size, d=1.0 / fs)
    m = (f >= 1.0 / band[1]) & (f <= 1.0 / band[0])
    return float(np.sum(p[m]) * (f[1] - f[0]))


def arr_of(ds):
    return np.atleast_2d(np.asarray(ds["smb_syn"].values, dtype=float))


def ens_band_var(arr, band):
    return float(np.mean([band_variance(r, band) for r in arr]))


def ens_total_var(arr):
    return float(np.mean([np.var(r) for r in arr]))


def invert_curve(gammas, delivered, target):
    g = np.asarray(gammas, float)
    d = np.asarray(delivered, float)
    order = np.argsort(d)
    d, g = d[order], g[order]
    if target <= d[0]:
        return float(g[0]), "at/below grid floor"
    if target >= d[-1]:
        return None, f"INFEASIBLE (max deliverable {d[-1]:.4g})"
    return float(np.interp(target, d, g)), "ok"


# ---------- experiment construction ----------------------------------------
def make_exp(Experiment, kind, gamma, common):
    """Build an Experiment for a given kind and gamma."""
    if kind == "baseline":
        return Experiment.baseline(**common)
    if kind == "decadal":
        return Experiment.decadal_enhanced(factor=gamma, **common)
    if kind == "annual_seasonal":
        return (Experiment.annual_suppressed(factor=gamma, **common)
                if gamma < 1 else
                Experiment.annual_enhanced(factor=gamma, **common))
    if kind == "annual_redist":
        # Band-only annual scaling, seasonal cycle NOT amplified -> total
        # variance conserved. Constructed directly (no preset does this).
        return Experiment(
            band_scales=[(ANNUAL_BAND[0], ANNUAL_BAND[1], gamma)],
            seasonal_scale=1.0,
            name=f"annual_redist_g{gamma:.3g}",
            description="Band-only annual redistribution (total-conserving).",
            **common,
        )
    raise ValueError(kind)


# ---------- calibration -----------------------------------------------------
def calibrate(gen, Experiment, kind, grid, band):
    common = dict(n_years=CAL_YEARS, n_members=CAL_MEMBERS, seed=SEED)
    gammas, delivered = [], []
    print(f"  calibrating {kind} over {len(grid)} gammas ...")
    for gm in grid:
        ds = gen.generate(make_exp(Experiment, kind, gm, common))
        v = ens_band_var(arr_of(ds), band)
        gammas.append(gm)
        delivered.append(v)
        print(f"    gamma={gm:8.2f} -> delivered band var = {v:.6e}")
    return gammas, delivered


# ---------- white control ---------------------------------------------------
def make_white_control(base_arr, total_ratio, V_tot_base, seed):
    """Baseline + spectrally-flat Gaussian noise matched to a total-var ratio."""
    dV = (total_ratio - 1.0) * V_tot_base
    if dV <= 0:
        return base_arr.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(dV), size=base_arr.shape)
    # remove each member's sample-mean noise so the control preserves the
    # record mean exactly (a control must not perturb the mean)
    noise -= noise.mean(axis=1, keepdims=True)
    return base_arr + noise


def save_array(base_ds, arr, path):
    """Save an ensemble array using the baseline ds coords/attrs."""
    try:
        wc = base_ds.copy(deep=True)
        wc["smb_syn"].values = arr
        wc.to_netcdf(path)
    except Exception as e:                           # noqa: BLE001
        print(f"    (save failed for {os.path.basename(path)}: {e})")


# ---------- main ------------------------------------------------------------
def main():
    SMBGenerator, Experiment, base_pkg = _import_pipeline()
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Pipeline: {base_pkg}")

    probe = Experiment.annual_enhanced(factor=2.0, n_years=10, n_members=1, seed=0)
    if not hasattr(probe, "seasonal_scale") or probe.seasonal_scale == 1.0:
        print("\nWARNING: seasonal-scaling fix does not appear applied "
              "(annual_enhanced.seasonal_scale != factor). Apply it first.\n")

    if not os.path.exists(RACMO_PATH):
        print(f"ERROR: data not found: {RACMO_PATH}. Edit RACMO_PATH.")
        sys.exit(1)

    # ---- PHASE 0: fit + baseline ----
    print("\n" + "=" * 70)
    print("PHASE 0  fit + baseline")
    print("=" * 70)
    gen = SMBGenerator.from_path(RACMO_PATH, var=SMB_VAR)
    try:
        val = gen.validate(verbose=False)
        print(f"  component validation passed: {val.get('passed')}")
    except Exception as e:                           # noqa: BLE001
        print(f"  (validate() unavailable: {e})")

    common_final = dict(n_years=FINAL_YEARS, n_members=FINAL_MEMBERS, seed=SEED)
    base_ds = gen.generate(Experiment.baseline(**common_final))
    base_arr = arr_of(base_ds)
    V_ann_base = ens_band_var(base_arr, ANNUAL_BAND)
    V_dec_base = ens_band_var(base_arr, DECADAL_BAND)
    V_tot_base = ens_total_var(base_arr)
    mean_base  = float(np.mean(base_arr))
    print(f"  baseline annual-timescale variance : {V_ann_base:.6e}")
    print(f"  baseline decadal-band variance     : {V_dec_base:.6e}")
    print(f"  baseline total variance            : {V_tot_base:.6e}")
    print(f"  baseline mean                      : {mean_base:.8f}")
    save_array(base_ds, base_arr, os.path.join(OUTDIR, "baseline.nc"))

    # ---- PHASE 1: calibrate three curves ----
    print("\n" + "=" * 70)
    print("PHASE 1  calibrate delivered variance")
    print("=" * 70)
    arg, ard = calibrate(gen, Experiment, "annual_redist",
                         ANNUAL_REDIST_GRID, ANNUAL_BAND)
    asg, asd = calibrate(gen, Experiment, "annual_seasonal",
                         ANNUAL_SEASONAL_GRID, ANNUAL_BAND)
    dg, dd = calibrate(gen, Experiment, "decadal", DECADAL_GRID, DECADAL_BAND)

    redist_ceiling = max(ard) / V_ann_base
    dec_ceiling_abs = max(dd)
    max_matched_r = 1.0 + (dec_ceiling_abs - V_dec_base) / V_ann_base
    print(f"\n  annual-redistribute ceiling  ~ {redist_ceiling:.2f}x delivered annual")
    print(f"  decadal ceiling              ~ {dec_ceiling_abs:.4e} "
          f"(matched annual ~ {max_matched_r:.2f}x)")

    # ---- PHASE 2: resolve the plan ----
    print("\n" + "=" * 70)
    print("PHASE 2  resolve requested gammas")
    print("=" * 70)
    # each entry: (label, kind, gamma, target_ratio, group, axis)
    plan = [("baseline", "baseline", None, 1.0, None, 0)]

    # AXIS 1: matched redistribute pairs (annual_redist vs decadal)
    for r in REDIST_MATCHED:
        ga, sa = invert_curve(arg, ard, r * V_ann_base)
        injected = (r - 1.0) * V_ann_base
        gd, sd = invert_curve(dg, dd, V_dec_base + injected)
        print(f"  [A1] matched r={r}: annual_redist gamma={ga} [{sa}] | "
              f"decadal gamma={gd} [{sd}]")
        if ga is not None and gd is not None:
            plan.append((f"annualredist_matched_{r}x", "annual_redist", ga, r, f"a1_{r}", 1))
            plan.append((f"decadal_matched_{r}x", "decadal", gd, r, f"a1_{r}", 1))
        else:
            print(f"       -> dropping matched pair r={r} (infeasible)")

    # AXIS 1 suppression (annual redistribute)
    ga, sa = invert_curve(arg, ard, SUPPRESS_RATIO * V_ann_base)
    print(f"  [A1] suppress r={SUPPRESS_RATIO}: annual_redist gamma={ga} [{sa}]")
    if ga is not None:
        plan.append((f"annualredist_suppressed_{SUPPRESS_RATIO}x",
                     "annual_redist", ga, SUPPRESS_RATIO, None, 1))

    # AXIS 2: seasonal amplification ladder
    for r in SEASONAL_LADDER:
        ga, sa = invert_curve(asg, asd, r * V_ann_base)
        print(f"  [A2] seasonal r={r}: annual_seasonal gamma={ga} [{sa}]")
        if ga is not None:
            plan.append((f"annualseasonal_{r}x", "annual_seasonal", ga, r,
                         f"a2_{r}", 2))

    # decadal maximum
    plan.append(("decadal_max", "decadal", max(DECADAL_GRID),
                 max_matched_r, None, 1))

    # ---- PHASE 3: run pipeline experiments ----
    print("\n" + "=" * 70)
    print("PHASE 3  run production ensembles")
    print("=" * 70)
    rows = []
    seasonal_total_ratios = {}   # r -> delivered total ratio (for white controls)
    for label, kind, gamma, target_r, group, axis in plan:
        ds = gen.generate(make_exp(Experiment, kind, gamma, common_final))
        arr = arr_of(ds)
        Va, Vd = ens_band_var(arr, ANNUAL_BAND), ens_band_var(arr, DECADAL_BAND)
        Vt, mean = ens_total_var(arr), float(np.mean(arr))
        injected = (Va - V_ann_base if "annual" in kind
                    else Vd - V_dec_base if kind == "decadal" else 0.0)
        rows.append(dict(label=label, kind=kind, axis=axis, gamma=gamma,
                         group=group, V_annual=Va, V_decadal=Vd, V_total=Vt,
                         mean=mean, annual_ratio=Va / V_ann_base,
                         decadal_ratio=Vd / V_dec_base,
                         total_ratio=Vt / V_tot_base, injected=injected))
        if kind == "annual_seasonal":
            seasonal_total_ratios[target_r] = Vt / V_tot_base
        save_array(base_ds, arr, os.path.join(OUTDIR, f"{label}.nc"))
        print(f"  ran {label}")

    # ---- PHASE 3b: white controls (AXIS 2) ----
    print("\n" + "-" * 70)
    print("  white controls (flat-spectrum total-variance match to AXIS 2)")
    for r, tot_ratio in sorted(seasonal_total_ratios.items()):
        wc_arr = make_white_control(base_arr, tot_ratio, V_tot_base,
                                    seed=SEED + int(round(r * 1000)))
        Va, Vd = ens_band_var(wc_arr, ANNUAL_BAND), ens_band_var(wc_arr, DECADAL_BAND)
        Vt, mean = ens_total_var(wc_arr), float(np.mean(wc_arr))
        label = f"white_control_{r}x"
        rows.append(dict(label=label, kind="white", axis=2, gamma=None,
                         group=f"a2_{r}", V_annual=Va, V_decadal=Vd, V_total=Vt,
                         mean=mean, annual_ratio=Va / V_ann_base,
                         decadal_ratio=Vd / V_dec_base,
                         total_ratio=Vt / V_tot_base, injected=0.0))
        save_array(base_ds, wc_arr, os.path.join(OUTDIR, f"{label}.nc"))
        print(f"  ran {label} (matched total {tot_ratio:.2f}x)")

    # ---- PHASE 4: report ----
    print("\n" + "=" * 70)
    print("PHASE 4  delivered-variance summary")
    print("=" * 70)
    hdr = (f"{'experiment':28} {'ax':>2} {'gamma':>8} {'ann':>7} {'dec':>8} "
           f"{'total':>7} {'injected':>11} {'mean':>10}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        g = "--" if r["gamma"] is None else f"{r['gamma']:.2f}"
        print(f"{r['label']:28} {r['axis']:>2} {g:>8} {r['annual_ratio']:6.2f}x "
              f"{r['decadal_ratio']:7.2f}x {r['total_ratio']:6.2f}x "
              f"{r['injected']:11.3e} {r['mean']:10.6f}")

    # AXIS 1 checks: matched injected variance AND total conserved
    print("\n  AXIS 1 - matched pairs (inject same band variance, "
          "conserve total):")
    groups = {}
    for r in rows:
        if r["group"] and r["group"].startswith("a1_"):
            groups.setdefault(r["group"], []).append(r)
    for grp, items in sorted(groups.items()):
        by = {it["kind"]: it for it in items}
        if "annual_redist" in by and "decadal" in by:
            ia, idd = by["annual_redist"]["injected"], by["decadal"]["injected"]
            ta, td = by["annual_redist"]["total_ratio"], by["decadal"]["total_ratio"]
            rel = abs(ia - idd) / abs(ia) if ia else float("nan")
            print(f"    {grp}: injected annual +{ia:.3e} vs decadal +{idd:.3e} "
                  f"(diff {rel:.1%}); total {ta:.2f}x vs {td:.2f}x")

    # AXIS 2 checks: seasonal vs white control at matched total variance
    print("\n  AXIS 2 - seasonal vs white control (matched TOTAL variance,"
          " differ in timescale):")
    for r in rows:
        if r["kind"] == "annual_seasonal":
            grp = r["group"]
            wc = next((x for x in rows if x["kind"] == "white"
                       and x["group"] == grp), None)
            if wc:
                print(f"    {grp}: seasonal total {r['total_ratio']:.2f}x "
                      f"(annual {r['annual_ratio']:.2f}x) vs white total "
                      f"{wc['total_ratio']:.2f}x (annual {wc['annual_ratio']:.2f}x)")

    mean_ok = all(np.isclose(r["mean"], mean_base, rtol=1e-4) for r in rows)
    print(f"\n  mean preserved across all experiments: {'YES' if mean_ok else 'NO'}")

    try:
        import pandas as pd
        csv = os.path.join(OUTDIR, "suite_summary.csv")
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"  wrote {csv} and {len(rows)} NetCDF ensembles to {OUTDIR}/")
    except Exception as e:                           # noqa: BLE001
        print(f"  (CSV skipped: {e})")
    print("\nDONE.")


if __name__ == "__main__":
    main()