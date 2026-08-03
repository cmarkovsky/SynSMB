"""
run_seasonal_test.py
====================
Verifies the seasonal-scaling fix on the real PIG pipeline.

WHAT IT CHECKS
--------------
1. Delivered annual-TIMESCALE variance of the final physical series scales
   with the requested factor and does NOT saturate.
2. The record mean is unchanged by seasonal scaling (invariant 1).
3. (optional) The raw `seasonal_scale` knob works independently, if the fix
   is applied.

WHY IT MATTERS
--------------
The deterministic seasonal cycle carries ~66% of PIG annual-timescale
variance and was previously added back UNSCALED, so `annual_enhanced` saturated
at ~3.6x regardless of the requested factor. This script measures the delivered
ratio directly and is the acceptance test for the fix.

The measurement uses a FULL-RECORD periodogram, not Welch nperseg=60 — the
latter cannot resolve the band structure you are trying to measure.

HOW TO RUN
----------
    poetry run python tests/run_seasonal_test.py

Edit the CONFIG block below: the data path and, if needed, the import path.
The script runs against whatever code is installed, so:
  - BEFORE applying the fix: expect the ceiling (~3.6x) and FAIL.
  - AFTER applying the fix : expect near-linear scaling and PASS.
Running it both ways is the cleanest demonstration that the fix works.
"""

from __future__ import annotations
import sys
import numpy as np

# =====================================================================
#  CONFIG — EDIT THIS
# =====================================================================
RACMO_PATH = "./data/PIG_smb.nc"     # your observed SMB NetCDF
SMB_VAR    = "smbgl"                  # SMB variable name

# Import path for your package. Adjust if your layout differs.
# The 2-D code used `syn_smb.core.*`, so that is the default guess.
IMPORT_CANDIDATES = [
    "syn_smb.core",
    "syn_smb",
    "src.syn_smb.core",
    "core",
]

GAMMAS      = [2, 5, 10, 20]   # requested annual factors to test
N_YEARS     = 1000
N_MEMBERS   = 5
SEED        = 0
CEILING_REF = 3.64             # the old (broken) saturation ceiling for PIG
# =====================================================================


def _import_pipeline():
    """Locate SMBGenerator and Experiment regardless of exact package name."""
    last_err = None
    for base in IMPORT_CANDIDATES:
        try:
            gen_mod = __import__(f"{base}.generator", fromlist=["SMBGenerator"])
            exp_mod = __import__(f"{base}.experiment", fromlist=["Experiment"])
            return gen_mod.SMBGenerator, exp_mod.Experiment, base
        except Exception as e:                       # noqa: BLE001
            last_err = e
    print("ERROR: could not import SMBGenerator / Experiment.")
    print(f"Tried: {IMPORT_CANDIDATES}")
    print(f"Last error: {last_err}")
    print("Fix: add your package's import path to IMPORT_CANDIDATES at the top,")
    print("or run from a directory where the package is importable")
    print("(e.g. `poetry run python tests/run_seasonal_test.py` from the repo root).")
    sys.exit(1)


def annual_timescale_variance(series, fs=12.0):
    """
    Variance in the 0.8-1.5 yr band of a physical series, measured with a
    full-record periodogram. Verified against a pure sinusoid: recovers
    A^2/2 exactly.
    """
    x = np.asarray(series, dtype=float).ravel()
    x = x - x.mean()
    X = np.fft.rfft(x)
    p = (np.abs(X) ** 2) / (fs * x.size)
    p[1:-1] *= 2.0
    f = np.fft.rfftfreq(x.size, d=1.0 / fs)
    m = (f >= 1.0 / 1.5) & (f <= 1.0 / 0.8)
    return float(np.sum(p[m]) * (f[1] - f[0]))


def total_variance(series):
    return float(np.var(np.asarray(series, dtype=float).ravel()))


def main():
    SMBGenerator, Experiment, base_pkg = _import_pipeline()
    print(f"Imported pipeline from '{base_pkg}'.")
    print(f"Fitting on {RACMO_PATH} (var='{SMB_VAR}') ...\n")

    try:
        gen = SMBGenerator.from_path(RACMO_PATH, var=SMB_VAR)
    except FileNotFoundError:
        print(f"ERROR: data file not found: {RACMO_PATH}")
        print("Edit RACMO_PATH at the top of this script.")
        sys.exit(1)

    common = dict(n_years=N_YEARS, n_members=N_MEMBERS, seed=SEED)

    # --- baseline ---
    base = gen.generate(Experiment.baseline(**common))
    s0   = base["smb_syn"].isel(member=0).values
    v0   = annual_timescale_variance(s0)
    tot0 = total_variance(s0)
    mean0 = float(base["smb_syn"].mean())

    # --- annual_enhanced sweep ---
    print("=" * 60)
    print("TEST 1 — delivered annual-timescale variance vs requested factor")
    print("=" * 60)
    print(f"  baseline annual-timescale variance = {v0:.6e}")
    print(f"  (old broken code saturates near {CEILING_REF:.2f}x)\n")
    print(f"  {'gamma':>6} {'delivered ratio':>16} {'total var ratio':>16}")

    delivered = {}
    for g in GAMMAS:
        ds = gen.generate(Experiment.annual_enhanced(factor=g, **common))
        s  = ds["smb_syn"].isel(member=0).values
        delivered[g] = annual_timescale_variance(s) / v0
        tot_ratio = total_variance(s) / tot0
        print(f"  {g:6d} {delivered[g]:15.2f}x {tot_ratio:15.3f}")

    # --- mean preservation ---
    print("\n" + "=" * 60)
    print("TEST 2 — mean preserved under seasonal scaling")
    print("=" * 60)
    ds10  = gen.generate(Experiment.annual_enhanced(factor=10, **common))
    mean10 = float(ds10["smb_syn"].mean())
    mean_ok = np.isclose(mean0, mean10, rtol=1e-6)
    print(f"  baseline mean     = {mean0:.8f}")
    print(f"  annual x10 mean   = {mean10:.8f}")
    print(f"  relative diff     = {abs(mean10-mean0)/abs(mean0):.2e}")
    print(f"  {'PASS' if mean_ok else 'FAIL'}")

    # --- optional: raw seasonal_scale knob (only exists after the fix) ---
    print("\n" + "=" * 60)
    print("TEST 3 — raw seasonal_scale knob (skipped if fix not applied)")
    print("=" * 60)
    seasonal_knob_ok = None
    try:
        exp = Experiment.baseline(**common)
        exp.seasonal_scale = 9.0            # variance x9 -> amplitude x3
        # re-validate if the dataclass supports it
        if hasattr(exp, "_validate"):
            exp._validate()
        ds_s = gen.generate(exp)
        # with band_scales=None, ONLY the seasonal cycle is scaled
        v_s = annual_timescale_variance(ds_s["smb_syn"].isel(member=0).values)
        ratio = v_s / v0
        # seasonal is ~66% of annual-timescale variance; x9 on that part gives
        # roughly 0.66*9 + 0.34 ~ 6.3x. Just check it clearly exceeds the ceiling.
        seasonal_knob_ok = ratio > 4.0
        print(f"  seasonal_scale=9.0 -> annual-timescale ratio = {ratio:.2f}x")
        print(f"  {'PASS' if seasonal_knob_ok else 'FAIL'}")
    except (AttributeError, TypeError) as e:
        print(f"  SKIPPED — seasonal_scale not present yet ({e}).")
        print("  Apply the fix (experiment.py / preprocessor.py / generator.py),")
        print("  then re-run.")

    # --- verdict ---
    print("\n" + "#" * 60)
    print("VERDICT")
    print("#" * 60)
    no_ceiling = delivered[max(GAMMAS)] > CEILING_REF + 1.0
    monotone   = all(delivered[a] < delivered[b]
                     for a, b in zip(GAMMAS, GAMMAS[1:]))
    print(f"  no ceiling (gamma={max(GAMMAS)} clears {CEILING_REF:.1f}x): "
          f"{'YES' if no_ceiling else 'NO'}  ({delivered[max(GAMMAS)]:.2f}x)")
    print(f"  monotone in gamma                          : "
          f"{'YES' if monotone else 'NO'}")
    print(f"  mean preserved                             : "
          f"{'YES' if mean_ok else 'NO'}")

    passed = no_ceiling and monotone and mean_ok
    if passed:
        print("\n  RESULT: PASS — seasonal-scaling fix is working.")
        sys.exit(0)
    else:
        print("\n  RESULT: FAIL.")
        if not no_ceiling:
            print("  The annual experiment is still saturating. Either the fix")
            print("  is not applied, or annual_enhanced is not setting")
            print("  seasonal_scale. Check experiment.py.")
        sys.exit(2)


# ---- pytest entry points (optional): `poetry run pytest tests/run_seasonal_test.py`
def test_no_ceiling():
    """Importable as a pytest test; delegates to main()'s checks."""
    SMBGenerator, Experiment, _ = _import_pipeline()
    gen = SMBGenerator.from_path(RACMO_PATH, var=SMB_VAR)
    common = dict(n_years=500, n_members=3, seed=0)
    base = gen.generate(Experiment.baseline(**common))
    v0 = annual_timescale_variance(base["smb_syn"].isel(member=0).values)
    ds = gen.generate(Experiment.annual_enhanced(factor=20, **common))
    ratio = annual_timescale_variance(ds["smb_syn"].isel(member=0).values) / v0
    assert ratio > CEILING_REF + 1.0, f"still saturating: {ratio:.2f}x"


def test_mean_preserved():
    SMBGenerator, Experiment, _ = _import_pipeline()
    gen = SMBGenerator.from_path(RACMO_PATH, var=SMB_VAR)
    common = dict(n_years=500, n_members=3, seed=1)
    base = gen.generate(Experiment.baseline(**common))
    amp  = gen.generate(Experiment.annual_enhanced(factor=10, **common))
    assert np.isclose(float(base["smb_syn"].mean()),
                      float(amp["smb_syn"].mean()), rtol=1e-6)


if __name__ == "__main__":
    main()