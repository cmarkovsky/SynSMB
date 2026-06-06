"""
test_experiment_real_data.py
============================
Integration test: runs the full standard_suite of Experiments against
real RACMO data, end-to-end through the complete 1D pipeline.

Preprocessor → GaussianTransform → SpectralSynthesizer + Experiment
→ GaussianTransform.inverse_transform → Preprocessor.inverse_transform

Run with:
    python test_experiment_real_data.py /path/to/racmo_file.nc

Or edit RACMO_PATH below and run without arguments:
    python test_experiment_real_data.py
"""

import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import welch

from syn_smb import SMBDataLoader
from syn_smb import Preprocessor
from syn_smb import GaussianTransform
from syn_smb import SpectralSynthesizer
from syn_smb import Experiment


RACMO_PATH = "/path/to/your/racmo_file.nc"
SMB_VAR    = "smbgl"


def run(path: str, var: str = "smbgl") -> None:

    print("=" * 60)
    print("Experiment real data integration test")
    print("Full standard suite through complete 1D pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Build and fit the pipeline (done once, shared across all experiments)
    # ------------------------------------------------------------------
    print("\n[1] Loading and fitting pipeline on RACMO data...")
    smb = SMBDataLoader(path, var=var).load()
    smb_mean = float(smb.mean())

    pp = Preprocessor(remove_trend=True, remove_seasonal=True)
    residuals = pp.fit_transform(smb)

    gt = GaussianTransform()
    gt.fit(residuals)
    g_resid = gt.transform(residuals)

    ss = SpectralSynthesizer(nperseg=60, dt_years=1/12)
    ss.fit(g_resid.values if hasattr(g_resid, 'values') else g_resid)

    print(f"    Pipeline fitted. n_obs={ss.n_obs}, n_segments={ss.n_segments}")

    # ------------------------------------------------------------------
    # Step 2: Print the standard suite
    # ------------------------------------------------------------------
    print("\n[2] Standard experiment suite...")
    suite = Experiment.standard_suite(n_years=1000, n_members=30, seed=0)
    for exp in suite:
        exp.summary()
        print()

    # ------------------------------------------------------------------
    # Step 3: Run each experiment and collect ensemble statistics
    # ------------------------------------------------------------------
    print("[3] Running all experiments...")
    results = {}

    for exp in suite:
        print(f"    Running: {exp.name}...", end=" ", flush=True)

        ensemble_g = ss.synthesize(
            n_years=exp.n_years,
            n_members=exp.n_members,
            band_scales=exp.band_scales,
            rng=exp.rng,
        )

        syn_time = xr.cftime_range(start="1979", periods=ensemble_g.shape[1], freq="MS")

        smb_members = []
        for i in range(ensemble_g.shape[0]):
            g_da     = xr.DataArray(ensemble_g[i], coords={"time": syn_time}, dims=["time"])
            resid    = gt.inverse_transform(g_da)
            smb_syn  = resid + smb_mean
            smb_full = pp.inverse_transform(smb_syn, add_trend=False)
            smb_members.append(smb_full)

        all_vals = np.concatenate([m.values for m in smb_members])

        # Compute ensemble-mean PSD
        member_psds = []
        for i in range(ensemble_g.shape[0]):
            _, p = welch(ensemble_g[i], fs=ss.fs, nperseg=ss.nperseg)
            member_psds.append(p)
        mean_psd = np.array(member_psds).mean(axis=0)

        results[exp.name] = {
            "experiment": exp,
            "smb_mean":   all_vals.mean(),
            "smb_std":    all_vals.std(),
            "smb_min":    all_vals.min(),
            "smb_max":    all_vals.max(),
            "mean_psd":   mean_psd,
            "smb_members": smb_members,
        }
        print("done")

    # ------------------------------------------------------------------
    # Step 4: Compare results across experiments
    # ------------------------------------------------------------------
    print("\n[4] Experiment comparison (reconstructed SMB statistics)...")
    print(f"    {'Experiment':<35}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    print(f"    {'Observed':<35}  {smb_mean:>8.4f}  {float(smb.std()):>8.4f}  "
          f"{float(smb.min()):>8.4f}  {float(smb.max()):>8.4f}")
    print(f"    {'-'*67}")
    for name, r in results.items():
        print(f"    {name:<35}  {r['smb_mean']:>8.4f}  {r['smb_std']:>8.4f}  "
              f"{r['smb_min']:>8.4f}  {r['smb_max']:>8.4f}")

    # ------------------------------------------------------------------
    # Step 5: Mean preservation check across all experiments
    # ------------------------------------------------------------------
    print("\n[5] Mean preservation check...")
    print(f"    Observed SMB mean: {smb_mean:.5f}")
    all_passed = True
    for name, r in results.items():
        diff = abs(r['smb_mean'] - smb_mean)
        status = "✓" if diff < 0.005 else "✗"
        print(f"    {name:<35}  mean={r['smb_mean']:.5f}  diff={diff:.2e}  {status}")
        if diff >= 0.005:
            all_passed = False
    print(f"\n    All means preserved: {'YES ✓' if all_passed else 'NO ✗'}")

    # ------------------------------------------------------------------
    # Step 6: Serialisation round-trip
    # ------------------------------------------------------------------
    print("\n[6] Serialisation round-trip for each experiment...")
    for exp in suite:
        restored = Experiment.from_dict(exp.to_dict())
        match = (restored == exp)
        print(f"    {exp.name:<35}  round-trip: {'✓' if match else '✗'}")

    # ------------------------------------------------------------------
    # Step 7: Diagnostic plot
    # ------------------------------------------------------------------
    print("\n[7] Generating diagnostic plot...")
    _plot_diagnostics(smb, results, ss)

    print("\n" + "=" * 60)
    print("Integration test complete.")
    print("=" * 60)


def _plot_diagnostics(smb, results, ss):
    """
    Two-panel diagnostic figure:
      left:  PSD comparison across all experiments (Gaussian space)
      right: reconstructed SMB time series — observed vs one member
             per experiment (first 45 years of synthetic)
    """
    n_show = len(smb)
    colors = {
        "baseline":              "tab:blue",
        "annual_enhanced_10.0x": "tab:orange",
        "annual_suppressed_0.1x":"tab:green",
        "decadal_enhanced_10.0x":"tab:red",
        "decadal_suppressed_0.1x":"tab:purple",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Experiment standard suite — real data diagnostics", fontsize=13)

    # ---- left: PSD comparison ----
    ax = axes[0]
    ax.loglog(ss.freqs[1:], ss.psd[1:],
              color="black", lw=2, linestyle="--", label="Observed (fitted PSD)", zorder=5)
    ax.fill_between(ss.freqs[1:], ss.psd_ci_lower[1:], ss.psd_ci_upper[1:],
                    color="gray", alpha=0.15, label="95% CI")

    for name, r in results.items():
        color = colors.get(name, "gray")
        ax.loglog(ss.freqs[1:], r["mean_psd"][1:],
                  color=color, lw=1.2, alpha=0.85, label=name)

    ax.axvline(1.0, color="gray", linestyle=":", lw=1, alpha=0.6)
    ax.axvline(0.1, color="gray", linestyle=":", lw=1, alpha=0.6)
    ax.set_xlabel("Frequency (cycles/yr)")
    ax.set_ylabel("PSD (Gaussianized space)")
    ax.set_title("Ensemble mean PSD by experiment\n(Gaussianized residuals)")
    ax.legend(fontsize=7, loc="lower left")

    # ---- right: SMB time series comparison ----
    ax = axes[1]
    ax.plot(smb['time'].values, smb.values,
            color="black", lw=1.2, label="Observed", zorder=5, alpha=0.9)

    for name, r in results.items():
        color = colors.get(name, "gray")
        member = r["smb_members"][0]
        ax.plot(smb['time'].values, member.values[:n_show],
                color=color, lw=0.8, alpha=0.7, label=name)

    ax.axhline(float(smb.mean()), color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("SMB (m w.e. a⁻¹)")
    ax.set_title("Reconstructed SMB — one member per experiment\n(first 45 yrs of synthetic)")
    ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig("experiment_real_data_diagnostics.png", dpi=150, bbox_inches="tight")
    print("    Saved: experiment_real_data_diagnostics.png")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else RACMO_PATH
    var  = sys.argv[2] if len(sys.argv) > 2 else SMB_VAR
    run(path, var)