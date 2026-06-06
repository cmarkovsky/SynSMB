"""
test_generator_real_data.py
============================
Integration test: complete end-to-end pipeline on real RACMO data.
SMBGenerator.from_path() → fit → validate → generate_suite → save

Run with:
    python test_generator_real_data.py /path/to/racmo_file.nc

Or edit RACMO_PATH below:
    python test_generator_real_data.py
"""

import sys
import os
import tempfile
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import welch

from syn_smb.core.generator import SMBGenerator
from syn_smb.core.experiment import Experiment


RACMO_PATH = "/path/to/your/racmo_file.nc"
SMB_VAR    = "smbgl"


def run(path: str, var: str = "smbgl") -> None:

    print("=" * 60)
    print("SMBGenerator — full pipeline real data test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load and fit
    # ------------------------------------------------------------------
    print(f"\n[1] Loading and fitting from {path}...")
    gen = SMBGenerator.from_path(path, var=var)
    print(f"    {gen}")
    gen.summarize()

    # ------------------------------------------------------------------
    # Step 2: Validate the fitted pipeline
    # ------------------------------------------------------------------
    print("\n[2] Validating fitted pipeline...")
    val_results = gen.validate(verbose=True)

    # ------------------------------------------------------------------
    # Step 3: Generate standard suite
    # ------------------------------------------------------------------
    print("\n[3] Generating standard suite (30 members, 1000 years)...")
    suite = Experiment.standard_suite(n_years=1000, n_members=30, seed=0)
    datasets = gen.generate_suite(suite)
    print(f"    Experiments generated: {list(datasets.keys())}")

    # ------------------------------------------------------------------
    # Step 4: Mean preservation across all experiments
    # ------------------------------------------------------------------
    print("\n[4] Mean preservation check across all experiments...")
    print(f"    Observed SMB mean: {gen.smb_mean:.5f} m w.e. a⁻¹")
    print(f"    {'Experiment':<35}  {'syn mean':>10}  {'diff':>10}  {'status':>6}")
    all_passed = True
    for name, ds in datasets.items():
        m    = float(ds["smb_syn"].values.ravel().mean())
        diff = abs(m - gen.smb_mean)
        ok   = diff < 0.005
        if not ok:
            all_passed = False
        print(f"    {name:<35}  {m:>10.5f}  {diff:>10.2e}  {'✓' if ok else '✗'}")
    print(f"\n    All means preserved: {'YES ✓' if all_passed else 'NO ✗'}")

    # ------------------------------------------------------------------
    # Step 5: SMB statistics comparison
    # ------------------------------------------------------------------
    print("\n[5] SMB statistics (mean ± std across all members)...")
    from data_loader import SMBDataLoader
    smb_obs = SMBDataLoader(path, var=var).load()
    print(f"    {'Experiment':<35}  {'mean':>8}  {'std':>8}")
    print(f"    {'Observed':<35}  {float(smb_obs.mean()):>8.4f}  {float(smb_obs.std()):>8.4f}")
    print(f"    {'-'*55}")
    for name, ds in datasets.items():
        vals = ds["smb_syn"].values.ravel()
        print(f"    {name:<35}  {vals.mean():>8.4f}  {vals.std():>8.4f}")

    # ------------------------------------------------------------------
    # Step 6: Save one ensemble to NetCDF
    # ------------------------------------------------------------------
    print("\n[6] Saving baseline ensemble to NetCDF...")
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmppath = f.name
    gen.save(datasets["baseline"], tmppath, experiment=suite[0])
    loaded = xr.open_dataset(tmppath)
    print(f"    Saved and reloaded successfully.")
    print(f"    Variables: {list(loaded.data_vars)}")
    print(f"    Dimensions: {dict(loaded.dims)}")
    print(f"    Attributes: {dict(loaded.attrs)}")
    os.unlink(tmppath)

    # ------------------------------------------------------------------
    # Step 7: Diagnostic plot
    # ------------------------------------------------------------------
    print("\n[7] Generating diagnostic plot...")
    _plot_diagnostics(smb_obs, gen, datasets, suite)

    print("\n" + "=" * 60)
    print("Integration test complete.")
    print("=" * 60)


def _plot_diagnostics(smb_obs, gen, datasets, suite):
    """
    Three-panel diagnostic figure:
      left:   PSD of g_syn ensemble mean vs observed, for each experiment
      centre: synthetic SMB time series — first 45 yrs, one member per experiment
      right:  annual cycle — observed vs synthetic ensemble mean
    """
    colors = {
        "baseline":               "tab:blue",
        "annual_enhanced_10.0x":  "tab:orange",
        "annual_suppressed_0.1x": "tab:green",
        "decadal_enhanced_10.0x": "tab:red",
        "decadal_suppressed_0.1x":"tab:purple",
    }
    n_show = gen.n_obs

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("SMBGenerator — full pipeline diagnostics", fontsize=13)

    # ---- left: PSD by experiment ----
    ax = axes[0]
    ax.loglog(gen.freqs[1:], gen.psd[1:],
              color="black", lw=2, linestyle="--", label="Observed PSD", zorder=5)
    ax.fill_between(gen.freqs[1:], gen.psd_ci_lower[1:], gen.psd_ci_upper[1:],
                    color="gray", alpha=0.15)
    for name, ds in datasets.items():
        psds = []
        for i in range(ds.sizes["member"]):
            _, p = welch(ds["g_syn"].isel(member=i).values,
                         fs=gen.spectral_synthesizer.fs,
                         nperseg=gen.nperseg)
            psds.append(p)
        mean_psd = np.array(psds).mean(axis=0)
        ax.loglog(gen.freqs[1:], mean_psd[1:],
                  color=colors.get(name, "gray"), lw=1, alpha=0.8, label=name)
    ax.axvline(1.0, color="gray", linestyle=":", lw=1, alpha=0.5)
    ax.axvline(0.1, color="gray", linestyle=":", lw=1, alpha=0.5)
    ax.set_xlabel("Frequency (cycles/yr)")
    ax.set_ylabel("PSD (Gaussianized space)")
    ax.set_title("Ensemble mean PSD by experiment")
    ax.legend(fontsize=7, loc="lower left")

    # ---- centre: SMB time series ----
    ax = axes[1]
    ax.plot(smb_obs['time'].values, smb_obs.values,
            color="black", lw=1.2, label="Observed", zorder=5, alpha=0.9)
    for name, ds in datasets.items():
        member = ds["smb_syn"].isel(member=0)
        ax.plot(smb_obs['time'].values, member.values[:n_show],
                color=colors.get(name, "gray"), lw=0.8, alpha=0.6, label=name)
    ax.set_xlabel("Time")
    ax.set_ylabel("SMB (m w.e. a⁻¹)")
    ax.set_title("First 45 yrs synthetic vs observed")
    ax.legend(fontsize=7)

    # ---- right: annual cycle ----
    ax = axes[2]
    months = np.arange(1, 13)
    month_labels = ['J','F','M','A','M','J','J','A','S','O','N','D']
    obs_monthly = np.array([
        float(smb_obs.sel(time=smb_obs.time.dt.month == m).mean())
        for m in months
    ])
    ax.plot(months, obs_monthly, color="black", lw=2,
            marker="o", markersize=4, label="Observed", zorder=5)
    for name, ds in datasets.items():
        all_vals = ds["smb_syn"].values.ravel()
        # Flatten and compute monthly means across all members and years
        n_members = ds.sizes["member"]
        n_time = ds.sizes["time"]
        month_indices = np.tile(np.arange(1, 13), n_time // 12)
        syn_monthly = np.array([
            all_vals[np.where(np.tile(month_indices, n_members) == m)].mean()
            for m in months
        ])
        ax.plot(months, syn_monthly,
                color=colors.get(name, "gray"), lw=1, alpha=0.7,
                marker=".", markersize=3, label=name)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean SMB (m w.e. a⁻¹)")
    ax.set_title("Annual cycle: observed vs synthetic\n(all members, all years)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("generator_real_data_diagnostics.png", dpi=150, bbox_inches="tight")
    print("    Saved: generator_real_data_diagnostics.png")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else RACMO_PATH
    var  = sys.argv[2] if len(sys.argv) > 2 else SMB_VAR
    run(path, var)