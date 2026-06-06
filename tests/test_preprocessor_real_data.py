"""
test_preprocessor_real_data.py
===============================
Integration test: SMBDataLoader → Preprocessor → GaussianTransform
→ SpectralSynthesizer on real RACMO data.

Run with:
    python test_preprocessor_real_data.py /path/to/racmo_file.nc

Or edit RACMO_PATH below and run without arguments:
    python test_preprocessor_real_data.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from syn_smb import SMBDataLoader
from syn_smb import Preprocessor
from syn_smb import GaussianTransform
from syn_smb import SpectralSynthesizer


RACMO_PATH = "/path/to/your/racmo_file.nc"
SMB_VAR    = "smbgl"


def run(path: str, var: str = "smbgl") -> None:

    print("=" * 60)
    print("Real data integration test")
    print("DataLoader → Preprocessor → GaussianTransform → SpectralSynthesizer")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("\n[1] Loading data...")
    loader = SMBDataLoader(path, var=var)
    try:
        smb = loader.load()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)
    loader.summarize()

    # ------------------------------------------------------------------
    # Step 2: Preprocessor — fit and transform
    # ------------------------------------------------------------------
    print("\n[2] Preprocessor — fit and transform...")
    pp = Preprocessor(remove_trend=True, remove_seasonal=True)
    pp.fit(smb)
    pp.summarize()
    residuals = pp.transform(smb)

    print(f"\n    After preprocessing:")
    print(f"    mean:          {float(residuals.mean()):+.4e}  (target: ~0)")
    print(f"    std:           {float(residuals.std()):.5f}")

    monthly_means = residuals.groupby('time.month').mean()
    print(f"    monthly mean range: "
          f"[{float(monthly_means.min()):.2e}, {float(monthly_means.max()):.2e}]"
          f"  (target: ~0)")

    fit_check = residuals.polyfit(dim='time', deg=1)
    slope = float(fit_check['polyfit_coefficients'].sel(degree=1).values)
    print(f"    residual slope: {slope:.2e}  (target: ~0)")

    # ------------------------------------------------------------------
    # Step 3: Stationarity checks — before and after preprocessing
    # ------------------------------------------------------------------
    print("\n[3] Stationarity check on raw SMB...")
    pp.check_stationarity(smb, verbose=True)

    print("\n[4] Stationarity check on preprocessed residuals...")
    pp.check_stationarity(residuals, verbose=True)

    # ------------------------------------------------------------------
    # Step 4: GaussianTransform
    # ------------------------------------------------------------------
    print("\n[5] GaussianTransform — fit and transform...")
    gt = GaussianTransform()
    gt.fit(residuals)
    results_gt = gt.validate(residuals, verbose=True)
    g_resid = gt.transform(residuals)
    g_vals = g_resid.values

    # ------------------------------------------------------------------
    # Step 5: SpectralSynthesizer — compare PSD before/after preprocessing
    # ------------------------------------------------------------------
    print("\n[6] SpectralSynthesizer — fit on preprocessed Gaussianized residuals...")
    ss = SpectralSynthesizer(nperseg=60, dt_years=1/12)
    ss.fit(g_vals)
    print(f"    {ss}")

    # Also fit on raw (non-preprocessed) Gaussianized residuals for comparison
    smb_mean = float(smb.mean())
    raw_residuals = smb - smb_mean
    gt_raw = GaussianTransform()
    gt_raw.fit(raw_residuals)
    g_raw = gt_raw.transform(raw_residuals)
    ss_raw = SpectralSynthesizer(nperseg=60, dt_years=1/12)
    ss_raw.fit(g_raw.values)

    print(f"\n    PSD at f=1.0 (annual):")
    print(f"    With preprocessing:    {_psd_at(ss, 1.0):.4e}")
    print(f"    Without preprocessing: {_psd_at(ss_raw, 1.0):.4e}")
    print(f"    Reduction factor:      {_psd_at(ss_raw, 1.0) / _psd_at(ss, 1.0):.2f}x")
    print(f"\n    PSD at f=0.1 (decadal):")
    print(f"    With preprocessing:    {_psd_at(ss, 0.1):.4e}")
    print(f"    Without preprocessing: {_psd_at(ss_raw, 0.1):.4e}")

    # ------------------------------------------------------------------
    # Step 6: Synthesize and validate full chain
    # ------------------------------------------------------------------
    print("\n[7] Synthesize and reconstruct (10 members, 100 years)...")
    rng = np.random.default_rng(0)
    import xarray as xr
    ensemble_g = ss.synthesize(n_years=100, n_members=10, rng=rng)

    synthetic_time = xr.cftime_range(start="1979", periods=ensemble_g.shape[1], freq="MS")

    smb_members = []
    for i in range(ensemble_g.shape[0]):
        g_da = xr.DataArray(ensemble_g[i], coords={"time": synthetic_time}, dims=["time"])
        resid_syn = gt.inverse_transform(g_da)
        smb_syn = resid_syn + smb_mean
        smb_reconstructed = pp.inverse_transform(smb_syn, add_trend=False)
        smb_members.append(smb_reconstructed)

    print(f"    Reconstructed SMB stats (across all members):")
    all_vals = np.concatenate([m.values for m in smb_members])
    print(f"    mean: {all_vals.mean():.5f}  (observed: {float(smb.mean()):.5f})")
    print(f"    std:  {all_vals.std():.5f}   (observed: {float(smb.std()):.5f})")
    print(f"    min:  {all_vals.min():.5f}   (observed: {float(smb.min()):.5f})")
    print(f"    max:  {all_vals.max():.5f}   (observed: {float(smb.max()):.5f})")

    # ------------------------------------------------------------------
    # Step 7: Diagnostic plot
    # ------------------------------------------------------------------
    print("\n[8] Generating diagnostic plot...")
    _plot_diagnostics(smb, residuals, ss, ss_raw, smb_members, pp)

    print("\n" + "=" * 60)
    print("Integration test complete.")
    print("=" * 60)


def _psd_at(ss, target_freq):
    idx = np.argmin(np.abs(ss.freqs - target_freq))
    return float(ss.psd[idx])


def _plot_diagnostics(smb, residuals, ss, ss_raw, smb_members, pp):
    """
    Four-panel diagnostic figure:
      top-left:     raw SMB vs detrended deseasoned residuals
      top-right:    PSD comparison — before vs after preprocessing
      bottom-left:  seasonal cycle extracted by Preprocessor
      bottom-right: synthetic ensemble members vs observed SMB
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Preprocessor — real data diagnostics", fontsize=13)

    # ---- top-left: raw vs residuals ----
    ax = axes[0, 0]
    ax.plot(smb['time'].values, smb.values,
            lw=0.8, color="tab:blue", label="Raw SMB", alpha=0.8)
    ax.plot(residuals['time'].values, residuals.values,
            lw=0.8, color="tab:orange", label="Preprocessed residuals", alpha=0.8)
    ax.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("SMB (m w.e. a⁻¹)")
    ax.set_title("Raw SMB vs preprocessed residuals")
    ax.legend(fontsize=9)

    # ---- top-right: PSD before vs after ----
    ax = axes[0, 1]
    ax.loglog(ss_raw.freqs[1:], ss_raw.psd[1:],
              color="tab:blue", lw=1.5, label="Without preprocessing")
    ax.fill_between(ss_raw.freqs[1:], ss_raw.psd_ci_lower[1:], ss_raw.psd_ci_upper[1:],
                    color="tab:blue", alpha=0.1)
    ax.loglog(ss.freqs[1:], ss.psd[1:],
              color="tab:orange", lw=1.5, label="After preprocessing")
    ax.fill_between(ss.freqs[1:], ss.psd_ci_lower[1:], ss.psd_ci_upper[1:],
                    color="tab:orange", alpha=0.1)
    ax.axvline(1.0, color="gray", linestyle="--", lw=1, alpha=0.7)
    ax.axvline(0.1, color="gray", linestyle=":",  lw=1, alpha=0.7)
    ax.set_xlabel("Frequency (cycles/yr)")
    ax.set_ylabel("PSD (Gaussianized residuals)")
    ax.set_title("PSD: effect of preprocessing\n"
                 "(annual peak should shrink after seasonal removal)")
    ax.legend(fontsize=9)

    # ---- bottom-left: seasonal cycle ----
    ax = axes[1, 0]
    months = np.arange(1, 13)
    month_names = ['J','F','M','A','M','J','J','A','S','O','N','D']
    sc = pp.seasonal_cycle.values
    ax.bar(months, sc, color="tab:green", alpha=0.7, edgecolor="white")
    ax.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_xlabel("Month")
    ax.set_ylabel("SMB anomaly (m w.e. a⁻¹)")
    ax.set_title("Extracted seasonal cycle\n(monthly climatology from detrended data)")

    # ---- bottom-right: synthetic ensemble vs observed ----
    ax = axes[1, 1]
    n_show = len(smb)  # show same length as observed
    for i, member in enumerate(smb_members):
        ax.plot(smb['time'].values, member.values[:n_show],
                color="tab:orange", lw=0.6, alpha=0.4,
                label="Synthetic" if i == 0 else None)
    ax.plot(smb['time'].values, smb.values,
            color="tab:blue", lw=1.2, label="Observed", alpha=0.9)
    ax.axhline(float(smb.mean()), color="tab:blue",
               linestyle="--", lw=1, alpha=0.6)
    ax.set_xlabel("Time")
    ax.set_ylabel("SMB (m w.e. a⁻¹)")
    ax.set_title("Synthetic ensemble vs observed\n(first 45 yrs of synthetic members)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("preprocessor_real_data_diagnostics.png", dpi=150, bbox_inches="tight")
    print("    Saved: preprocessor_real_data_diagnostics.png")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else RACMO_PATH
    var  = sys.argv[2] if len(sys.argv) > 2 else SMB_VAR
    run(path, var)