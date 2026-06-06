"""
test_spectral_real_data.py
==========================
Integration test: SMBDataLoader → GaussianTransform → SpectralSynthesizer
on real RACMO data.

Run with:
    python test_spectral_real_data.py /path/to/racmo_file.nc

Or edit RACMO_PATH below and run without arguments:
    python test_spectral_real_data.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from syn_smb import SMBDataLoader
from syn_smb import GaussianTransform
from syn_smb import SpectralSynthesizer


# --- Edit this if running without a command-line argument ---
RACMO_PATH = "/path/to/your/racmo_file.nc"
SMB_VAR    = "smbgl"
# ------------------------------------------------------------


def run(path: str, var: str = "smbgl") -> None:

    print("=" * 60)
    print("Real data integration test")
    print("DataLoader → GaussianTransform → SpectralSynthesizer")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load and inspect data
    # ------------------------------------------------------------------
    print(f"\n[1] Loading data...")
    loader = SMBDataLoader(path, var=var)
    try:
        smb = loader.load()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)
    loader.summarize()

    # ------------------------------------------------------------------
    # Step 2: Compute residuals and fit GaussianTransform
    # ------------------------------------------------------------------
    print("\n[2] GaussianTransform — fit and forward transform...")
    smb_mean = float(smb.mean())
    residuals = smb - smb_mean

    gt = GaussianTransform()
    gt.fit(residuals)
    results_gt = gt.validate(residuals, verbose=True)

    g_resid = gt.transform(residuals)
    g_vals = g_resid.values if hasattr(g_resid, 'values') else g_resid
    print(f"    Gaussianized residuals: mean={np.mean(g_vals):+.4f}, std={np.std(g_vals):.4f}")

    # ------------------------------------------------------------------
    # Step 3: Fit SpectralSynthesizer on Gaussianized residuals
    # ------------------------------------------------------------------
    print("\n[3] SpectralSynthesizer — fit...")
    ss = SpectralSynthesizer(nperseg=60, dt_years=1/12)
    ss.fit(g_vals)
    print(f"    {ss}")
    print(f"    Freq range:    [{ss.freqs[1]:.4f}, {ss.freqs[-1]:.4f}] cycles/yr  (skipping DC)")
    print(f"    PSD range:     [{ss.psd[1:].min():.4e}, {ss.psd[1:].max():.4e}]")
    print(f"    CI width:      mean upper/lower ratio = "
          f"{(ss.psd_ci_upper[1:] / ss.psd_ci_lower[1:]).mean():.2f}x")
    print(f"    Annual peak:   PSD at f=1.0 = {_psd_at_freq(ss, 1.0):.4e}")
    print(f"    Decadal band:  PSD at f=0.1 = {_psd_at_freq(ss, 0.1):.4e}")

    # ------------------------------------------------------------------
    # Step 4: Synthesize a baseline ensemble
    # ------------------------------------------------------------------
    print("\n[4] Synthesize baseline ensemble (30 members, 1000 years)...")
    rng = np.random.default_rng(0)
    ensemble = ss.synthesize(n_years=1000, n_members=30, rng=rng)
    print(f"    Output shape:     {ensemble.shape}")
    variances = ensemble.var(axis=1)
    means     = ensemble.mean(axis=1)
    print(f"    Variance (mean):  {variances.mean():.6f}  (target: 1.0)")
    print(f"    Variance (std):   {variances.std():.6f}  (should be small)")
    print(f"    Mean (mean):      {means.mean():.4e}    (target: ~0)")

    # ------------------------------------------------------------------
    # Step 5: Spectral fidelity of ensemble
    # ------------------------------------------------------------------
    print("\n[5] Spectral fidelity — ensemble mean PSD vs observed...")
    member_psds = []
    for i in range(ensemble.shape[0]):
        _, p = welch(ensemble[i], fs=ss.fs, nperseg=ss.nperseg)
        member_psds.append(p)
    member_psds = np.array(member_psds)
    mean_psd    = member_psds.mean(axis=0)
    ratio       = mean_psd / ss.psd

    print(f"    Ratio ensemble mean PSD / observed PSD:")
    print(f"    min={ratio[1:].min():.3f}, max={ratio[1:].max():.3f}, "
          f"mean={ratio[1:].mean():.3f}  (target: all ~1.0)")

    within = (mean_psd >= ss.psd_ci_lower) & (mean_psd <= ss.psd_ci_upper)
    print(f"    Fraction within 95% CI: {within[1:].mean():.3f}  (target: ~1.0)")

    # ------------------------------------------------------------------
    # Step 6: Band scaling — std growth check
    # ------------------------------------------------------------------
    print("\n[6] Band scaling — annual band (0.8–1.5 yr periods)...")
    print("    Each member still has unit variance after scaling.")
    rng2 = np.random.default_rng(0)
    for factor in [0.1, 1.0, 2.0, 5.0, 10.0]:
        out = ss.synthesize(
            n_years=200, n_members=10,
            band_scales=[(0.8, 1.5, factor)],
            rng=np.random.default_rng(0),
        )
        v = out.var(axis=1)
        print(f"    annual_scale={factor:4.1f}x  → variance: {v.mean():.6f} ± {v.std():.6f}")

    # ------------------------------------------------------------------
    # Step 7: Full validate()
    # ------------------------------------------------------------------
    print("\n[7] SpectralSynthesizer.validate()...")
    ss.validate(n_check=200, verbose=True)

    # ------------------------------------------------------------------
    # Step 8: Diagnostic plot
    # ------------------------------------------------------------------
    print("\n[8] Generating diagnostic plot...")
    _plot_diagnostics(smb, g_vals, ss, ensemble, member_psds)

    print("\n" + "=" * 60)
    print("Integration test complete.")
    print("=" * 60)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _psd_at_freq(ss: SpectralSynthesizer, target_freq: float) -> float:
    """Return interpolated PSD at the nearest frequency to target_freq."""
    idx = np.argmin(np.abs(ss.freqs - target_freq))
    return float(ss.psd[idx])


def _plot_diagnostics(smb, g_vals, ss, ensemble, member_psds):
    """
    Four-panel diagnostic figure:
      top-left:     observed PSD with 95% CI and ensemble spaghetti
      top-right:    ensemble mean PSD vs observed, ratio panel
      bottom-left:  one synthetic member vs observed time series
                    (first 45 years of synthetic, same length as observed)
      bottom-right: distribution of ensemble member variances
    """
    n_obs_years = int(len(g_vals) * ss.dt_years)
    n_obs_steps = len(g_vals)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("SpectralSynthesizer — real data diagnostics", fontsize=13)

    # ---- top-left: PSD spaghetti + observed + CI ----
    ax = axes[0, 0]
    for i in range(min(30, member_psds.shape[0])):
        ax.loglog(ss.freqs[1:], member_psds[i, 1:],
                  color="tab:orange", alpha=0.15, lw=0.7)
    mean_psd = member_psds.mean(axis=0)
    ax.loglog(ss.freqs[1:], mean_psd[1:],
              color="tab:red", lw=1.5, label="Ensemble mean")
    ax.loglog(ss.freqs[1:], ss.psd[1:],
              color="tab:blue", lw=2, label="Observed")
    ax.fill_between(ss.freqs[1:], ss.psd_ci_lower[1:], ss.psd_ci_upper[1:],
                    color="tab:blue", alpha=0.15, label="95% CI")
    ax.axvline(1.0, color="orange", linestyle="--", lw=1, alpha=0.7)
    ax.axvline(0.1, color="green",  linestyle="--", lw=1, alpha=0.7)
    ax.set_xlabel("Frequency (cycles/yr)")
    ax.set_ylabel("PSD")
    ax.set_title("PSD: observed vs ensemble (Gaussianized residuals)")
    ax.legend(fontsize=8)

    # ---- top-right: ratio of ensemble mean PSD to observed ----
    ax = axes[0, 1]
    ratio = mean_psd[1:] / ss.psd[1:]
    ax.semilogx(ss.freqs[1:], ratio, color="tab:purple", lw=1.5)
    ax.axhline(1.0, color="gray", linestyle="--", lw=1)
    ax.axhline(2.0, color="gray", linestyle=":",  lw=0.8)
    ax.axhline(0.5, color="gray", linestyle=":",  lw=0.8)
    ax.axvline(1.0, color="orange", linestyle="--", lw=1, alpha=0.7)
    ax.axvline(0.1, color="green",  linestyle="--", lw=1, alpha=0.7)
    ax.set_xlabel("Frequency (cycles/yr)")
    ax.set_ylabel("Ensemble mean PSD / Observed PSD")
    ax.set_title("Spectral fidelity ratio  (target: ~1.0 across all freqs)")
    ax.set_ylim(0, 3)

    # ---- bottom-left: first synthetic member vs observed ----
    ax = axes[1, 0]
    t = np.arange(n_obs_steps)
    ax.plot(t, g_vals, lw=0.8, color="tab:blue", label="Observed", alpha=0.8)
    ax.plot(t, ensemble[0, :n_obs_steps], lw=0.8,
            color="tab:orange", label="Synthetic member 0", alpha=0.8)
    ax.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax.set_xlabel(f"Months (first {n_obs_years} yrs of synthetic)")
    ax.set_ylabel("Gaussianized residual")
    ax.set_title("Time series comparison (Gaussianized space)")
    ax.legend(fontsize=9)

    # ---- bottom-right: distribution of member variances ----
    ax = axes[1, 1]
    variances = ensemble.var(axis=1)
    ax.hist(variances, bins=15, color="tab:green", alpha=0.7, edgecolor="white")
    ax.axvline(1.0, color="black", linestyle="--", lw=1.5, label="Target = 1.0")
    ax.axvline(variances.mean(), color="tab:red", linestyle="-",
               lw=1.5, label=f"Mean = {variances.mean():.4f}")
    ax.set_xlabel("Member variance")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of ensemble member variances\n"
                 "(all should be 1.0 — normalisation check)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("spectral_real_data_diagnostics.png", dpi=150, bbox_inches="tight")
    print("    Saved: spectral_real_data_diagnostics.png")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else RACMO_PATH
    var  = sys.argv[2] if len(sys.argv) > 2 else SMB_VAR
    run(path, var)