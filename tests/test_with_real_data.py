"""
test_with_real_data.py
======================
Integration test: SMBDataLoader → GaussianTransform on real RACMO data.

Run with:
    python test_with_real_data.py /path/to/your/racmo_file.nc

Or edit RACMO_PATH below and run without arguments:
    python test_with_real_data.py
"""

import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from syn_smb.core.data_loader import SMBDataLoader
from syn_smb.core.gaussianize import GaussianTransform


# --- Edit this if you want to run without a command-line argument ---
RACMO_PATH = "/data/PIG_smb.nc"  # <-- Update this to your local path if not using CLI argument
SMB_VAR    = "smbgl"
# -------------------------------------------------------------------


def run(path: str, var: str = "smbgl") -> None:

    print("=" * 60)
    print("Real data integration test")
    print("SMBDataLoader → GaussianTransform")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print(f"\n[1] Loading data from: {path}")
    loader = SMBDataLoader(path, var=var)

    try:
        smb = loader.load()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    loader.summarize()

    # ------------------------------------------------------------------
    # Step 2: Compute a simple residual
    # (full Preprocessor comes later — for now just remove the mean
    #  so we have a zero-mean series to pass to GaussianTransform)
    # ------------------------------------------------------------------
    print("\n[2] Computing residuals (mean removal only — Preprocessor not built yet)...")
    smb_mean = float(smb.mean())
    residuals = smb - smb_mean
    print(f"    smb mean removed:   {smb_mean:.5f}")
    print(f"    residual mean:      {float(residuals.mean()):.2e}  (should be ~0)")
    print(f"    residual std:       {float(residuals.std()):.5f}")

    # ------------------------------------------------------------------
    # Step 3: Fit GaussianTransform
    # ------------------------------------------------------------------
    print("\n[3] Fitting GaussianTransform...")
    gt = GaussianTransform()
    gt.fit(residuals)
    print(f"    {gt}")

    # ------------------------------------------------------------------
    # Step 4: Forward transform
    # ------------------------------------------------------------------
    print("\n[4] Forward transform (residuals → Gaussian)...")
    g = gt.transform(residuals)
    g_vals = g.values if isinstance(g, xr.DataArray) else g
    print(f"    output mean:  {np.mean(g_vals):+.6f}  (target: ~0)")
    print(f"    output std:   {np.std(g_vals):.6f}   (target: ~1)")
    print(f"    output min:   {np.min(g_vals):.4f}")
    print(f"    output max:   {np.max(g_vals):.4f}")
    print(f"    any NaN/Inf:  {np.any(np.isnan(g_vals)) or np.any(np.isinf(g_vals))}")

    # ------------------------------------------------------------------
    # Step 5: Inverse transform
    # ------------------------------------------------------------------
    print("\n[5] Inverse transform (Gaussian → residuals)...")
    r = gt.inverse_transform(g)
    r_vals = r.values if isinstance(r, xr.DataArray) else r
    resid_vals = residuals.values if isinstance(residuals, xr.DataArray) else residuals
    x_centered = resid_vals - np.mean(resid_vals)
    max_err = np.max(np.abs(r_vals - x_centered))
    print(f"    recovered mean:      {np.mean(r_vals):+.2e}  (mean shift fix — should be ~0)")
    print(f"    max round-trip err:  {max_err:.2e}  (tolerance: 1e-6)")

    # ------------------------------------------------------------------
    # Step 6: Mean shift check — simulate band scaling
    # ------------------------------------------------------------------
    print("\n[6] Mean shift check — inverse_transform on scaled Gaussian inputs...")
    for scale in [0.1, 2.0, 5.0, 10.0]:
        g_scaled = g_vals * scale
        r_scaled = gt.inverse_transform(g_scaled)
        r_scaled_vals = r_scaled.values if isinstance(r_scaled, xr.DataArray) else r_scaled
        print(f"    scale={scale:4.1f}x  →  mean: {np.mean(r_scaled_vals):+.2e}  "
              f"std: {np.std(r_scaled_vals):.5f}")

    # ------------------------------------------------------------------
    # Step 7: Full validate()
    # ------------------------------------------------------------------
    print("\n[7] GaussianTransform.validate()...")
    results = gt.validate(residuals, verbose=True)

    # ------------------------------------------------------------------
    # Step 8: Diagnostic plot
    # ------------------------------------------------------------------
    print("\n[8] Generating diagnostic plot...")
    _plot_diagnostics(smb, residuals, g, r, gt)

    print("\n" + "=" * 60)
    print("Integration test complete.")
    print("=" * 60)


def _plot_diagnostics(smb, residuals, g, r, gt):
    """
    Four-panel diagnostic figure:
      top-left:     observed SMB time series
      top-right:    residuals vs recovered residuals (round-trip check)
      bottom-left:  histogram of Gaussianized residuals vs N(0,1)
      bottom-right: QQ plot of Gaussianized residuals
    """
    from scipy.stats import norm, probplot

    smb_vals     = smb.values if isinstance(smb, xr.DataArray) else smb
    resid_vals   = residuals.values if isinstance(residuals, xr.DataArray) else residuals
    g_vals       = g.values if isinstance(g, xr.DataArray) else g
    r_vals       = r.values if isinstance(r, xr.DataArray) else r

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GaussianTransform — real data diagnostics", fontsize=13)

    # -- top-left: SMB time series --
    ax = axes[0, 0]
    if isinstance(smb, xr.DataArray) and "time" in smb.coords:
        ax.plot(smb["time"].values, smb_vals, lw=0.8, color="tab:blue")
    else:
        ax.plot(smb_vals, lw=0.8, color="tab:blue")
    ax.axhline(np.mean(smb_vals), color="tab:orange", linestyle="--", lw=1, label="mean")
    ax.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax.set_title("Observed SMB")
    ax.set_ylabel("SMB (m w.e. a⁻¹)")
    ax.legend(fontsize=9)

    # -- top-right: residuals vs round-trip recovered --
    ax = axes[0, 1]
    x_centered = resid_vals - np.mean(resid_vals)
    ax.plot(x_centered, lw=0.8, color="tab:blue", label="original (centered)", alpha=0.8)
    ax.plot(r_vals, lw=0.8, color="tab:orange", linestyle="--", label="round-trip recovered", alpha=0.8)
    max_err = np.max(np.abs(r_vals - x_centered))
    ax.set_title(f"Round-trip check (max error: {max_err:.2e})")
    ax.set_ylabel("Residual (m w.e. a⁻¹)")
    ax.legend(fontsize=9)

    # -- bottom-left: histogram of Gaussianized residuals vs N(0,1) --
    ax = axes[1, 0]
    ax.hist(g_vals, bins=30, density=True, color="tab:blue", alpha=0.6, label="g_resid")
    x_range = np.linspace(-4, 4, 200)
    ax.plot(x_range, norm.pdf(x_range), "k--", lw=1.5, label="N(0,1)")
    ax.set_title("Gaussianized residuals vs N(0,1)")
    ax.set_xlabel("g")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    # -- bottom-right: QQ plot --
    ax = axes[1, 1]
    (osm, osr), (slope, intercept, _) = probplot(g_vals, dist="norm")
    ax.scatter(osm, osr, s=8, color="tab:blue", alpha=0.5, label="data quantiles")
    ax.plot(osm, slope * np.array(osm) + intercept, "k--", lw=1.5, label="N(0,1) reference")
    ax.set_title("QQ plot: Gaussianized vs N(0,1)")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("gaussianize_real_data_diagnostics.png", dpi=150, bbox_inches="tight")
    print("    Saved: gaussianize_real_data_diagnostics.png")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else RACMO_PATH
    var  = sys.argv[2] if len(sys.argv) > 2 else SMB_VAR
    run(path, var)