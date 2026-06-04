"""
test_validator_real_data.py
============================
Integration test: complete Validator suite on real RACMO data.

Produces all validation figures and quantitative metrics.

Run with:
    python test_validator_real_data.py /path/to/racmo_file.nc
"""

import sys
import numpy as np
import xarray as xr

from syn_smb.core.generator import SMBGenerator
from syn_smb.core.experiment import Experiment
from syn_smb.core.validator import Validator


RACMO_PATH = "/path/to/your/racmo_file.nc"
SMB_VAR    = "smbgl"


def run(path: str, var: str = "smbgl") -> None:

    print("=" * 60)
    print("Validator — real data integration test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Fit pipeline
    # ------------------------------------------------------------------
    print("\n[1] Fitting pipeline...")
    gen = SMBGenerator.from_path(path, var=var)
    smb = gen._residuals  # use stored residuals to reconstruct observed SMB
    # Reload full observed SMB for the Validator
    from data_loader import SMBDataLoader
    smb_obs = SMBDataLoader(path, var=var).load()
    val = Validator(gen, smb_obs)
    print(f"    {val}")

    # ------------------------------------------------------------------
    # Generate standard suite
    # ------------------------------------------------------------------
    print("\n[2] Generating standard suite (30 members, 1000 years)...")
    suite    = Experiment.standard_suite(n_years=1000, n_members=30, seed=0)
    datasets = gen.generate_suite(suite)

    # ------------------------------------------------------------------
    # Compute metrics for all experiments
    # ------------------------------------------------------------------
    print("\n[3] Metrics across standard suite...")
    print(f"    {'Experiment':<35}  {'mean_r':>7}  {'var_r':>7}  "
          f"{'ks':>7}  {'psd_rms':>8}  {'seas_rmse':>10}")
    for name, ds in datasets.items():
        m = val.compute_metrics(ds, verbose=False)
        print(f"    {name:<35}  {m['mean_ratio']:>7.4f}  "
              f"{m['variance_ratio']:>7.4f}  {m['ks_statistic']:>7.4f}  "
              f"{m['psd_rms_error']:>8.4f}  {m['seasonal_rmse']:>10.6f}")

    # ------------------------------------------------------------------
    # Calibration split test
    # ------------------------------------------------------------------
    print("\n[4] Calibration split test...")
    split_results = val.calibration_split_test(
        n_members=30, n_years_syn=100, verbose=True
    )

    # ------------------------------------------------------------------
    # Convergence test
    # ------------------------------------------------------------------
    print("\n[5] Convergence test...")
    conv = val.convergence_test(
        member_counts=[5, 10, 15, 20, 30, 50],
        n_years=100,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Produce all figures for the baseline experiment
    # ------------------------------------------------------------------
    print("\n[6] Producing validation figures (baseline experiment)...")
    ds_baseline = datasets["baseline"]

    val.plot_validation_suite(
        ds_baseline,
        save_path="validator_validation_suite.png",
    )
    val.plot_ensemble_spaghetti(
        ds_baseline,
        save_path="validator_ensemble_spaghetti.png",
    )
    val.plot_running_windows(
        ds_baseline,
        save_path="validator_running_windows.png",
    )
    val.plot_return_periods(
        ds_baseline,
        save_path="validator_return_periods.png",
    )

    print("\n" + "=" * 60)
    print("Integration test complete.")
    print("=" * 60)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else RACMO_PATH
    var  = sys.argv[2] if len(sys.argv) > 2 else SMB_VAR
    run(path, var)