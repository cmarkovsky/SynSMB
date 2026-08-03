import numpy as np
import xarray as xr
from syn_smb.core.generator import SMBGenerator      # adjust import to your layout
from syn_smb.core.experiment import Experiment


def annual_timescale_variance(series, fs=12.0):
    """Variance in the 0.8-1.5 yr band of a physical series, full-record."""
    x = np.asarray(series, float).ravel(); x = x - x.mean()
    X = np.fft.rfft(x); p = (np.abs(X) ** 2) / (fs * x.size); p[1:-1] *= 2
    f = np.fft.rfftfreq(x.size, d=1 / fs)
    m = (f >= 1 / 1.5) & (f <= 1 / 0.8)
    return float(np.sum(p[m]) * (f[1] - f[0]))


def test_annual_experiment_has_no_ceiling(pig_smb):   # pig_smb: observed DataArray fixture
    gen = SMBGenerator().fit(pig_smb)

    base = gen.generate(Experiment.baseline(n_years=1000, n_members=5, seed=0))
    v0 = annual_timescale_variance(base["smb_syn"].isel(member=0).values)

    delivered = {}
    for g in (2, 5, 10, 20):
        ds = gen.generate(Experiment.annual_enhanced(factor=g, n_years=1000,
                                                     n_members=5, seed=0))
        delivered[g] = annual_timescale_variance(
            ds["smb_syn"].isel(member=0).values) / v0

    print("\n gamma  delivered annual-timescale ratio")
    for g, r in delivered.items():
        print(f"  {g:3d}   {r:6.2f}x")

    # The old (unscaled-seasonal) code ceilings at ~3.6x, so gamma=20 must
    # clear it decisively. Expect roughly linear scaling.
    assert delivered[20] > 5.0, "annual experiment still saturating"
    assert delivered[10] > delivered[5] > delivered[2], "not monotone in gamma"


def test_mean_preserved_under_seasonal_scaling(pig_smb):
    gen = SMBGenerator().fit(pig_smb)
    base = gen.generate(Experiment.baseline(n_years=500, n_members=3, seed=1))
    amp  = gen.generate(Experiment.annual_enhanced(factor=10, n_years=500,
                                                   n_members=3, seed=1))
    assert np.isclose(float(base["smb_syn"].mean()),
                      float(amp["smb_syn"].mean()), rtol=1e-6), \
        "seasonal scaling shifted the mean"