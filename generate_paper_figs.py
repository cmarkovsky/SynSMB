# ---- setup.py ----
import numpy as np, xarray as xr, matplotlib
matplotlib.use("Agg")
from scipy.signal import welch
from scipy import stats
from scipy.stats import norm
import figure_generation as fg
fg.set_style()

# adjust to your package layout (the 2-D code used syn_smb.core.*)
from syn_smb.core.generator   import SMBGenerator
from syn_smb.core.data_loader import SMBDataLoader

RACMO = "./data/PIG_smb.nc"; VAR = "smbgl"

SUITE = "./results/suite"
def load(name):
    return xr.open_dataset(f"{SUITE}/{name}.nc")["smb_syn"].values  # (member, time)

# --- fit once (or reuse the generator you already fitted) ---
gen = SMBGenerator.from_path(RACMO, var=VAR)

# --- observed series (gen does not retain it; load the same way from_path does) ---
smb = SMBDataLoader(RACMO, var=VAR).load()          # xr.DataArray (time,)
obs = np.asarray(smb.values)

# --- the three decomposition pieces of Eq. (1) ---
# trend line: evaluate the stored polynomial coefficients on the time axis
trend       = np.asarray(xr.polyval(smb["time"], gen.preprocessor.trend_coeffs).values)
# seasonal climatology: 12 monthly anomalies (month coord 1..12)
seasonal_12 = np.asarray(gen.seasonal_cycle.values)
# stochastic residual r(t): preprocessor.transform removes trend + seasonal
resid_da    = gen.preprocessor.transform(smb)
resid       = np.asarray(resid_da.values)

# --- Gaussianised residual g(t) ~ N(0,1) ---
g_resid = np.asarray(gen.gaussian_transform.transform(resid_da))

# --- FITTED spectrum + CI, GAUSSIAN space, straight from the synthesizer ---
# (use these for Fig. 4 "the fitted model" — label the axis "Gaussian space")
freqs   = gen.freqs
psd_g   = gen.psd
ci_lo_g = gen.psd_ci_lower
ci_hi_g = gen.psd_ci_upper

# --- OBSERVED spectrum + CI, PHYSICAL space, for the validation figure ---
# Fig. 6 compares against physical synthetic series, so both sides must be
# physical. Compute the observed full-series PSD with the same estimator.
f_phys, psd_phys = welch(obs - obs.mean(), fs=12.0, nperseg=60)
nu = 34
ci_lo_p = nu * psd_phys / stats.chi2.ppf(0.975, nu)
ci_hi_p = nu * psd_phys / stats.chi2.ppf(0.025, nu)

# --- inverse-CDF callables for Fig. 3 (functions of u = Phi(g)) ---
gt = gen.gaussian_transform
q_semi = lambda u: float(gt.inverse_transform(np.array([norm.ppf(u)]))[0])  # A2 spliced
rs = np.sort(resid); pp = (np.arange(1, rs.size + 1) - 0.5) / rs.size
q_emp  = lambda u: float(np.interp(u, pp, rs))    # empirical-only (clips at obs min/max)

### Figure 2

# fg.fig02_decomposition(obs, trend, seasonal_12, resid, "figs/fig02.png")

### Figure 3
factors = [1, 2, 5, 10, 20]
# sat_semi, sat_emp = realised variance ratios with semi-parametric vs
# empirical-only inverse CDF (toggle the parametric tail off for sat_emp)
# fg.fig03_transform(resid, g_resid, q_semi, q_emp, factors, q_semi, q_emp, "figs/fig03.pdf")
# from scipy.stats import norm

# ss, gt = gen.spectral_synthesizer, gen.gaussian_transform

# def inverse_empirical(g_vals):                    # the OLD clipping behaviour
#     u = np.clip(norm.cdf(np.asarray(g_vals)), 1e-15, 1 - 1e-15)
#     r = np.interp(u, gt.quantiles, gt.x_sorted)   # np.interp clips at x_sorted[0], [-1]
#     return r - r.mean()

# # one long Gaussian ensemble (baseline spectrum is enough — A2 is about the tail)
# g_ens = ss.synthesize(n_years=1000, n_members=10, band_scales=None,
#                       rng=np.random.default_rng(0))          # (members, N)

# def return_levels(residual_2d):
#     """Pooled residual -> (return_period_yr, level) via Weibull positions."""
#     vals = np.sort(np.asarray(residual_2d).ravel())[::-1]
#     rp = (vals.size + 1) / (np.arange(1, vals.size + 1))      # in months
#     return rp / 12.0, vals

# semi_res = np.array([np.asarray(gt.inverse_transform(g)) for g in g_ens])
# emp_res  = np.array([inverse_empirical(g)               for g in g_ens])
# rp_semi, lv_semi = return_levels(semi_res)
# rp_emp,  lv_emp  = return_levels(emp_res)
# obs_max = float(np.max(resid))    # observed 45-yr max, for the reference line

# fg.fig03_transform(resid, g_resid, q_semi, q_emp, rp_semi, lv_semi, rp_emp, lv_emp, obs_max,"figs/fig03_transform.png")

### Figure 4
# fg.fig04_psd_ci(freqs, psd_g, ci_lo_g, ci_hi_g,          # Gaussian-space fitted model
#                 bands=(("annual", 0.8, 1.5), ("decadal", 8.0, 20.0)),
#                 path="figs/fig04_psd_ci.png")

### Figure 6

# --- baseline + suite ensembles (written by run_experiment_suite.py) ---

base_ens = load("baseline")
# fg.fig06_validation_suite(obs, base_ens, f_phys, psd_phys, ci_lo_p, ci_hi_p,
#                           fs=12.0, nperseg=60, nlags=36, path="figs/fig06_validation_suite.png")  # physical space

### Figure. 7 return period curves (semi-parametric vs empirical-only)
fg.fig09_return_period(obs, base_ens, path="figs/fig07_return_period.png")

### Figure 8 band demo (annual vs decadal, seasonal vs white control)
#
axis1 = {
    "baseline":            load("baseline"),
    "annual (redist) 3x":  load("annualredist_matched_3.0x"),
    "decadal (matched)":   load("decadal_matched_3.0x"),
}
fg.fig12_band_demo(axis1, path="figs/fig08_axis1.png")

axis2 = {
    "baseline":      load("baseline"),
    "seasonal 10x":  load("annualseasonal_10.0x"),
    "white control": load("white_control_10.0x"),
}
fg.fig12_band_demo(axis2, path="figs/fig08_axis2.png")