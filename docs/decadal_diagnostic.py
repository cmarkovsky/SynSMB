"""
decadal_diagnostic.py  (v2)
===========================
Answers, on YOUR real data: how much genuine decadal (8-20 yr) variance does
the PIG SMB residual contain, and how much of it does an nperseg=60 Welch fit
actually see?

WHAT CHANGED IN v2
------------------
v1 used the variance of non-overlapping k-year block means as a "grid-free"
measure of low-frequency power. That estimator is structurally broken for this
purpose: block-averaging over L years is a boxcar filter whose gain on a
period-P sinusoid is |sinc(L/P)|, which is EXACTLY ZERO when L = P. A 10-year
block mean therefore suppresses ~96 % of a 12-year cycle, and a 12-year block
mean is completely blind to it. That is why v1 reported "consistent with white
noise" at k=10 for a signal that was 66 % decadal by construction.

v2 replaces it with a band-integrated spectral estimate on the FULL record:
  - multitaper (DPSS) with NW chosen so the resolution bandwidth
    2*NW/(N*dt) fits INSIDE the target band -> NW = 1.5 for a 45-yr record
    and an 8-20 yr band (2W = 0.067 < 0.075 cyc/yr band width);
  - cross-checked against the raw full-record periodogram (unbiased, noisier).
Validation on a planted 12-yr cycle with true band variance 0.4972:
    multitaper NW=1.5 -> 0.4991    periodogram -> 0.4757
    (v1 block-means   -> 0.0090, a factor of ~55 too small)

Significance is assessed by Monte Carlo against white-noise and AR(1)
surrogates, with the IDENTICAL estimator applied to data and surrogates, so
any residual estimator bias cancels in the comparison.

USAGE
-----
Edit the CONFIG block at the bottom - you only need the path to the raw RACMO
NetCDF. Then:

    python decadal_diagnostic.py

Requires numpy, scipy, xarray (+ netCDF4 or h5netcdf). geopandas and
regionmask are optional, used only if you supply a shapefile for masking.
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.signal import welch
from scipy.signal.windows import dpss


# =====================================================================
#  PART 1 - Loading raw RACMO data
# =====================================================================

_SMB_ALIASES = ["smbgl", "smb", "SMB", "smbcorr", "precip", "snowfall"]
_LAT_ALIASES = ["lat", "latitude", "LAT", "nav_lat"]
_LON_ALIASES = ["lon", "longitude", "LON", "nav_lon"]


def load_smb_series(
    racmo_path: str,
    smb_var: str | None = None,
    shp_path: str | None = None,
    basin_name: str | None = None,
    name_col: str = "NAME",
    verbose: bool = True,
):
    """
    Load a raw RACMO NetCDF and return a 1-D monthly basin-mean SMB series.

    If `shp_path` and `basin_name` are given (and geopandas + regionmask are
    installed) the field is masked to that basin before averaging. Otherwise
    the mean is taken over all valid (non-NaN) grid cells in the file, which
    is appropriate if the file is already subset to your basin.

    Returns
    -------
    series : (T,) float ndarray
    meta   : dict
    """
    import xarray as xr

    ds = xr.open_dataset(racmo_path)

    # --- resolve the SMB variable ---
    var = smb_var
    if var is None:
        var = next((a for a in _SMB_ALIASES if a in ds), None)
    if var is None:
        raise ValueError(
            f"Could not find an SMB variable. Searched {_SMB_ALIASES}. "
            f"Available: {list(ds.data_vars)}. Pass smb_var= explicitly."
        )
    da = ds[var]

    # --- units: kg m-2 -> m w.e. ---
    units = str(da.attrs.get("units", "")).strip()
    converted = False
    if "kg" in units and ("m-2" in units or "m^-2" in units
                          or "m**-2" in units):
        da = da / 1000.0
        converted = True

    # --- drop size-1 dims (e.g. height) ---
    size1 = [d for d in da.dims if da.sizes[d] == 1]
    if size1:
        da = da.squeeze(size1, drop=True)

    if "time" not in da.dims:
        raise ValueError(
            f"No 'time' dimension in '{var}'. dims={list(da.dims)}"
        )

    spatial_dims = [d for d in da.dims if d != "time"]

    # --- optional basin masking ---
    masked_by = "all valid cells in file"
    if shp_path and basin_name and spatial_dims:
        try:
            import geopandas as gpd
            import regionmask

            lat_name = next((a for a in _LAT_ALIASES if a in ds), None)
            lon_name = next((a for a in _LON_ALIASES if a in ds), None)
            if lat_name is None or lon_name is None:
                raise RuntimeError("lat/lon not found for masking")

            lat, lon = ds[lat_name], ds[lon_name]
            if lat.ndim == 1 and lon.ndim == 1:
                lon2d, lat2d = np.meshgrid(lon.values, lat.values)
                lat = xr.DataArray(lat2d, dims=spatial_dims)
                lon = xr.DataArray(lon2d, dims=spatial_dims)
            lon = xr.where(lon > 180, lon - 360, lon)

            gdf = gpd.read_file(shp_path)
            gdf = gdf[gdf[name_col] == basin_name].copy()
            if gdf.empty:
                raise RuntimeError(f"basin '{basin_name}' not in {name_col}")
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            bad = ~gdf.geometry.is_valid
            if bad.any():
                gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].buffer(0)
            if len(gdf) > 1:
                gc = gdf.geometry.name
                gdf = gdf[[gc]].dissolve().reset_index(drop=True)
                gdf[name_col] = basin_name

            regions = regionmask.from_geopandas(
                gdf, names=name_col, overlap=True
            )
            mask = regions.mask_3D(lon.values, lat.values).isel(region=0)
            mask = xr.DataArray(mask.values.astype(bool), dims=spatial_dims)
            da = da.where(mask)
            masked_by = f"basin '{basin_name}' ({int(mask.sum())} cells)"
        except Exception as e:                       # noqa: BLE001
            warnings.warn(
                f"Basin masking failed ({e}); falling back to the mean over "
                f"all valid cells. Install geopandas+regionmask, or "
                f"pre-subset the file to your basin."
            )

    # --- collapse to a scalar series ---
    if spatial_dims:
        series = da.mean(dim=spatial_dims, skipna=True).values
    else:
        series = da.values
    series = np.asarray(series, dtype=float).ravel()

    meta = {
        "var": var,
        "units": "m w.e." if converted else (units or "unknown"),
        "converted_kg_to_m": converted,
        "n_time": int(series.size),
        "spatial_mean_over": masked_by,
        "n_nan": int(np.sum(~np.isfinite(series))),
    }

    if verbose:
        print("-" * 68)
        print("DATA LOADED")
        print("-" * 68)
        print(f"  file            : {racmo_path}")
        print(f"  variable        : {var}")
        print(f"  units           : {meta['units']}"
              f"{'  (converted from kg m-2)' if converted else ''}")
        print(f"  spatial mean    : {masked_by}")
        print(f"  n time steps    : {series.size} "
              f"({series.size/12:.1f} yr at monthly resolution)")
        print(f"  mean / std      : {np.nanmean(series):.5f} / "
              f"{np.nanstd(series):.5f}")
        if meta["n_nan"]:
            print(f"  WARNING: {meta['n_nan']} non-finite values in series")
        print()

    return series, meta


def preprocess(series, deg: int = 1, fs: int = 12, verbose: bool = True):
    """
    Remove mean, linear trend, and monthly climatology (Eq. 1 of the paper).

    Returns
    -------
    resid : (T,) stochastic residual
    parts : dict with 'mean', 'trend', 'seasonal_12', 'coeffs'
    """
    x = np.asarray(series, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        n_bad = int(np.sum(~np.isfinite(x)))
        warnings.warn(f"{n_bad} non-finite values; interpolating over them.")
        idx = np.arange(x.size)
        good = np.isfinite(x)
        x = np.interp(idx, idx[good], x[good])

    raw_var = float(np.var(x))
    mu = float(x.mean())
    t = np.arange(x.size, dtype=float)
    coeffs = np.polyfit(t, x - mu, deg)
    trend = np.polyval(coeffs, t)
    detr = x - mu - trend

    seasonal_12 = np.array([detr[i::fs].mean() for i in range(fs)])
    seasonal_12 = seasonal_12 - seasonal_12.mean()
    seas_full = np.tile(seasonal_12,
                        int(np.ceil(x.size / fs)))[: x.size]

    resid = detr - seas_full

    if verbose:
        print("-" * 68)
        print("PREPROCESSING  (mean + linear trend + monthly climatology)")
        print("-" * 68)
        print(f"  record mean          : {mu:.6f}")
        print(f"  total trend change   : {trend[-1]-trend[0]:+.6f}")
        print(f"  seasonal peak-to-peak: "
              f"{seasonal_12.max()-seasonal_12.min():.6f}")
        print(f"  residual mean / std  : {resid.mean():.2e} / "
              f"{resid.std():.6f}")
        print(f"  variance removed     : "
              f"{100*(1 - resid.var()/raw_var):.1f} % of raw variance")
        print()

    return resid, {"mean": mu, "trend": trend,
                   "seasonal_12": seasonal_12, "coeffs": coeffs}


# =====================================================================
#  PART 2 - Spectral estimators (the v2 fix)
# =====================================================================

def multitaper_psd(x, fs=12.0, NW=1.5, K=None):
    """
    One-sided multitaper PSD using DPSS tapers, on the FULL record.

    NW sets the half-bandwidth; the resolution bandwidth is 2*NW/(N*dt).
    Choose NW so this is SMALLER than the width of the band you want to
    measure, otherwise power leaks out of the band and you under-read it.
    For a 45-yr monthly record and the 8-20 yr band (width 0.075 cyc/yr),
    NW=1.5 gives 2W = 0.067 cyc/yr, which fits.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    N = x.size
    if K is None:
        K = max(1, int(2 * NW - 1))
    tapers = dpss(N, NW, K)
    psds = []
    for w in tapers:
        X = np.fft.rfft(w * x)
        p = (np.abs(X) ** 2) / fs
        p[1:-1] *= 2.0
        psds.append(p)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    return f, np.mean(psds, axis=0)


def periodogram_psd(x, fs=12.0):
    """Raw one-sided periodogram of the full record (unbiased, noisy)."""
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    N = x.size
    X = np.fft.rfft(x)
    p = (np.abs(X) ** 2) / (fs * N)
    p[1:-1] *= 2.0
    return np.fft.rfftfreq(N, d=1.0 / fs), p


def band_variance(f, p, period_lo_yr, period_hi_yr):
    """Integrate a PSD over a period band. Returns (variance, n_bins)."""
    f = np.asarray(f)
    p = np.asarray(p)
    df = f[1] - f[0]
    m = (f >= 1.0 / period_hi_yr) & (f <= 1.0 / period_lo_yr)
    return float(np.sum(p[m]) * df), int(m.sum())


def resolution_bandwidth(n_time, NW, fs=12.0):
    """Multitaper resolution bandwidth 2*NW/(N*dt), in cycles per year."""
    return 2.0 * NW / (n_time / fs)


def welch_grid(x, fs, nperseg, detrend="constant"):
    """Welch PSD helper returning (f, p) suitable for band_variance."""
    x = np.asarray(x, dtype=float).ravel()
    nps = int(min(nperseg, x.size))
    return welch(x, fs=fs, nperseg=nps, detrend=detrend)


# =====================================================================
#  PART 3 - Monte Carlo significance
# =====================================================================

def _ar1_fit(x):
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    if x.size < 3:
        return 0.0, float(x.std())
    phi = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    phi = float(np.clip(phi, -0.99, 0.99))
    sigma = float(x.std() * np.sqrt(max(1e-12, 1.0 - phi ** 2)))
    return phi, sigma


def _ar1_surrogate(n, phi, sigma, rng):
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


def monte_carlo_band_test(resid, period_lo_yr, period_hi_yr, fs=12.0,
                          NW=1.5, n_surr=500, seed=0, null="both"):
    """
    Is the observed band variance larger than a null process would give?

    The SAME estimator is applied to the data and to the surrogates, so
    estimator bias cancels in the comparison.
    """
    resid = np.asarray(resid, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    n = resid.size

    f, p = multitaper_psd(resid, fs=fs, NW=NW)
    obs, nb = band_variance(f, p, period_lo_yr, period_hi_yr)

    out = {"observed": obs, "n_bins": nb, "nulls": {}}

    nulls = []
    if null in ("white", "both"):
        nulls.append(("white noise", None))
    if null in ("ar1", "both"):
        nulls.append(("AR(1)", _ar1_fit(resid)))

    for label, params in nulls:
        vals = np.empty(n_surr)
        for i in range(n_surr):
            if params is None:
                y = rng.normal(0.0, resid.std(), size=n)
            else:
                phi, sigma = params
                y = _ar1_surrogate(n, phi, sigma, rng)
            ff, pp = multitaper_psd(y, fs=fs, NW=NW)
            vals[i], _ = band_variance(ff, pp, period_lo_yr, period_hi_yr)
        pct = 100.0 * float(np.mean(vals < obs))
        out["nulls"][label] = {
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "percentile_of_obs": pct,
            "phi": (params[0] if params else None),
        }
    return out


# =====================================================================
#  PART 4 - Report
# =====================================================================

def report(resid, ens=None, fs=12.0,
           bands=(("decadal", 8.0, 20.0), ("annual", 0.8, 1.5)),
           NW=1.5, n_surr=500, seed=0):
    """
    Full diagnostic report.

    resid : (T,)   observed residual (output of preprocess)
    ens   : (M,T2) optional synthetic ensemble IN THE SAME SPACE as resid
                   (i.e. residuals: mean, trend and seasonal removed)
    """
    resid = np.asarray(resid, dtype=float).ravel()
    n = resid.size

    print("=" * 68)
    print("1. IS THE BAND RESOLVED BY WELCH?  (arithmetic, data-independent)")
    print("=" * 68)
    print(f"  record length = {n} months ({n/fs:.1f} yr)")
    for nps in (60, 120, 240, 360):
        if nps > n:
            continue
        df = fs / nps
        K = 1 + (n - nps) // (nps // 2)
        print(f"  nperseg={nps:4d} ({nps/fs:4.1f} yr): df={df:.4f} cyc/yr, "
              f"nu~{2*K:3d}")
        fw, pw = welch_grid(resid, fs, nps)
        for (lab, plo, phi) in bands:
            _, nb = band_variance(fw, pw, plo, phi)
            flag = "UNRESOLVED" if nb == 0 else f"{nb} bins"
            print(f"        {lab:8s} [{plo:4.1f},{phi:5.1f}] yr -> {flag}")
    print()

    print("=" * 68)
    print("2. HOW MUCH BAND VARIANCE IS REALLY THERE?")
    print("   full-record estimators, not tied to the Welch grid")
    print("=" * 68)
    bw = resolution_bandwidth(n, NW, fs)
    print(f"  multitaper NW={NW} -> resolution bandwidth {bw:.4f} cyc/yr")
    for (lab, plo, phi) in bands:
        width = 1.0 / plo - 1.0 / phi
        ok = "OK" if bw < width else "TOO WIDE - will under-read this band"
        print(f"     {lab:8s} band width {width:.4f} cyc/yr -> {ok}")
    print()
    print(f"  total residual variance = {resid.var():.6f}")
    print()
    f_mt, p_mt = multitaper_psd(resid, fs=fs, NW=NW)
    f_pg, p_pg = periodogram_psd(resid, fs=fs)
    f_w60, p_w60 = welch_grid(resid, fs, 60)
    print(f"  {'band':10s} {'multitaper':>12s} {'periodogram':>12s} "
          f"{'welch60':>12s} {'% of total':>11s}")
    for (lab, plo, phi) in bands:
        v_mt, _ = band_variance(f_mt, p_mt, plo, phi)
        v_pg, _ = band_variance(f_pg, p_pg, plo, phi)
        v_w, nbw = band_variance(f_w60, p_w60, plo, phi)
        wtxt = "0 (unres.)" if nbw == 0 else f"{v_w:.6f}"
        pct = 100.0 * v_mt / resid.var() if resid.var() > 0 else np.nan
        print(f"  {lab:10s} {v_mt:12.6f} {v_pg:12.6f} {wtxt:>12s} "
              f"{pct:10.1f} %")
    print()

    print("=" * 68)
    print(f"3. IS THAT MORE THAN NOISE WOULD GIVE?  "
          f"(Monte Carlo, {n_surr} surrogates)")
    print("=" * 68)
    for (lab, plo, phi) in bands:
        res = monte_carlo_band_test(resid, plo, phi, fs=fs, NW=NW,
                                    n_surr=n_surr, seed=seed)
        print(f"  {lab} band [{plo},{phi}] yr: observed = "
              f"{res['observed']:.6f}")
        for nlab, d in res["nulls"].items():
            extra = (f", phi={d['phi']:.3f}" if d["phi"] is not None else "")
            verdict_txt = ("SIGNIFICANT excess"
                           if d["percentile_of_obs"] >= 95
                           else "not distinguishable from null")
            print(f"      vs {nlab:12s}{extra}: null median "
                  f"{d['median']:.6f}, 95th {d['p95']:.6f} -> obs at "
                  f"{d['percentile_of_obs']:5.1f}th pctile  [{verdict_txt}]")
        print()

    print("=" * 68)
    print("4. WHAT DOES nperseg=60 KEEP?  (Parseval check)")
    print("=" * 68)
    for det in ("constant", False):
        f_w, p_w = welch(resid, fs=fs, nperseg=int(min(60, n)), detrend=det)
        rec = float(np.sum(p_w) * (f_w[1] - f_w[0]))
        pct = 100.0 * rec / resid.var() if resid.var() > 0 else np.nan
        print(f"  detrend={str(det):9s}: PSD accounts for {rec:.6f} of "
              f"{resid.var():.6f}  ({pct:5.1f} %)")
    print("  A large shortfall with detrend='constant' means per-segment mean")
    print("  removal is deleting low-frequency variance, not merely failing")
    print("  to resolve it.")
    print()

    if ens is not None:
        ens = np.atleast_2d(np.asarray(ens, dtype=float))
        print("=" * 68)
        print("5. DOES THE SYNTHETIC ENSEMBLE REPRODUCE IT?")
        print("=" * 68)
        print(f"  {'band':10s} {'observed':>12s} {'synthetic':>12s} "
              f"{'syn/obs':>9s}")
        for (lab, plo, phi) in bands:
            vo, _ = band_variance(*multitaper_psd(resid, fs=fs, NW=NW),
                                  plo, phi)
            vs = float(np.mean([
                band_variance(*multitaper_psd(r, fs=fs, NW=NW), plo, phi)[0]
                for r in ens
            ]))
            ratio = vs / vo if vo > 0 else np.nan
            print(f"  {lab:10s} {vo:12.6f} {vs:12.6f} {ratio:9.2f}")
        print()
        print("  A decadal ratio well below 1 is the smoking gun: the")
        print("  generator is not reproducing observed decadal variance.")
    print("=" * 68)


def verdict(resid, fs=12.0, band=(8.0, 20.0), NW=1.5, n_surr=500, seed=0):
    """Plain-language recommendation based on the numbers."""
    resid = np.asarray(resid, dtype=float).ravel()
    res = monte_carlo_band_test(resid, band[0], band[1], fs=fs, NW=NW,
                                n_surr=n_surr, seed=seed)
    frac = 100.0 * res["observed"] / resid.var() if resid.var() > 0 else 0.0
    sig = any(d["percentile_of_obs"] >= 95 for d in res["nulls"].values())

    print()
    print("#" * 68)
    print("VERDICT")
    print("#" * 68)
    print(f"  decadal band variance = {res['observed']:.6f} "
          f"({frac:.1f} % of residual variance)")
    print(f"  significant vs nulls  = {sig}")
    print()
    if sig and frac >= 5:
        print("  -> There IS substantial, statistically detectable decadal")
        print("     variance in the record, and nperseg=60 cannot see it.")
        print("     RECOMMEND: re-fit with an estimator that resolves the")
        print("     band (nperseg=240, or multitaper NW=1.5 on the full")
        print("     record), and report the decadal experiment as")
        print("     observationally constrained.")
    elif sig:
        print("  -> Decadal variance is detectable but small in absolute")
        print("     terms. Either estimator is defensible; state the choice")
        print("     explicitly and show the sensitivity test in an appendix.")
    else:
        print("  -> Decadal variance is NOT distinguishable from the null.")
        print("     nperseg=60 is defensible, BUT the decadal experiment must")
        print("     then be framed as a PRESCRIBED SENSITIVITY test: it")
        print("     imposes decadal variance the record cannot constrain,")
        print("     rather than perturbing an observed decadal spectrum.")
    print("#" * 68)


# =====================================================================
#  CONFIG - EDIT THIS
# =====================================================================
if __name__ == "__main__":

    # ---------------------------------------------------------------
    # 1. Point this at your raw RACMO NetCDF.
    RACMO_PATH = "./data/RACMO2.4p1_ANT11.nc"

    # 2. Optional basin masking. Leave as None if the file is already
    #    subset to the basin, or if geopandas/regionmask are unavailable.
    SHP_PATH = None        # e.g. "./data/IceBoundaries_Antarctica_V2.shp"
    BASIN_NAME = None      # e.g. "PineIsland"
    NAME_COL = "NAME"

    # 3. Optional SMB variable name. None = auto-detect.
    SMB_VAR = None

    # 4. Optional synthetic ensemble of RESIDUALS, shape (M, T).
    #    Leave as None to skip section 5.
    ENSEMBLE = None
    # ---------------------------------------------------------------

    import os

    if not os.path.exists(RACMO_PATH):
        print("=" * 68)
        print("RACMO file not found - running SELF-TEST instead.")
        print("Edit RACMO_PATH at the bottom of this file to use real data.")
        print("=" * 68)
        print()
        print("SELF-TEST: planted 12-yr cycle, true decadal variance 0.4972.")
        print("v1's block-means method reported 0.0090 here (a false")
        print("negative); v2 should recover ~0.50.")
        print()
        rng = np.random.default_rng(0)
        T = 540
        t = np.arange(T)
        demo = np.sin(2 * np.pi * t / 144.0) + 0.5 * rng.standard_normal(T)
        report(demo, n_surr=200)
        verdict(demo, n_surr=200)
    else:
        series, meta = load_smb_series(
            RACMO_PATH, smb_var=SMB_VAR, shp_path=SHP_PATH,
            basin_name=BASIN_NAME, name_col=NAME_COL,
        )
        resid, parts = preprocess(series)
        report(resid, ens=ENSEMBLE)
        verdict(resid)
