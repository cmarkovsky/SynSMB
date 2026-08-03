"""
paper_figures.py
================
Publication-quality figure functions for the synthetic SMB manuscript.

Design principle: every function here takes **plain arrays / DataArrays**,
not pipeline objects. That keeps them independent of the exact class API,
so they work whatever your package internals look like, and makes them
easy to test.

Figures with a built-in method in the package (EOF scree/patterns, variance
maps, per-PC PSD, spatial decomposition) are NOT duplicated here - call the
class methods directly. See the accompanying figure guide.

Usage
-----
    import paper_figures as pf
    pf.set_style()
    pf.fig02_decomposition(smb, trend, seasonal, resid, "figs/fig02.pdf")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch

# ── Colour-blind-safe palette (Okabe-Ito) ────────────────────────────
C_OBS   = "#000000"   # observed: black
C_SYN   = "#0072B2"   # synthetic: blue
C_ENS   = "#56B4E9"   # ensemble members: light blue
C_CI    = "#E69F00"   # confidence interval: orange
C_ALT   = "#D55E00"   # emphasis / second experiment: vermillion
C_ALT2  = "#009E73"   # third experiment: green
C_GREY  = "#999999"

# Copernicus figure widths (inches). VERIFY against current author guidelines.
W_SINGLE = 3.35   # ~8.5 cm, single column
W_DOUBLE = 6.70   # ~17 cm, full width


def set_style(base_fontsize: int = 8) -> None:
    """Consistent, journal-appropriate matplotlib defaults."""
    plt.rcParams.update({
        "font.size":         base_fontsize,
        "axes.labelsize":    base_fontsize,
        "axes.titlesize":    base_fontsize,
        "xtick.labelsize":   base_fontsize - 1,
        "ytick.labelsize":   base_fontsize - 1,
        "legend.fontsize":   base_fontsize - 1,
        "axes.linewidth":    0.6,
        "grid.linewidth":    0.4,
        "lines.linewidth":   1.0,
        "axes.grid":         False,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "figure.constrained_layout.use": True,
    })


def _save(fig, path: str | None) -> None:
    if path:
        fig.savefig(path)
        print(f"  saved: {path}")


# =====================================================================
#  Figure 2 - Observed record and its decomposition
# =====================================================================
def fig02_decomposition(smb, trend, seasonal_12, resid, path=None):
    """
    Three stacked panels: (a) raw SMB + fitted trend, (b) seasonal
    climatology, (c) stochastic residual.

    Why this figure: it introduces the data AND previews Eq. (1), so the
    reader sees exactly which pieces the method strips off before it
    synthesises anything.

    Parameters
    ----------
    smb          : (T,) observed monthly SMB, m w.e. a^-1
    trend        : (T,) fitted linear trend evaluated on the same axis
    seasonal_12  : (12,) monthly climatological anomalies
    resid        : (T,) stochastic residual after trend + seasonal removal
    """
    smb   = np.asarray(smb).ravel()
    trend = np.asarray(trend).ravel()
    resid = np.asarray(resid).ravel()
    t     = np.arange(smb.size) / 12.0 + 1979

    fig, ax = plt.subplots(3, 1, figsize=(W_DOUBLE, 5.0))

    ax[0].plot(t, smb, color=C_OBS, lw=0.6)
    ax[1].axhline(0, color=0.5, lw=0.6)
    ax[0].plot(t, trend, color=C_ALT, lw=1.4, ls = '--', label="linear trend")
    ax[0].set_ylabel("SMB (m w.e. a$^{-1}$)")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].set_title("(a) Observed monthly SMB", loc="left")

    months = np.arange(1, 13)
    s12 = np.asarray(seasonal_12).ravel()
    ax[1].bar(months, s12, color=[C_SYN if v >= 0 else C_ALT for v in s12],
              alpha=0.85, width=0.7)
    ax[1].axhline(0, color=C_OBS, lw=0.6)
    ax[1].set_xticks(months)
    ax[1].set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax[1].set_ylabel("SMB anomaly (m w.e. a$^{-1}$)")
    ax[1].set_title("(b) Seasonal climatology", loc="left")

    ax[2].plot(t, resid, color=C_SYN, lw=0.6)
    ax[2].axhline(0, color=C_GREY, lw=0.6, ls="--")
    ax[2].set_ylabel("SMB residual (m w.e. a$^{-1}$)")
    ax[2].set_xlabel("Year")
    ax[2].set_title("(c) Stochastic residual $r(t)$ - target of synthesis",
                    loc="left")

    _save(fig, path)
    return fig


# =====================================================================
#  Figure 3 - Gaussian transform + semi-parametric inverse (A1 + A2)
# =====================================================================
def fig03_transform(resid, g_resid, q_semi, q_emp=None,
                    rp_semi=None, lv_semi=None, rp_emp=None, lv_emp=None, obs_max=None,
                    path=None):
    """
    Four panels proving A1 and A2:
      (a) residual histogram with fitted normal
      (b) transformed residuals vs N(0,1)  -> QQ plot (A1 works)
      (c) empirical vs semi-parametric inverse CDF, tails highlighted (A2)
      (d) anti-saturation demo: realised variance vs requested band factor

    Panel (d) is the money panel - it is the direct evidence that the
    parametric tail removes the variance ceiling. If you only have time
    for one new analysis, do that one.

    Parameters
    ----------
    resid       : (T,) residual in physical units
    g_resid     : (T,) Gaussianised residual
    q_semi      : callable u -> value, the semi-parametric inverse CDF
    q_emp       : callable u -> value, the purely empirical inverse CDF
                  (optional; used for the comparison in panel c)
    rp_semi     : (F,) return periods for the semi-parametric case
    lv_semi     : (F,) corresponding levels for the semi-parametric case
    rp_emp      : (F,) return periods for the empirical case
    lv_emp      : (F,) corresponding levels for the empirical case
    obs_max     : float, the maximum observed residual (for reference)
    """
    resid   = np.asarray(resid).ravel()
    g_resid = np.asarray(g_resid).ravel()

    fig, ax = plt.subplots(2, 2, figsize=(W_DOUBLE, 4.6))

    # (a) histogram + fitted normal
    ax[0, 0].hist(resid, bins=40, density=True, color=C_ENS,
                  edgecolor="white", linewidth=0.3)
    xs = np.linspace(resid.min(), resid.max(), 300)
    mu, sd = stats.norm.fit(resid)
    ax[0, 0].plot(xs, stats.norm.pdf(xs, mu, sd), color=C_ALT, lw=1.2,
                  label="fitted normal")
    ax[0, 0].set_xlabel("residual (m w.e. a$^{-1}$)")
    ax[0, 0].set_ylabel("density")
    ax[0, 0].legend(frameon=False)
    ax[0, 0].set_title("(a) Residual distribution", loc="left")

    # (b) QQ plot of transformed residuals vs standard normal
    n  = g_resid.size
    pp = (np.arange(1, n + 1) - 0.5) / n
    ax[0, 1].plot(stats.norm.ppf(pp), np.sort(g_resid), ".", ms=2,
                  color=C_SYN)
    lim = [min(stats.norm.ppf(pp).min(), g_resid.min()),
           max(stats.norm.ppf(pp).max(), g_resid.max())]
    ax[0, 1].plot(lim, lim, color=C_OBS, lw=0.8, ls="--")
    ax[0, 1].set_xlabel("theoretical N(0,1) quantile")
    ax[0, 1].set_ylabel("transformed quantile")
    ax[0, 1].set_title("(b) Gaussian rank transform (A1)", loc="left")

    # (c) inverse CDF: empirical vs semi-parametric
    u = np.linspace(1e-4, 1 - 1e-4, 500)
    ax[1, 0].plot(u, [q_semi(ui) for ui in u], color=C_SYN, lw=1.2,
                  label="semi-parametric")
    if q_emp is not None:
        ax[1, 0].plot(u, [q_emp(ui) for ui in u], color=C_ALT, lw=1.0,
                      ls="--", label="empirical only")
    p1, pn = 0.5 / resid.size, 1 - 0.5 / resid.size
    for pc in (p1, pn):
        ax[1, 0].axvline(pc, color=C_GREY, lw=0.6, ls=":")
    ax[1, 0].set_xlabel("cumulative probability $u$")
    ax[1, 0].set_ylabel("residual (m w.e. a$^{-1}$)")
    ax[1, 0].legend(frameon=False)
    ax[1, 0].set_title("(c) Inverse CDF, splice points dotted (A2)",
                       loc="left")

    # (d) anti-saturation demonstration
    if rp_semi is not None and lv_semi is not None:
        ax[1, 1].plot(rp_semi, lv_semi, "o-", ms=3, color=C_SYN,
                      label="semi-parametric")
    if rp_emp is not None and lv_emp is not None:
        ax[1, 1].plot(rp_emp, lv_emp, "s-", ms=3, color=C_ALT,
                      label="empirical only")
    if obs_max is not None:
        ax[1, 1].axhline(obs_max, color=C_OBS, lw=0.8, ls="--",
                         label="observed maximum")

    ax[1, 1].set_title("(d) Variance scaling, no saturation (A2)", loc="left")

    _save(fig, path)
    return fig


# =====================================================================
#  Figure 4 - Observed PSD with 95 % CI and named bands
# =====================================================================
def fig04_psd_ci(freqs, psd, ci_lo, ci_hi,
                 bands=(("annual", 0.8, 1.5), ("decadal", 8.0, 20.0)),
                 path=None):
    """
    Log-log PSD with chi-squared CI (Eq. 5) and shaded experiment bands.

    Why: this is the fitted statistical model. Shading the bands here makes
    the A3 experiment design legible before the reader reaches Sect. demo,
    and shows honestly how few bins cover the decadal band.

    Parameters
    ----------
    freqs   : (F,) frequency, cycles per year (exclude f=0 or it is dropped)
    psd     : (F,) fitted PSD
    ci_lo   : (F,) lower 95 % bound
    ci_hi   : (F,) upper 95 % bound
    bands   : sequence of (label, period_min_yr, period_max_yr)
    """
    freqs = np.asarray(freqs).ravel()
    psd   = np.asarray(psd).ravel()
    m     = freqs > 0

    fig, ax = plt.subplots(figsize=(W_SINGLE * 1.6, 2.8))
    ax.fill_between(freqs[m], np.asarray(ci_lo).ravel()[m],
                    np.asarray(ci_hi).ravel()[m],
                    color=C_CI, alpha=0.25, lw=0, label="95 % CI")
    ax.plot(freqs[m], psd[m], color=C_OBS, lw=1.2, label="observed PSD")

    colors = [C_SYN, C_ALT2, C_ALT]
    for i, (label, pmin, pmax) in enumerate(bands):
        ax.axvspan(1.0 / pmax, 1.0 / pmin, color=colors[i % len(colors)],
                   alpha=0.13, lw=0)
        ax.text(np.sqrt((1.0 / pmax) * (1.0 / pmin)), ax.get_ylim()[1],
                label, ha="center", va="bottom", fontsize=6,
                color=colors[i % len(colors)])

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("frequency (cycles yr$^{-1}$)")
    ax.set_ylabel("PSD (Gaussian space)")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, path)
    return fig


# =====================================================================
#  Figure 6 - Validation suite (the key scalar figure)
# =====================================================================
def fig06_validation_suite(obs, ens, freqs, psd_obs, ci_lo, ci_hi,
                           fs=12.0, nperseg=60, nlags=36, path=None):
    """
    Four panels: (a) PSD obs+CI vs ensemble mean, (b) marginal distribution,
    (c) ACF, (d) seasonal cycle.

    Panel (a) is the centrepiece of the paper - it is the visual form of the
    "coverage = 1.000" claim.

    Parameters
    ----------
    obs     : (T,) observed SMB series
    ens     : (M, T2) synthetic ensemble, physical units
    freqs, psd_obs, ci_lo, ci_hi : fitted spectrum + CI (Gaussian space or
              physical - just be consistent and say which in the caption)
    fs      : sampling frequency, samples per year (12 for monthly)
    """
    obs = np.asarray(obs).ravel()
    ens = np.atleast_2d(np.asarray(ens))

    fig, ax = plt.subplots(2, 2, figsize=(W_DOUBLE, 4.4))

    # (a) PSD comparison
    psds = []
    for row in ens:
        f_s, p_s = welch(row - row.mean(), fs=fs, nperseg=nperseg)
        psds.append(p_s)
    psd_syn = np.mean(psds, axis=0)
    mo = np.asarray(freqs).ravel() > 0
    ms = f_s > 0
    ax[0, 0].fill_between(np.asarray(freqs).ravel()[mo],
                          np.asarray(ci_lo).ravel()[mo],
                          np.asarray(ci_hi).ravel()[mo],
                          color=C_CI, alpha=0.25, lw=0, label="95 % CI")
    ax[0, 0].plot(np.asarray(freqs).ravel()[mo],
                  np.asarray(psd_obs).ravel()[mo],
                  color=C_OBS, lw=1.2, label="observed")
    ax[0, 0].plot(f_s[ms], psd_syn[ms], color=C_SYN, lw=1.2, ls="--",
                  label="ensemble mean")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("frequency (cycles yr$^{-1}$)")
    ax[0, 0].set_ylabel("PSD")
    ax[0, 0].legend(frameon=False, loc="lower left")
    ax[0, 0].set_title("(a) Power spectrum", loc="left")

    # (b) marginal distribution
    bins = np.linspace(min(obs.min(), ens.min()),
                       max(obs.max(), ens.max()), 45)
    ax[0, 1].hist(obs, bins=bins, density=True, histtype="step",
                  color=C_OBS, lw=1.2, label="observed")
    ax[0, 1].hist(ens.ravel(), bins=bins, density=True, histtype="stepfilled",
                  color=C_SYN, alpha=0.35, label="synthetic")
    ax[0, 1].set_xlabel("SMB (m w.e. a$^{-1}$)")
    ax[0, 1].set_ylabel("density")
    ax[0, 1].legend(frameon=False)
    ax[0, 1].set_title("(b) Marginal distribution", loc="left")

    # (c) autocorrelation
    def _acf(x, nl):
        x = x - x.mean()
        full = np.correlate(x, x, mode="full")[len(x) - 1:]
        return full[:nl + 1] / full[0]

    lags = np.arange(nlags + 1)
    ax[1, 0].plot(lags, _acf(obs, nlags), color=C_OBS, lw=1.2,
                  label="observed")
    acf_syn = np.mean([_acf(r, nlags) for r in ens], axis=0)
    ax[1, 0].plot(lags, acf_syn, color=C_SYN, lw=1.2, ls="--",
                  label="ensemble mean")
    ax[1, 0].axhline(0, color=C_GREY, lw=0.6)
    ax[1, 0].set_xlabel("lag (months)")
    ax[1, 0].set_ylabel("ACF")
    ax[1, 0].legend(frameon=False)
    ax[1, 0].set_title("(c) Autocorrelation", loc="left")

    # (d) seasonal cycle
    months = np.arange(12)
    obs_seas = np.array([obs[i::12].mean() for i in range(12)])
    syn_seas = np.array([ens[:, i::12].mean() for i in range(12)])
    ax[1, 1].plot(months + 1, obs_seas, "o-", ms=3, color=C_OBS,
                  label="observed")
    ax[1, 1].plot(months + 1, syn_seas, "s--", ms=3, color=C_SYN,
                  label="synthetic")
    ax[1, 1].set_xticks(months + 1)
    ax[1, 1].set_xticklabels(["J","F","M","A","M","J",
                              "J","A","S","O","N","D"])
    ax[1, 1].set_ylabel("SMB (m w.e. a$^{-1}$)")
    ax[1, 1].legend(frameon=False)
    ax[1, 1].set_title("(d) Seasonal cycle", loc="left")

    _save(fig, path)
    return fig


# =====================================================================
#  Figure 7 - Ensemble spaghetti + rolling variability
# =====================================================================
def fig07_spaghetti(obs, ens, window_years=45, n_show=8, path=None):
    """
    (a) A few synthetic members (annual means) with the observed record
    overlaid; (b) rolling `window_years` standard deviation across members.

    Why: this is the figure that makes the case for going beyond 45 years -
    it shows visually how much variability a short record cannot sample.
    """
    obs = np.asarray(obs).ravel()
    ens = np.atleast_2d(np.asarray(ens))

    def _annual(x):
        n = (x.size // 12) * 12
        return x[:n].reshape(-1, 12).mean(axis=1)

    obs_a = _annual(obs)
    ens_a = np.array([_annual(r) for r in ens])

    fig, ax = plt.subplots(2, 1, figsize=(W_DOUBLE, 3.8),
                           gridspec_kw={"height_ratios": [2, 1]})

    for r in ens_a[:n_show]:
        ax[0].plot(np.arange(r.size), r, color=C_ENS, lw=0.4, alpha=0.7)
    ax[0].plot(np.arange(obs_a.size), obs_a, color=C_OBS, lw=1.4,
               label=f"observed ({obs_a.size} yr)")
    ax[0].set_ylabel("annual SMB (m w.e. a$^{-1}$)")
    ax[0].legend(frameon=False, loc="upper right")
    ax[0].set_title("(a) Synthetic ensemble vs observed record", loc="left")

    w = window_years
    if ens_a.shape[1] >= w:
        roll = np.array([[r[i:i + w].std() for i in range(r.size - w + 1)]
                         for r in ens_a])
        q16, q50, q84 = np.percentile(roll, [16, 50, 84], axis=0)
        xs = np.arange(q50.size)
        ax[1].fill_between(xs, q16, q84, color=C_ENS, alpha=0.5,
                           lw=0, label="16-84 %")
        ax[1].plot(xs, q50, color=C_SYN, lw=1.0, label="median")
        ax[1].axhline(obs_a.std(), color=C_OBS, lw=1.2, ls="--",
                      label="observed")
        ax[1].set_ylabel(f"{w}-yr s.d.")
        ax[1].legend(frameon=False, ncol=3, loc="upper right")
    ax[1].set_xlabel("year of synthetic record")
    ax[1].set_title(f"(b) Rolling {w}-year standard deviation", loc="left")

    _save(fig, path)
    return fig


# =====================================================================
#  Figure 8 - Running-window distributions (representativeness)
# =====================================================================
def fig08_running_windows(obs, ens, window_years=45, path=None):
    """
    Distribution of statistics computed on all `window_years` windows drawn
    from the synthetic ensemble, with the observed value marked and its
    percentile annotated.

    Why: reframes "does synthetic match observed?" into the stronger
    "is the observed record a typical realisation of the fitted process?"
    """
    obs = np.asarray(obs).ravel()
    ens = np.atleast_2d(np.asarray(ens))

    def _annual(x):
        n = (x.size // 12) * 12
        return x[:n].reshape(-1, 12).mean(axis=1)

    obs_a = _annual(obs)
    w     = window_years

    means, sds, ranges = [], [], []
    for r in ens:
        ra = _annual(r)
        for i in range(0, ra.size - w + 1, max(1, w // 4)):
            seg = ra[i:i + w]
            means.append(seg.mean()); sds.append(seg.std())
            ranges.append(seg.max() - seg.min())

    obs_stats = [obs_a.mean(), obs_a.std(), obs_a.max() - obs_a.min()]
    dists     = [np.array(means), np.array(sds), np.array(ranges)]
    titles    = [f"(a) {w}-yr mean", f"(b) {w}-yr s.d.", f"(c) {w}-yr range"]

    fig, ax = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.0))
    for a, d, o, ti in zip(ax, dists, obs_stats, titles):
        a.hist(d, bins=35, color=C_ENS, edgecolor="white", linewidth=0.3)
        a.axvline(o, color=C_ALT, lw=1.4)
        pct = 100.0 * (d < o).mean()
        a.set_title(f"{ti}\nobserved = {pct:.0f}th pctile", loc="left")
        a.set_xlabel("m w.e. a$^{-1}$")
    ax[0].set_ylabel("count")
    _save(fig, path)
    return fig


# =====================================================================
#  Figure 9 - Return-period plot
# =====================================================================
def fig09_return_period(obs, ens, path=None):
    """
    Empirical return periods of annual-mean SMB anomalies, observed vs
    synthetic, on a log return-period axis.

    Why: the single clearest motivation for synthetic generation - the
    observed curve stops at ~n years, the synthetic extends to centuries.

    Uses Weibull plotting positions i/(n+1), which are unbiased for
    exceedance probability (Hazen positions are used elsewhere for the
    quantile map; the distinction is deliberate).
    """
    obs = np.asarray(obs).ravel()
    ens = np.atleast_2d(np.asarray(ens))

    def _annual(x):
        n = (x.size // 12) * 12
        return x[:n].reshape(-1, 12).mean(axis=1)

    def _rp(series):
        a = np.sort(_annual(series))[::-1]     # descending
        n = a.size
        rank = np.arange(1, n + 1)
        return n / rank, a                      # return period, magnitude

    fig, ax = plt.subplots(figsize=(W_SINGLE * 1.6, 2.6))

    for r in ens[:20]:
        rp, mag = _rp(r)
        ax.plot(rp, mag, color=C_ENS, lw=0.4, alpha=0.5)
    rp_o, mag_o = _rp(obs)
    ax.plot(rp_o, mag_o, "o-", ms=3, color=C_OBS, lw=1.2, label="observed")
    ax.plot([], [], color=C_ENS, lw=1.0, label="synthetic members")

    ax.axvline(rp_o.max(), color=C_GREY, lw=0.8, ls=":")
    ax.text(rp_o.max(), ax.get_ylim()[1], " observed limit",
            fontsize=6, va="top", color=C_GREY)
    ax.set_xscale("log")
    ax.set_xlabel("return period (yr)")
    ax.set_ylabel("annual-mean SMB (m w.e. a$^{-1}$)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)
    return fig


# =====================================================================
#  Figure 12 - Band-scaling demonstration (A3)
# =====================================================================
def fig12_band_demo(series_by_exp, bands=(("annual", 0.8, 1.5),
                                          ("decadal", 8.0, 20.0)),
                    fs=12.0, nperseg=60, excerpt_years=60, path=None):
    """
    (a) Overlaid ensemble-mean PSDs for baseline vs band-scaled experiments
    with the scaled bands shaded; (b) time-series excerpts showing the
    visual character of each; (c) marginal distributions overlaid to show
    they are unchanged.

    Why: this is the figure that proves A3 does what it claims - the band
    moves, everything else does not.

    Parameters
    ----------
    series_by_exp : dict {experiment_name: (M, T) ensemble array}
                    e.g. {"baseline": ..., "annual x10": ...,
                          "decadal x10": ...}
    """
    colors = [C_OBS, C_SYN, C_ALT, C_ALT2]
    fig, ax = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.2))

    # (a) PSDs
    for i, (name, ens) in enumerate(series_by_exp.items()):
        ens = np.atleast_2d(np.asarray(ens))
        psds = []
        for r in ens:
            f, p = welch(r - r.mean(), fs=fs, nperseg=nperseg)
            psds.append(p)
        p_mean = np.mean(psds, axis=0)
        m = f > 0
        ax[0].plot(f[m], p_mean[m], color=colors[i % len(colors)], lw=1.1,
                   label=name)
    for (label, pmin, pmax) in bands:
        ax[0].axvspan(1.0 / pmax, 1.0 / pmin, color=C_GREY, alpha=0.15, lw=0)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("frequency (cycles yr$^{-1}$)")
    ax[0].set_ylabel("PSD")
    ax[0].legend(frameon=False, fontsize=6)
    ax[0].set_title("(a) Spectra", loc="left")

    # (b) excerpts
    npts = int(excerpt_years * 12)
    for i, (name, ens) in enumerate(series_by_exp.items()):
        ens = np.atleast_2d(np.asarray(ens))
        seg = ens[0, :npts]
        ax[1].plot(np.arange(seg.size) / 12.0, seg + i * 0.0,
                   color=colors[i % len(colors)], lw=0.5, alpha=0.85)
    ax[1].set_xlabel("year")
    ax[1].set_ylabel("SMB (m w.e. a$^{-1}$)")
    ax[1].set_title(f"(b) {excerpt_years}-yr excerpts", loc="left")

    # (c) marginals
    allv = np.concatenate([np.asarray(e).ravel()
                           for e in series_by_exp.values()])
    bins = np.linspace(allv.min(), allv.max(), 45)
    for i, (name, ens) in enumerate(series_by_exp.items()):
        ax[2].hist(np.asarray(ens).ravel(), bins=bins, density=True,
                   histtype="step", color=colors[i % len(colors)], lw=1.0)
    ax[2].set_xlabel("SMB (m w.e. a$^{-1}$)")
    ax[2].set_ylabel("density")
    ax[2].set_title("(c) Marginals (unchanged)", loc="left")

    _save(fig, path)
    return fig


# =====================================================================
#  Helper: band variance diagnostic (fills Table 10)
# =====================================================================
def band_variance_ratio(ens_base, ens_scaled, band_yr, fs=12.0, nperseg=60):
    """
    Realised variance ratio inside a period band, scaled vs baseline.

    Returns (ratio_in_band, ratio_out_of_band). Use this to fill the
    "realised vs requested" table and to support the claim that physical
    band scaling is only approximately gamma_b.
    """
    def _band_var(ens, inside=True):
        ens = np.atleast_2d(np.asarray(ens))
        tot = []
        for r in ens:
            f, p = welch(r - r.mean(), fs=fs, nperseg=nperseg)
            lo, hi = 1.0 / band_yr[1], 1.0 / band_yr[0]
            m = (f >= lo) & (f <= hi)
            if not inside:
                m = ~m & (f > 0)
            df = f[1] - f[0]
            tot.append(np.sum(p[m]) * df)
        return np.mean(tot)

    return (_band_var(ens_scaled, True) / _band_var(ens_base, True),
            _band_var(ens_scaled, False) / _band_var(ens_base, False))