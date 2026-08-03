# Figure Guide — Synthetic SMB Manuscript

What each figure must show, where it goes, why it earns its place, and the
exact command to generate it.

Two code sources are used:
- **Built-in class methods** (confirmed by reading your uploaded 2-D code) —
  used for the spatial figures and EOF diagnostics.
- **`paper_figures.py`** (provided alongside this guide, smoke-tested) — for
  the scalar figures that have no built-in method. Every function takes plain
  arrays, so it does not depend on your exact class API.

---

## ⚠ Priority zero — read this before making any figures

While testing the band-variance diagnostic I found a real problem, not a
plotting bug:

**At `nperseg=60`, the decadal band is not resolved at all.**

```
Rayleigh frequency  Δf = fs / nperseg = 12 / 60 = 0.2 cycles yr⁻¹
  → longest independently resolved period = 5 yr

annual  band [0.8, 1.5] yr → [0.667, 1.250] cyc/yr →  3 resolved bins  ✓
decadal band [8, 20]  yr → [0.050, 0.125] cyc/yr →  0 resolved bins  ✗
```

The decadal band lies entirely *below* the lowest non-zero Welch frequency.
When the fitted PSD is interpolated onto the fine synthetic grid, the power in
that band comes from interpolation between the **zero-frequency term** and the
**0.2 cyc/yr bin** — it is not an estimate of decadal power from the data.

The decadal experiment still *does something* well-defined (it injects a
prescribed amount of variance at 8–20 yr periods), but it is a **sensitivity
experiment**, not a perturbation of an observationally constrained spectrum.
The manuscript previously said the band was "covered by one to two frequency
bins," which is wrong; I have corrected the Methods and Limitations text in
the current draft to state the situation accurately.

**Your options, in order of increasing effort:**

| Option | What it buys | What it costs |
|---|---|---|
| Keep `nperseg=60`, reframe decadal as prescribed sensitivity | nothing to redo; honest | decadal band not data-constrained |
| `nperseg=240` (20 yr) | resolves 8–20 yr band (~2 bins) | ν ≈ 6 dof → CI ratio ~10×; annual band coarser |
| Multitaper (e.g. `nitime`/`spectrum`, NW=4) | uses full 45 yr, controlled leakage, better low-f behaviour | change of estimator; must re-justify |
| Two-band hybrid: Welch for f > 0.2, multitaper/AR for f < 0.2 | best of both | more machinery to describe and defend |

**Make Figure S1 below first.** It is a two-panel diagnostic that settles this
question for you empirically, and it belongs in the appendix regardless of
which option you choose.

---

## Setup — run once, reuse everywhere

```bash
mkdir -p figs
```

```python
# ============ setup.py — fit once, cache, reuse for all figures ============
import numpy as np, xarray as xr, matplotlib
matplotlib.use("Agg")            # drop this line if you want interactive plots
import paper_figures as pf
pf.set_style()

RACMO = "./data/RACMO2.4p1_ANT11.nc"
SHP   = "./data/IceBoundaries_Antarctica_V2.shp"
BASIN = "PineIsland"

# ---- 2-D pipeline (API confirmed from your uploaded code) ----
from syn_smb.core.smb_field_loader    import SMBFieldLoader
from syn_smb.core.smb_field_generator import SMBFieldGenerator
from syn_smb.core.spatial_validator   import SpatialValidator
from syn_smb.core.experiment          import Experiment

loader = SMBFieldLoader(racmo_path=RACMO, shp_path=SHP, basin_name=BASIN)
field  = loader.load()                       # (time, rlat, rlon)

gen2d = SMBFieldGenerator(n_modes=10, nperseg=60)
gen2d.fit(field, lat=loader.lat, lon=loader.lon)

ds2d = gen2d.generate(Experiment.baseline())
val2d = SpatialValidator(gen2d, field)

# ---- 1-D pipeline ----
# NOTE: use YOUR scalar generator here. The figure functions below only need
# arrays, so however you build these is fine:
#   obs      : (T,)    observed basin-mean SMB, m w.e. a^-1
#   ens_base : (M, T2) baseline synthetic ensemble
#   resid, g_resid, freqs, psd, ci_lo, ci_hi  from the fitted 1-D objects
```

---

## Figure-by-figure

### Fig. 1 — Unified workflow schematic
**Where:** Introduction, after the contributions list (`\label{fig:pipeline}`).
**Shows:** the 1-D spine (load → preprocess → engine → reconstruct → validate)
with the EOF wrapper inserted, advancements A1–A4 tagged at their injection
points.
**Why:** orients the whole paper in one glance and makes the "same engine,
applied per mode" claim visual rather than verbal.
**Command:** already built — `smb_pipeline_unified.tex`. Compile with
pdfLaTeX, or paste the `tikzpicture` into a `figure*` float.
**Caption draft:** *Synthetic SMB generation workflow. The calibration pass
(blue) fits a spectral and distributional model to the observed record; the
synthesis pass (green) inverts it. For the spatial pipeline, EOF decomposition
and recomposition (violet) wrap the shared engine, which is applied once per
principal component. Advancements A1–A4 are marked at their injection points.*

---

### Fig. 2 — Observed record and its decomposition
**Where:** Data section (`\label{fig:data}`).
**Shows:** (a) raw monthly SMB + fitted linear trend; (b) 12-month seasonal
climatology; (c) stochastic residual.
**Why:** introduces the data *and* previews Eq. (1) — the reader sees exactly
what gets stripped before synthesis. Does double duty, so it is cheap.
```python
pf.fig02_decomposition(obs, trend_eval, seasonal_12, resid, "figs/fig02.pdf")
```
**Also available (2-D version):** `gen2d.preprocessor.plot_decomposition(field, save_path="figs/fig02_spatial.png")`
— four panels including the residual variance map. Use this *instead* if you
want the Data figure to be field-based rather than scalar.

---

### Fig. 3 — Gaussian transform and semi-parametric inverse (A1 + A2)
**Where:** Methods, after §"Semi-parametric inverse transform"
(`\label{fig:transform}`).
**Shows:** (a) residual histogram + fitted normal; (b) QQ plot vs N(0,1);
(c) empirical vs semi-parametric inverse CDF with splice points;
(d) **realised variance vs requested band factor** — the anti-saturation demo.
**Why:** panel (d) is the direct empirical evidence for A2. Everything else in
the paper argues that the parametric tail matters; this panel *shows* it. If
you only have time for one new analysis, do this one.
```python
# panel (d) requires a small experiment: generate at several band factors
# with the semi-parametric CDF and with a purely empirical CDF, then measure
# the realised variance ratio.
factors = [1, 2, 5, 10, 20]
sat_semi, sat_emp = [], []
for gam in factors:
    exp = Experiment(n_years=200, n_members=5, seed=0,
                     band_scales={"annual": gam})     # VERIFY your band_scales format
    ens_s = ...   # generate with semi-parametric inverse (default)
    ens_e = ...   # generate with empirical-only inverse (toggle/monkeypatch)
    sat_semi.append(np.var(ens_s) / var_baseline)
    sat_emp .append(np.var(ens_e) / var_baseline)

pf.fig03_transform(resid, g_resid, q_semi, q_emp,
                   factors, sat_semi, sat_emp, "figs/fig03.pdf")
```
> If toggling the empirical-only inverse is awkward in your code, the honest
> minimum is to plot the semi-parametric curve alone and state in the caption
> that the empirical-only ceiling is at the observed max — but the two-curve
> version is far more persuasive.

---

### Fig. 4 — Observed PSD with 95 % CI and named bands
**Where:** Methods, §"Spectral estimation" (`\label{fig:psd}`).
**Shows:** log–log fitted PSD, χ² confidence band (Eq. 5), annual and decadal
bands shaded.
**Why:** this *is* the fitted statistical model. Shading the bands makes the
A3 design legible early — and, given the priority-zero issue, makes the
decadal resolution problem visible instead of hidden.
```python
pf.fig04_psd_ci(freqs, psd, ci_lo, ci_hi,
                bands=(("annual", 0.8, 1.5), ("decadal", 8.0, 20.0)),
                path="figs/fig04.pdf")
```
**Add to the caption:** state the Rayleigh frequency (0.2 cyc yr⁻¹) explicitly
so the reader can see the decadal band sits below the lowest resolved bin.

---

### Fig. 5 — EOF variance, patterns, and PCs (A4)
**Where:** Methods §"EOF decomposition" (`\label{fig:scree}`).
**Shows:** (a) per-mode + cumulative explained variance with 80/95 %
thresholds and chosen *K*; (b) leading EOF spatial patterns; (c) leading PC
time series.
**Why:** justifies the truncation *K* — the first thing a reviewer will
challenge about the EOF step — and shows what the modes physically represent.
```python
gen2d.eof.plot_variance(save_path="figs/fig05a_scree.png")   # scree + cumulative
gen2d.eof.plot_eofs(n=4, save_path="figs/fig05b_patterns.png")
gen2d.eof.plot_pcs(n=4,  save_path="figs/fig05c_pcs.png")
gen2d.eof.summary()                     # prints the variance table → Table 3
print("K for 95%:", gen2d.eof.suggest_n_modes(0.95))
print("K for 80%:", gen2d.eof.suggest_n_modes(0.80))
```
> These are three separate files; combine into one multi-panel figure in
> LaTeX with `subfigure`, or accept three sub-figures. For a GMD paper I would
> combine (a) and (b) into the main text and push (c) to the appendix.

---

### Fig. 6 — Validation suite ★ key scalar figure
**Where:** Validation, §"Spectral coverage" (`\label{fig:valsuite}`).
**Shows:** (a) PSD observed + CI vs ensemble mean; (b) marginal distribution;
(c) ACF; (d) seasonal cycle.
**Why:** panel (a) is the visual form of the "coverage = 1.000" claim, and the
other three panels close off the obvious "yes but does it match the
distribution / memory / seasonality?" objections in one place.
```python
pf.fig06_validation_suite(obs, ens_base, freqs, psd, ci_lo, ci_hi,
                          fs=12.0, nperseg=60, nlags=36,
                          path="figs/fig06.pdf")
```

---

### Fig. 7 — Ensemble spaghetti + rolling variability
**Where:** Validation, §"Ensemble convergence and representativeness".
**Shows:** (a) synthetic members (annual means) with the observed 45-yr record
overlaid; (b) rolling 45-yr standard deviation across members (median + 16–84 %).
**Why:** makes the case for going beyond 45 years — you can *see* how much
variability a short record fails to sample.
```python
pf.fig07_spaghetti(obs, ens_base, window_years=45, n_show=8,
                   path="figs/fig07.pdf")
```

---

### Fig. 8 — Running-window distributions
**Where:** Validation, §"representativeness".
**Shows:** distributions of 45-yr mean, s.d., and range from the synthetic
ensemble, with the observed value and its percentile marked.
**Why:** reframes "does synthetic match observed?" into the stronger "is the
observed record a *typical realisation* of the fitted process?"
```python
pf.fig08_running_windows(obs, ens_base, window_years=45, path="figs/fig08.pdf")
```
**Harvest:** the three percentiles printed in the panel titles are quotable
numbers for the Validation text.

---

### Fig. 9 — Return-period plot
**Where:** Validation (or Discussion, if you want to end on motivation).
**Shows:** annual-mean SMB vs empirical return period, observed (limited to
~45 yr) vs synthetic (centuries).
**Why:** the single clearest one-panel motivation for synthetic generation.
```python
pf.fig09_return_period(obs, ens_base, path="figs/fig09.pdf")
```
> Note this uses Weibull plotting positions `i/(n+1)` for exceedance
> probability, deliberately *different* from the Hazen positions used in the
> quantile map (Eq. 2). Say so in the caption — a careful reviewer will notice.

---

### Fig. 10 — Variance map comparison ★ key spatial figure
**Where:** Validation §"Validation of the spatial field" (`\label{fig:varmaps}`).
**Shows:** observed variance | synthetic variance | ratio, in EPSG:3031.
**Why:** this is the figure that proves spatial coherence is reproduced, not
just basin-mean statistics. A ratio map near 1 everywhere is the claim; a
spatially structured ratio map would reveal where the EOF truncation hurts.
```python
val2d.plot_variance_maps(ds2d, save_path="figs/fig10_varmaps.png")
```
**Watch for:** systematically low ratios near the basin margins would indicate
the discarded high-order modes carry margin variance — worth a sentence in
Limitations if you see it.

---

### Fig. 11 — Per-PC PSD validation
**Where:** Validation §spatial (`\label{fig:pcpsd}`).
**Shows:** for the leading modes, Gaussianised observed PC spectrum vs the
fitted PSD with 95 % CI, both in N(0,1) space.
**Why:** ties the 2-D result back to the 1-D method — it shows the engine works
identically on each mode, which is the whole architecture of A4.
```python
val2d.plot_pc_validation(n_pcs=4, save_path="figs/fig11_pcpsd.png")
```

---

### Fig. 12 — Band-scaling demonstration (A3)
**Where:** §"Demonstration of frequency-band control" (`\label{fig:bands}`).
**Shows:** (a) ensemble-mean PSDs for baseline vs band-scaled runs with bands
shaded; (b) time-series excerpts showing the visual character of each;
(c) marginal distributions overlaid (unchanged).
**Why:** proves A3 does exactly what it claims — the band moves, everything
else does not. Panel (c) is the invariance evidence; do not drop it.
```python
suite = gen2d.generate_suite(Experiment.standard_suite())   # or your 1-D suite
series = {
    "baseline":     ens_baseline,
    "annual ×10":   ens_annual,
    "decadal ×10":  ens_decadal,
}
pf.fig12_band_demo(series, path="figs/fig12.pdf")

# realised vs requested scaling → Table 10
for name, ens in series.items():
    r_in, r_out = pf.band_variance_ratio(ens_baseline, ens, (8.0, 20.0))
    print(f"{name}: in-band {r_in:.2f}, out-of-band {r_out:.2f}")
```

---

### Fig. S1 — Spectral resolution diagnostic (appendix) ← **make this first**
**Where:** Appendix B (sensitivity analyses).
**Shows:** (a) fitted PSD for `nperseg ∈ {60, 120, 240}` overlaid, with the
annual and decadal bands shaded and each estimator's Rayleigh frequency
marked; (b) the resulting CI width (dof) for each.
**Why:** it is the empirical answer to the priority-zero problem, and it
pre-empts the reviewer question "why 60?" with data instead of assertion.
```python
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import welch
from scipy import stats

g = np.asarray(g_resid).ravel()
fig, ax = plt.subplots(1, 2, figsize=(pf.W_DOUBLE, 2.4))
for nps, c in zip([60, 120, 240], [pf.C_OBS, pf.C_SYN, pf.C_ALT]):
    f, p = welch(g, fs=12.0, nperseg=nps)
    K  = 1 + (g.size - nps) // (nps // 2)
    m  = f > 0
    ax[0].loglog(f[m], p[m], color=c, lw=1.1,
                 label=f"nperseg={nps} ({nps/12:.0f} yr), ν≈{2*K}")
    ax[0].axvline(12.0/nps, color=c, lw=0.6, ls=":")
    nu = 2*K
    ax[1].bar(str(nps), stats.chi2.ppf(0.975, nu)/stats.chi2.ppf(0.025, nu),
              color=c)
for pmin, pmax, lab in [(0.8,1.5,"annual"), (8.0,20.0,"decadal")]:
    ax[0].axvspan(1/pmax, 1/pmin, color=pf.C_GREY, alpha=0.15, lw=0)
ax[0].set_xlabel("frequency (cycles yr$^{-1}$)"); ax[0].set_ylabel("PSD")
ax[0].legend(frameon=False, fontsize=6)
ax[0].set_title("(a) PSD vs segment length; dotted = Rayleigh freq.", loc="left")
ax[1].set_ylabel("CI width ratio (upper/lower)")
ax[1].set_xlabel("nperseg")
ax[1].set_title("(b) Cost of resolution", loc="left")
fig.savefig("figs/figS1_resolution.pdf", dpi=300, bbox_inches="tight")
```

---

## Triage — what goes in the main text

Twelve figures is a lot even for a GMD model-description paper. My
recommendation:

**Main text (8):** Fig. 1 (workflow), 2 (data), 3 (transform + A2 evidence),
4 (PSD + bands), 5a–b (EOF scree + patterns), 6 (validation suite ★),
10 (variance maps ★), 12 (band demo).

**Appendix (4):** Fig. 5c (PC series), 7 (spaghetti), 8 (running windows),
9 (return period), 11 (per-PC PSD), S1 (resolution diagnostic).

Rationale: the main text should carry one figure per *claim* — the method
(1), the data (2), each advancement (3, 4+12, 5+10), and the headline
validation (6). Figures 7–9 are all variations on "long records show more
variability than short ones"; pick **one** for the main text (I would keep
Fig. 9, the return-period plot, since it is the most quantitative) and move
the rest to the appendix.

---

## Numbers to harvest while you make the figures

The draft has `[[...]]` placeholders that these same runs will fill. Grab them
as you go:

| Placeholder in draft | Where it comes from |
|---|---|
| `[[n_y × n_x]]`, `[[n_valid]]` | `loader.load()` printout; `loader.n_valid_cells` |
| `[[K]]`, `[[cumvar]]` | `gen2d.eof.summary()`, `suggest_n_modes(0.95)` |
| `[[mean_map_corr]]`, `[[var_map_corr]]`, `[[var_map_ratio]]`, `[[mean_ratio_2D]]`, `[[eof_cumvar]]` | `val2d.compute_metrics(ds2d)` |
| `[[split_2D]]` | `val2d.calibration_split_test(n_members=10, n_years_syn=50)` |

```python
m = val2d.compute_metrics(ds2d, verbose=True)     # prints all of the above
split = val2d.calibration_split_test(n_members=10, n_years_syn=50, verbose=True)
```

---

## Format and style notes

- **Sizes.** `paper_figures.py` uses `W_SINGLE = 3.35 in` (~8.5 cm) and
  `W_DOUBLE = 6.70 in` (~17 cm). *Verify against current Copernicus author
  guidelines* — I have not confirmed these against the 2026 spec.
- **Vector where possible.** Line plots → PDF. Maps and anything with many
  thousands of plotted elements (spaghetti, per-cell maps) → PNG at 300 dpi,
  or the PDF becomes enormous.
- **Colour.** The palette in `paper_figures.py` is Okabe–Ito (colour-blind
  safe). Check the variance-map ratio panel: a diverging map centred on 1 is
  right, but `RdBu_r` is not colour-blind ideal — consider `PuOr` if you want
  to be strict.
- **Consistency.** Call `pf.set_style()` once at the top of every script so
  fonts match across all figures, including the ones from class methods.
