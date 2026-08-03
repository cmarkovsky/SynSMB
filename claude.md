# Synthetic SMB generation — project memory

Methods paper: a phase-randomisation generator for synthetic surface mass
balance (SMB) forcing with controllable frequency structure, calibrated on
RACMO2.4p1 for the Pine Island Glacier (PIG) basin. Target: Geoscientific
Model Development (Copernicus).

## Current scope decision

**1-D (basin-integrated scalar) only.** The EOF/spatial extension is
implemented but is deferred to a second paper. Do not add 2-D material to the
manuscript unless explicitly asked. The 2-D code stays in the repo and stays
working; it is just out of scope for the current draft.

Claimed advancements in the 1-D paper:
- **A1** distribution-preserving synthesis (Gaussian rank transform)
- **A2** semi-parametric tail extrapolation (removes variance saturation)
- **A3** frequency-band variance control + the redistribution mathematics

## Established empirical findings — do not re-derive, do not contradict

Measured on the real PIG record (1979–2023, n=540 monthly, m w.e. a⁻¹):

| quantity | value |
|---|---|
| raw mean / std | 0.03586 / 0.04629 |
| residual variance (after mean, trend, seasonal) | 0.001742 |
| deterministic seasonal variance | 0.000384 (17.9 % of raw) |
| linear trend variance | 0.000016 (0.8 % of raw) |
| lag-1 autocorrelation φ | 0.08 (e-folding 0.4 months) |

**The PIG residual is statistically indistinguishable from white noise.**
Full-record multitaper (NW=1.5): decadal band (8–20 yr) holds 1.7 % of
residual variance against a white-noise expectation of 1.25 %; annual band
(0.8–1.5 yr) holds 11.5 % against 9.7 %. Neither is significant vs white or
AR(1) nulls (Monte Carlo, 500 surrogates; 84th and 78th percentiles).

Consequences that must be reflected in any text:
- Band experiments are **prescribed sensitivity tests**, not perturbations of
  an observed low-frequency spectrum. Never write that the decadal band
  "covers variability visible in the PIG spectrum" — there isn't any.
- This is the Hasselmann (1976) regime: near-white forcing into a slow
  integrator. Frame it as a result, not a disappointment.

**`nperseg=60` is justified empirically**, not by assertion: the estimate
accounts for 99.7 % of residual variance (Parseval check). The Rayleigh
frequency is 0.2 cycles yr⁻¹, so the decadal band contains **zero** resolved
bins — but there is nothing there to resolve. The record, not the estimator,
is the binding constraint.

## Method invariants and pitfalls

These are the non-obvious correctness constraints. Violating them silently
produces plausible-looking but wrong output.

1. **The mean is removed once and added back exactly once**, in the
   generator. Never add it in the preprocessor's `inverse_transform`.
2. **The linear trend is removed and never restored.** Extrapolating a 45-yr
   trend over 1000 yr is indefensible. Synthetic forcing is stationary in
   mean by design.
3. **Band scaling redistributes variance, it does not add it.** Per-member
   unit-variance normalisation means total variance is conserved, so
   out-of-band variance *must* fall when in-band rises. With `s_b` the
   baseline band share:
   - in-band ratio  = γ / (1 + (γ−1)·s_b)
   - out-of-band ratio = 1 / (1 + (γ−1)·s_b)
   Do not write "variance outside the scaled band is preserved" — it is
   mathematically impossible under the current normalisation.
4. **The deterministic seasonal cycle is added back UNSCALED.** It carries
   66 % of baseline annual-timescale variance, so the annual experiment
   saturates at a **hard ceiling of 3.64×** no matter how large γ is
   (γ=10 → 2.34×). This is a known open bug in the experiment design; see
   "Work order" below.
5. **Removing the monthly climatology notches the spectrum** at exactly
   1, 2, …, 6 cycles yr⁻¹. The annual band (bins 0.8, 1.0, 1.2) contains the
   1.0 notch.
6. Baseline band shares for experiment design: `s_annual = 0.115`,
   `s_decadal = 0.017`. Same γ buys very different perturbations — γ=10 gives
   the decadal band 14 % of variance but the annual band 57 %. To reach 50 %
   share needs γ≈59 (decadal) vs γ≈8 (annual). **Match experiments on
   realised variance partition, never on nominal γ.**

## Validation: identities vs genuine tests

Four of the seven metrics are true by construction and must be labelled as
correctness checks, not validation results:

- **identities**: mean ratio (1.0000), total variance (≈0.996), KS/marginal,
  seasonal RMSE. Marginal preservation under band scaling is a mathematical
  necessity (unit-variance Gaussian through a fixed monotone map), not a
  finding.
- **genuine**: PSD coverage, and the calibration-split test.
- ACF is largely implied by the PSD via Wiener–Khinchin — semi-redundant.

## Work order (current)

1. **Fix seasonal scaling.** Scale the stored 12-value climatology directly
   (by √γ) before adding it back, so the annual experiment has no ceiling and
   is comparable to the decadal one. Report seasonal-amplitude factor and
   residual band factor separately.
2. **Multi-basin survey.** Run `decadal_diagnostic.py` across ~8–10 RACMO
   basins (candidates: Dronning Maud Land, a Peninsula basin, Totten, Amery,
   Thwaites). This decides the paper's framing and must precede figure work.
3. Re-run the experiment suite; report delivered, not nominal, factors.
4. Reframe the validation section per the identity/test split above.
5. Then generate figures (see `docs/figure_guide.md`).

Known smaller bug: the convergence diagnostic computes `psd_rms` from member
0 only, so it is flat across ensemble size N and uninformative. Fix before
producing the convergence table.

## Writing conventions

- Copernicus (GMD) style; author–year citations, e.g. "Hugonnet et al., 2021".
- SI units throughout. Explicit uncertainty reporting.
- Prose over bullets in anything manuscript-bound. Sober, structured tone.
- Preserve the author's voice when editing. Flag overclaiming; mark any
  sentence where the cited evidence does not fully support the claim.
- Consistent English variant per document (currently British).
- Do not invent citations. `references.bib` marks unverified entries with
  `VERIFY` in a note field — `aloni2025` could **not** be verified and must
  not be submitted as-is. Verified: van Dalum et al. (2025), *The Cryosphere*
  19, 4061–4090, doi:10.5194/tc-19-4061-2025 (RACMO2.4p1);
  Muruganandham et al. (2023) — closest methodological precedent, EOF/PC
  resynthesis for ocean forcing, flags band perturbation as future work.

## Code conventions

- Python; numpy / scipy / xarray / matplotlib. xarray-idiomatic and lazy for
  anything large; flag memory hazards.
- Explicit CRS handling, no silent unit conversions, seeds set for anything
  stochastic.
- Comment the *why* of non-obvious numerical choices, not the what.
- Figure functions take plain arrays, not pipeline objects, so they stay
  decoupled from class APIs (see `paper_figures.py`).
- When results surprise you, check the pipeline (units, detrending, temporal
  alignment, frequency grids) before theorising about glaciology.

## Reference documents in this repo

Read on demand rather than loading every session:
- `docs/manuscript_draft.tex` — current draft (Copernicus body)
- `docs/figure_guide.md` — what each figure shows, where it goes, commands
- `docs/workflow_guide.md` — step-by-step method walkthrough with self-test
- `decadal_diagnostic.py` — spectral-resolution / whiteness diagnostic
- `paper_figures.py` — publication figure functions

## Environment

This project uses Poetry. Its virtualenv is the only correct interpreter.

- Run Python as `poetry run python ...`, tests as `poetry run pytest`.
- NEVER use bare `python`, `python3`, or `pip` — they may resolve to the
  system interpreter and silently run against the wrong packages.
- Add dependencies with `poetry add <pkg>`, never `pip install`.
- If an import fails, check `poetry run python -c "import sys; print(sys.executable)"`
  before assuming the package is missing.