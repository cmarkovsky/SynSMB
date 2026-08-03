# Synthetic SMB Generation — Step-by-Step Walkthrough & Self-Test

A guided tour of the whole pipeline, one stage at a time. For each stage:
**what goes in**, **what it does** (and the equation behind it), **what comes
out**, and **why it matters**. At the end of each stage there are one or two
**check-your-understanding** questions. Answers are collected at the very
bottom — try to answer before looking.

The pipeline has two passes that mirror each other:

```
FIT (learn from 45 yr of RACMO)          GENERATE (produce 1000 yr synthetic)
  load → preprocess → gaussianize          band-scale → phase-randomize
       → estimate PSD                       → inverse-gaussianize → reconstruct
```

Equation numbers below refer to the manuscript draft.

---

## Stage 0 — Load the data

**In:** RACMO2.4p1 NetCDF.
**Does:** For the *scalar* pipeline, integrate SMB over the Pine Island basin
to one monthly series (`smbgl`, 540 values, 1979–2023). For the *field*
pipeline (`SMBFieldLoader`), keep the full grid, mask to the basin using the
MEaSUREs boundary shapefile via `regionmask`, convert kg m⁻² → m w.e., and
crop to the basin bounding box.
**Out:** either a `(time,)` scalar series or a `(time, rlat, rlon)` field with
NaN outside the basin.
**Why it matters:** the scalar is the basin-integrated forcing; the field
preserves *where* anomalies happen, which is what the EOF step later exploits.

> **Q0.1** The loader converts kg m⁻² to m w.e. by dividing by 1000. Why is
> getting units right *here*, at load time, more than cosmetic for everything
> downstream?
>
> **Q0.2** Cropping to the bounding box shrinks the PIG grid from ~430k cells
> to ~1.2k. The discarded cells are all NaN (outside the basin). Does dropping
> them change the EOF result at all? Why or why not?

---

## Stage 1 — Preprocess: remove trend and seasonal cycle

**In:** the raw series/field.
**Does:** Split the signal per **Eq. (1)**:
`x(t) = μ + τ(t) + s(m(t)) + r(t)`.
Estimate and remove the record mean `μ`, a degree-1 (linear) trend `τ(t)` by
least squares, and a 12-value monthly climatology `s(m)` (computed *on the
detrended* series). In the field version (`SpatialPreprocessor`) this happens
independently at every grid cell.
**Out:** the stochastic residual `r(t)` — mean ≈ 0, no trend, no seasonal cycle.
**Why it matters:** the two next stages (Gaussianize, spectral fit) *assume*
a stationary, zero-mean input. Everything deterministic must be stripped so
that only the part we actually want to synthesize — the stochastic
variability — remains.

Two deliberate asymmetries to notice:
- `μ` is removed once and added back **once**, in the generator (never twice).
- `τ(t)` is removed but **never added back** to synthetic series — extrapolating
  a 45-yr linear trend over 1000 yr is physically indefensible.

> **Q1.1** Why is the seasonal climatology computed *after* detrending rather
> than on the raw series? What bias creeps in if you don't?
>
> **Q1.2** A colleague suggests a cubic (degree-3) trend "to fit the data
> better." In the context of the *decadal-band experiment*, why is that a bad
> idea? (Hint: what is a degree-3 polynomial capable of absorbing?)
>
> **Q1.3 (field)** `SpatialPreprocessor` removes the seasonal cycle *per grid
> cell before* the EOF step. What would the leading EOF end up representing if
> you skipped seasonal removal?

---

## Stage 2 — Gaussianize: rank transform to N(0,1)

**In:** the residual `r(t)`.
**Does:** Rank the values, assign Hazen plotting positions, map each to the
matching normal quantile — **Eq. (2)**:
`p_i = (i − ½)/n`, `g_(i) = Φ⁻¹(p_i)`.
**Out:** a series `g(t)` that is *exactly* standard-normal in its marginal
distribution (KS statistic D = 0.0009), but has the **same rank ordering** as
`r(t)` — so its autocorrelation/spectral structure is essentially preserved.
**Why it matters:** the phase-randomization engine (Stage 4) assumes Gaussian
input. SMB is right-skewed and bounded below by zero, so feeding it in raw
would distort the synthetic distribution. This transform **decouples** the
distribution (handled here and in Stage 5) from the spectrum (handled in
Stages 3–4). This is advancement **A1**.

> **Q2.1** The transform is monotone (rank-preserving). Why does that property
> mean it barely changes the *shape* of the power spectrum, even though it
> changes every individual value?
>
> **Q2.2** Hazen positions use `(i − ½)/n` rather than `i/n`. What goes wrong
> at the largest value `i = n` if you used `i/n` and then applied `Φ⁻¹`?

---

## Stage 3 — Estimate the power spectrum (PSD)

**In:** the Gaussianized residual `g(t)`.
**Does:** Welch's method — split into 60-month (5-yr) segments with 50%
overlap, Hann-window each, average the periodograms → a smoothed PSD at 31
frequencies. Attach a χ² confidence interval — **Eq. (5)** — with
ν = 2K = 34 degrees of freedom (K = 17 segments).
**Out:** the fitted PSD `P̂(f)` and its 95% CI. This *is* the statistical
model of the variability: it says how much variance sits at each timescale.
**Why it matters:** the PSD is the target the synthetic series must reproduce.
The CI is the honest uncertainty — with only 45 years, the decadal band is
covered by just 1–2 frequency bins, which is the single biggest caveat in the
paper.

> **Q3.1** Welch trades frequency resolution for lower variance in the
> estimate. In one sentence, what is the specific cost of choosing a *longer*
> segment (say 120 months instead of 60) for the annual vs the decadal band?
>
> **Q3.2** The CI in Eq. (5) assumes the K segments are independent, but 50%
> overlap makes neighbours correlated. Does this make the true CI *wider* or
> *narrower* than reported — i.e., is the reported uncertainty optimistic or
> conservative?

---

## Stage 4 — Synthesize: band-scale + phase-randomize

**In:** the fitted PSD `P̂(f)`.
**Does:** three sub-steps.
1. **Interpolate** `P̂` onto the long synthetic frequency grid (6001 points
   for 1000 yr vs 31 fitted).
2. **Band-scale** — **Eq. (6)**: multiply the PSD by `γ_b` inside a chosen
   period band (e.g. ×10 in [8,20] yr), leave the rest untouched. This is
   advancement **A3** — the controlled-experiment knob.
3. **Phase-randomize** — build amplitudes `A_k = √(P̃(f_k)·Δf·N)` (**Eq. 7**,
   Parseval-consistent), give each ensemble member independent random phases
   `φ_k ~ U(−π,π)`, and inverse-FFT to a time series (**Eq. 8**). Normalize
   each member to unit variance.
**Out:** an ensemble of synthetic Gaussian series `g̃(t)`, each 1000 yr long,
all sharing the same spectrum but statistically independent.
**Why it matters:** keeping amplitudes fixed and randomizing only phases is
what guarantees the ensemble-mean PSD converges to the target while the
members are independent realizations — the core of the whole method.

> **Q4.1** Every member shares the *same* amplitude spectrum `A_k` and differs
> only in phases. Explain why this makes (a) the ensemble-mean PSD converge to
> the target, but (b) *understate* the true realization-to-realization spread
> you'd see in nature. (This is a stated limitation.)
>
> **Q4.2** Band-scaling multiplies *Gaussian-space* PSD by `γ_b`, so the
> Gaussian-space band variance scales by exactly `γ_b`. Why is the scaling of
> the *physical* (post-inverse-transform) band variance only *approximately*
> `γ_b`? (Hint: what does Eq. (3) do to a rescaled Gaussian?)
>
> **Q4.3** Why must the DC (zero-frequency) and Nyquist phases be fixed at 0
> rather than randomized?

---

## Stage 5 — Inverse transform + reconstruct

**In:** synthetic Gaussian series `g̃(t)`.
**Does:** reverse Stages 2 and 1.
- Inverse Gaussianize — **Eq. (3)**: map `g̃` back through the
  semi-parametric inverse CDF (empirical interpolation inside the observed
  range, fitted-normal tails outside, spliced continuously via **Eq. 4**).
  This is advancement **A2**. Subtract any residual mean so the series is
  exactly zero-mean.
- Add back `μ` (once), add back the seasonal cycle `s(m)` (never the trend).
**Out:** physical synthetic SMB, `(member, time)` — ready to force an ice model.
**Why it matters:** A2's spliced tail is what lets amplified extremes exceed
the observed range *continuously* instead of piling up at the observed
min/max. Without it, a ×10 band experiment would have its variance **saturate**
against a hard ceiling — making the controlled experiment uninterpretable.

> **Q5.1** Describe the "variance saturation" failure mode a *purely
> empirical* inverse CDF (no parametric tail) would cause under ×10 band
> amplification. Why does it break the experiment's interpretation?
>
> **Q5.2** Passing a symmetric Gaussian through the *asymmetric* map of
> Eq. (3) produces a small non-zero mean. Why is enforcing zero mean *inside*
> the inverse transform (rather than trusting `μ` to handle it) essential for
> keeping band experiments "variability-only"?

---

## Stage 6 (field pipeline) — EOF decomposition & recombination

This wraps the 1-D engine to make **spatially coherent** fields — advancement
**A4**. The 1-D Stages 2–5 are reused *unchanged*, once per mode.

**In:** the residual *field* `r(x,y,t)` from the spatial preprocessor.
**Does:**
1. **Decompose** — **Eq. (9)**: flatten to a (time × valid-cells) matrix,
   area-weight each cell by `√cos φ`, take an SVD. This yields fixed spatial
   patterns `E_k(x,y)` (the EOFs) and their time series `PC_k(t)` (the
   principal components), plus the variance each mode explains.
2. **Truncate** at K modes (smallest K reaching, say, 95% cumulative variance).
3. **Synthesize each PC** independently through Stages 2–5 (Gaussianize →
   PSD → band-scale + phase-randomize → inverse). Each (member, mode) gets an
   independent RNG stream.
4. **Recombine** — **Eq. (10)**: `r̃(x,y,t) = Σ_k P̃C_k(t)·E_k(x,y)/√cos φ`,
   then restore per-cell seasonal cycle and mean.
**Out:** a synthetic SMB *field* ensemble, `(member, time, rlat, rlon)`.
**Why it matters:** the spatial patterns `E_k` are held **fixed** — only the
temporal PCs are resynthesized. That is *why* spatial coherence is preserved
by construction: every synthetic field is a combination of the real observed
spatial patterns, just with new, statistically-consistent time evolution.

> **Q6.1** Spatial coherence is "preserved by construction." State the
> one-sentence reason, referencing what is held fixed vs. what is resynthesized.
>
> **Q6.2** The basin-integrated *scalar* pipeline is described as the
> "single-mode limit" of the field pipeline. In what sense is integrating over
> the basin like keeping only one spatial mode?
>
> **Q6.3** Each PC is synthesized *independently*. What kind of real
> spatiotemporal behaviour (think of a feature that moves across the basin over
> time) can this construction **not** reproduce, and why? (This is a genuine
> limitation.)
>
> **Q6.4** Why area-weight by `√cos φ` before the SVD rather than weighting the
> variance by `cos φ` after? (Hint: SVD minimizes squared error on the matrix
> it's *given*.)

---

## Stage 7 — Validate

**Scalar (`Validator`):** in-sample metrics (mean ratio = 1.0000, variance
ratio ≈ 0.996, seasonal RMSE ~2×10⁻⁴), spectral **coverage** (ensemble-mean
PSD inside the 95% CI at all 31 frequencies), and the out-of-sample
**calibration-split** test (fit on one half, score the other).
**Field (`SpatialValidator`):** mean-map and variance-map correlations, per-PC
PSD checks, and a spatial calibration-split.

**Why it matters:** in-sample fidelity is *necessary but circular* (you fit and
score on the same data). The calibration-split is the real defence: it shows
the method generalizes to data it never saw.

> **Q7.1** In the calibration-split the mean ratios came out 1.19 and 0.84 (not
> 1.0). Why is that a *success*, not a failure? What real property of the
> 1979–2023 record does the asymmetry reveal?
>
> **Q7.2** "Spectral coverage = 1.000" means the ensemble-mean PSD lands inside
> the observed 95% CI at every frequency. Why is this a weaker claim than it
> first sounds — i.e., why does a *wide* CI (from the short record) make
> coverage easier to achieve? What's the honest way to report this?

---

## The big picture — one integrating question

> **QX** Trace a single number — the observed SMB anomaly in, say, March 1993 —
> conceptually through FIT and out through GENERATE. At which stages does its
> *specific value* still matter, and at which stage does it stop mattering as an
> individual value and start mattering only as part of a *distribution* or
> *spectrum*? (This tests whether you've internalized what the method keeps vs.
> discards.)

---

# Answers

**Q0.1** Units set the scale of *everything* downstream: the fitted PSD,
amplitudes (Eq. 7), band variances, and the reconstructed forcing all inherit
them. A silent kg m⁻² vs m w.e. error (factor 1000) wouldn't error out — it
would produce plausible-looking but physically wrong forcing, and the bug
would only surface when the ice model responds absurdly.

**Q0.2** No change. The discarded cells are all-NaN outside the basin and are
never part of the valid-cell matrix `X` that the SVD operates on. Cropping is
purely a memory/speed optimization.

**Q1.1** A leftover trend inflates the months that happen to sit at the start
vs. end of the record, biasing the 12 monthly means. Detrending first removes
that tilt so the climatology reflects the true seasonal shape, not the trend.

**Q1.2** A cubic can bend to follow multi-decadal swings — exactly the
low-frequency variability the decadal-band experiment is meant to *inject and
study*. Removing it as "trend" would strip the decadal signal out of the
residual, confounding the experiment. Linear is the conservative floor.

**Q1.3** The leading EOF would capture the **spatial pattern of the seasonal
cycle** (a huge, coherent, deterministic signal) instead of the stochastic
variability you actually want to synthesize — wasting your top mode(s) on
something deterministic.

**Q2.1** Spectrum/autocorrelation depend on the *ordering* of values through
time, not their exact magnitudes. A monotone map relabels magnitudes but keeps
every rank in place, so lag relationships — and thus the spectral shape — are
nearly unchanged.

**Q2.2** `i/n` gives `p_n = 1` at the largest value, and `Φ⁻¹(1) = +∞`. Hazen's
`(i−½)/n` keeps every position strictly inside (0,1), so all normal quantiles
are finite.

**Q3.1** A longer segment gives finer frequency resolution (good for resolving
the closely-spaced decadal frequencies) but fewer segments K, hence a noisier
PSD estimate and wider CI; a shorter segment does the reverse. For the annual
peak (well-separated, high-power) resolution isn't the constraint; for the
decadal band, resolution is exactly what's scarce.

**Q3.2** Overlap makes neighbouring segments correlated, so you have *fewer
effective independent* segments than K = 17. The true CI is therefore *wider*
than Eq. (5) reports — the reported uncertainty is mildly **optimistic**
(the paper notes this and uses the conservative ν = 2K anyway).

**Q4.1** (a) Random phases don't change `E|Z_k|² = A_k²`, so averaging over
members converges to the fixed target PSD. (b) In nature, different 1000-yr
realizations would differ in their *amplitudes* too (sampling variability of
the spectrum). Fixing `A_k` removes that source of spread, so between-member
variance is too small.

**Q4.2** In Gaussian space, multiplying PSD by `γ_b` scales that band's
variance by exactly `γ_b` (Parseval). But Eq. (3) is a *nonlinear* monotone
map to physical units, so it doesn't preserve variance ratios — it stretches
different parts of the distribution unequally. Hence physical band variance
scales only *approximately* by `γ_b`; the paper measures the realized ratio.

**Q4.3** The DC term must be real (it's the mean, fixed to zero); the Nyquist
term (for even N) must also be real for the inverse FFT to return a
real-valued time series. Randomizing their phases would inject an imaginary
component.

**Q5.1** A purely empirical inverse CDF can only output values within the
observed [min, max]. Under ×10 amplification, the many large synthetic
excursions all get clipped to those bounds, so the physical variance stops
growing — it *saturates* against a ceiling. The band experiment then can't be
read as "×10 variance in this band," because the actual delivered variance is
capped and confounded with the clipping artefact.

**Q5.2** A symmetric-Gaussian → asymmetric-CDF map yields a small mean offset;
under ×10 scaling that offset grows and would look like a change in *mean
accumulation*. Since the whole point of A3 is that experiments differ **only**
in spectral structure (not mean), the mean must be zeroed inside the inverse
transform so `μ` remains the single, controlled source of the mean.

**Q6.1** Because the observed spatial patterns `E_k(x,y)` are held fixed and
only the temporal PCs `PC_k(t)` are resynthesized, every synthetic field is a
linear combination of the *real* observed patterns — so it automatically has
realistic spatial structure.

**Q6.2** Basin integration is a single fixed spatial weighting (sum over
cells). Keeping one EOF mode is also a single fixed spatial pattern times a
time series. Both collapse the field to one spatial pattern × one temporal
signal; the scalar is just a particular (uniform-ish) choice of that pattern.

**Q6.3** A **propagating** or migrating feature — e.g. an anomaly that moves
across the basin over months — is represented by a *specific phase
relationship between two EOFs*. Synthesizing each PC independently randomizes
that relationship, destroying the coherent propagation. It reproduces each
mode's marginal spectrum but not cross-mode phase coupling.

**Q6.4** The SVD minimizes squared reconstruction error *of the matrix it is
handed*. To make it minimize *area-weighted* error, you must bake the weights
into the matrix (multiply by `√cos φ`) before the SVD — so that squared error
on the weighted matrix equals area-weighted squared error on the physical
field. Weighting afterwards would be too late; the modes would already be
optimized for the wrong (unweighted) objective. The `√` is because variance
is a *squared* quantity, so `(√cos φ)² = cos φ` gives the area weighting.

**Q7.1** The two halves of the record genuinely differ in mean and variance
(the record is non-stationary — there's a real trend/variability change over
1979–2023). A generator fit on one half *should* therefore mismatch the other
half by roughly that real difference. Getting 1.19/0.84 shows the pipeline is
faithfully transmitting the halves' actual statistics — not overfitting them
to 1.0.

**Q7.2** A short record gives a wide PSD CI (few Welch segments), and it's
*easy* for the ensemble mean to fall inside a wide band — coverage says little
when the band is loose. The honest report pairs coverage with the CI *width*
(the ~2.6× upper/lower ratio) and leans on the out-of-sample calibration-split
as the stronger evidence.

**QX** In FIT, the March-1993 value matters *individually* only through Stage 1
(it contributes to `μ`, the trend fit, and its month's climatology) and Stage 2
(it gets a rank). From the PSD estimate (Stage 3) onward it has dissolved into
an aggregate — the spectrum is a property of the whole series, and after that
the method only ever manipulates *distributions* (the inverse CDF) and
*spectra* (the PSD), never that specific value again. In GENERATE its
individual value plays no role at all: synthetic series are new draws from the
learned distribution + spectrum. So the method **keeps** the marginal
distribution and the second-order (spectral) structure, and **discards** the
specific temporal sequence of values.
