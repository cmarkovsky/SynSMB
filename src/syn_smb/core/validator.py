"""
validator.py
============
Quantitative validation and visualisation for SMBGenerator output.

Provides four methods of increasing rigour:

  compute_metrics()        — quantitative statistics comparing synthetic
                             ensemble to observed (in-sample)
  calibration_split_test() — out-of-sample validation by splitting the
                             observed record into two halves
  convergence_test()       — confirms the ensemble is large enough for
                             statistics to be stable
  plot_*()                 — publication-ready figures

The calibration_split_test() is the most important peer-review defence:
it breaks the circularity of calibrating and validating on the same
45-year record, and directly addresses reviewer questions about
overfitting to the training period.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import welch
from statsmodels.tsa.stattools import acf as compute_acf

from syn_smb.core.generator import SMBGenerator
from syn_smb.core.experiment import Experiment


class Validator:
    """
    Quantitative validation and visualisation for SMBGenerator.

    Parameters
    ----------
    generator : SMBGenerator
        A fitted generator. Must have is_fitted=True.
    smb : xr.DataArray
        The original observed SMB used to fit the generator.
        Stored for calibration split test and comparison figures.

    Examples
    --------
    >>> val = Validator(gen, smb)
    >>> metrics = val.compute_metrics(dataset)
    >>> split = val.calibration_split_test()
    >>> val.plot_validation_suite(dataset)
    >>> val.plot_ensemble_spaghetti(dataset)
    >>> val.plot_running_windows(dataset)
    """

    def __init__(self, generator: SMBGenerator, smb: xr.DataArray) -> None:
        if not generator.is_fitted:
            raise RuntimeError("Generator must be fitted before creating Validator.")
        self.generator = generator
        self.smb       = smb

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_ensemble(dataset: xr.Dataset, var: str = "smb_syn") -> np.ndarray:
        """Return all ensemble member values as a single 1D array."""
        return dataset[var].values.ravel()

    @staticmethod
    def _monthly_climatology(da: xr.DataArray) -> np.ndarray:
        """Monthly mean values, shape (12,), indexed Jan–Dec."""
        return np.array([
            float(da.sel(time=da.time.dt.month == m).mean())
            for m in range(1, 13)
        ])

    def _obs_n_years(self) -> int:
        return self.generator.n_obs // 12

    # ------------------------------------------------------------------
    # 1. compute_metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        dataset: xr.Dataset,
        verbose: bool = True,
    ) -> dict:
        """
        Compute quantitative metrics comparing synthetic ensemble to observed.

        Six metrics are reported:

        mean_ratio      — synthetic ensemble mean / observed mean.
                          Target: ~1.0. Deviations indicate mean shift.
        variance_ratio  — synthetic ensemble variance / observed variance.
                          Target: ~1.0.
        ks_statistic    — two-sample KS test statistic between observed
                          and a randomly drawn synthetic sample of the same
                          length. Target: small (< 0.1). The p-value is
                          not the primary target here since sample sizes
                          differ substantially.
        psd_rms_error   — root mean squared error of log10(PSD) between
                          ensemble mean and observed, excluding DC.
                          Target: < 0.5 (half a decade of PSD error).
        acf_lag1_error  — absolute difference in lag-1 month ACF.
                          Target: < 0.05.
        acf_lag12_error — absolute difference in lag-12 month (annual) ACF.
                          Target: < 0.05.
        seasonal_rmse   — RMSE of monthly climatology (synthetic vs observed).
                          Target: < 0.005 m w.e. (seasonal cycle restored
                          deterministically, so this should be near zero).

        Parameters
        ----------
        dataset : xr.Dataset
            Output of SMBGenerator.generate().
        verbose : bool
            If True, print formatted results.

        Returns
        -------
        metrics : dict
        """
        obs  = self.smb.values
        gen  = self.generator

        # --- Flatten ensemble for distributional metrics ---
        syn_flat = self._flatten_ensemble(dataset, "smb_syn")

        # For ACF and PSD, use one representative member (member 0)
        syn_member = dataset["smb_syn"].isel(member=0).values

        # 1. Mean ratio
        mean_ratio = float(np.mean(syn_flat) / np.mean(obs))

        # 2. Variance ratio
        variance_ratio = float(np.var(syn_flat) / np.var(obs))

        # 3. KS statistic (two-sample, match observed sample length)
        rng = np.random.default_rng(0)
        syn_sample = rng.choice(syn_flat, size=len(obs), replace=False)
        ks_stat, ks_pval = stats.ks_2samp(obs, syn_sample)

        # 4. PSD RMS error (log scale, excluding DC)
        fs = gen.spectral_synthesizer.fs
        nperseg = gen.nperseg
        _, psd_obs = welch(obs, fs=fs, nperseg=min(nperseg, len(obs)))
        _, psd_syn = welch(syn_member, fs=fs, nperseg=min(nperseg, len(syn_member)))
        # Trim to same length (synthetic may have more freqs)
        n_f = min(len(psd_obs), len(psd_syn))
        psd_rms = float(np.sqrt(np.mean(
            (np.log10(psd_obs[1:n_f]) - np.log10(psd_syn[1:n_f])) ** 2
        )))

        # 5 & 6. ACF errors
        nlags = 24
        acf_obs = compute_acf(obs,        nlags=nlags, fft=True)
        acf_syn = compute_acf(syn_member, nlags=nlags, fft=True)
        acf_lag1_err  = float(abs(acf_obs[1]  - acf_syn[1]))
        acf_lag12_err = float(abs(acf_obs[12] - acf_syn[12]))

        # 7. Seasonal cycle RMSE
        obs_clim = self._monthly_climatology(self.smb)
        syn_clim = np.array([
            float(dataset["smb_syn"].values[:, m::12].mean())
            for m in range(12)
        ])
        seasonal_rmse = float(np.sqrt(np.mean((obs_clim - syn_clim) ** 2)))

        metrics = {
            "mean_ratio":      mean_ratio,
            "variance_ratio":  variance_ratio,
            "ks_statistic":    ks_stat,
            "ks_pvalue":       ks_pval,
            "psd_rms_error":   psd_rms,
            "acf_lag1_error":  acf_lag1_err,
            "acf_lag12_error": acf_lag12_err,
            "seasonal_rmse":   seasonal_rmse,
        }

        if verbose:
            name = dataset.attrs.get("experiment_name", "")
            header = f"Metrics — {name}" if name else "Metrics"
            print(f"{header}")
            print(f"  mean_ratio:      {mean_ratio:.4f}   (target: ~1.0)")
            print(f"  variance_ratio:  {variance_ratio:.4f}   (target: ~1.0)")
            print(f"  ks_statistic:    {ks_stat:.4f}   (target: small)")
            print(f"  psd_rms_error:   {psd_rms:.4f}   (target: < 0.5)")
            print(f"  acf_lag1_error:  {acf_lag1_err:.4f}   (target: < 0.05)")
            print(f"  acf_lag12_error: {acf_lag12_err:.4f}   (target: < 0.05)")
            print(f"  seasonal_rmse:   {seasonal_rmse:.6f}  (target: ~0)")

        return metrics

    # ------------------------------------------------------------------
    # 2. calibration_split_test
    # ------------------------------------------------------------------

    def calibration_split_test(
        self,
        n_members: int = 30,
        n_years_syn: int = 100,
        seed: int = 0,
        verbose: bool = True,
    ) -> dict:
        """
        Out-of-sample validation via calibration-period split.

        Splits the observed SMB record into two approximately equal halves.
        Fits a new SMBGenerator on each half, generates a short ensemble,
        and computes metrics against the held-out half. Both directions
        are tested.

        This is the primary defence against the reviewer objection that
        the pipeline is being validated on its own training data. A
        generator calibrated on 1979–2001 should reproduce the statistics
        of 2001–2023, and vice versa.

        Parameters
        ----------
        n_members : int
            Ensemble size for each split. Default 30.
        n_years_syn : int
            Length of each synthetic ensemble in years. Should be at least
            as long as the split period (≥22 years). Default 100.
        seed : int
            Random seed. Default 0.
        verbose : bool
            If True, print formatted results.

        Returns
        -------
        results : dict
            Keys 'first_half' and 'second_half', each containing the
            metrics dict from compute_metrics().
        """
        n = self.generator.n_obs
        split = n // 2

        smb_first  = self.smb.isel(time=slice(None, split))
        smb_second = self.smb.isel(time=slice(split, None))

        results = {}

        for label, train, validate_on in [
            ("first_half",  smb_first,  smb_second),
            ("second_half", smb_second, smb_first),
        ]:
            if verbose:
                n_train = train.sizes['time'] // 12
                n_val   = validate_on.sizes['time'] // 12
                print(f"\n── Split: train on {label} ({n_train} yrs), "
                      f"validate on the other ({n_val} yrs) ──")

            # Fit a fresh generator on the training half
            gen_split = SMBGenerator(
                nperseg=min(self.generator.nperseg, train.sizes['time'] // 2),
                dt_years=self.generator.dt_years,
                remove_trend=self.generator.remove_trend,
                remove_seasonal=self.generator.remove_seasonal,
            )
            gen_split.fit(train)

            # Generate ensemble
            exp = Experiment(
                n_years=n_years_syn, n_members=n_members,
                seed=seed, name=f"split_{label}"
            )
            dataset = gen_split.generate(exp)

            # Compute metrics against the held-out half
            val_split = Validator(gen_split, validate_on)
            results[label] = val_split.compute_metrics(dataset, verbose=verbose)

        if verbose:
            print("\n── Summary ──")
            print(f"{'Metric':<20}  {'first_half':>12}  {'second_half':>12}")
            for k in results["first_half"]:
                v1 = results["first_half"][k]
                v2 = results["second_half"][k]
                print(f"  {k:<18}  {v1:>12.4f}  {v2:>12.4f}")
            print(
                "\n  Note: mean_ratio deviating from 1.0 is expected when "
                "the two halves\n  of the record have different means "
                "(e.g. due to a trend). The spectral\n  metrics "
                "(psd_rms_error, variance_ratio) are the primary "
                "indicators\n  of out-of-sample fidelity."
            )

        return results

    # ------------------------------------------------------------------
    # 3. convergence_test
    # ------------------------------------------------------------------

    def convergence_test(
        self,
        member_counts: list[int] | None = None,
        n_years: int = 100,
        seed: int = 0,
        verbose: bool = True,
    ) -> dict:
        """
        Test when ensemble statistics stabilise with increasing ensemble size.

        Generates ensembles of increasing size and reports how key metrics
        change. Stabilisation demonstrates that the chosen ensemble size
        is sufficient for the reported statistics to be reliable.

        Parameters
        ----------
        member_counts : list of int or None
            Ensemble sizes to test. Default [5, 10, 15, 20, 30, 50, 100].
        n_years : int
            Synthetic series length for each ensemble. Default 100.
        seed : int
            Random seed. Default 0.
        verbose : bool
            If True, print a table of results.

        Returns
        -------
        results : dict
            Keys 'member_counts', 'variance_ratio', 'psd_rms_error',
            'ks_statistic', 'mean_ratio'. Each value is a list aligned
            with member_counts.
        """
        if member_counts is None:
            member_counts = [5, 10, 15, 20, 30, 50, 100]

        variance_ratios = []
        psd_rms_errors  = []
        ks_statistics   = []
        mean_ratios     = []

        for n in member_counts:
            exp     = Experiment(n_years=n_years, n_members=n, seed=seed)
            dataset = self.generator.generate(exp)
            m       = self.compute_metrics(dataset, verbose=False)
            variance_ratios.append(m["variance_ratio"])
            psd_rms_errors.append(m["psd_rms_error"])
            ks_statistics.append(m["ks_statistic"])
            mean_ratios.append(m["mean_ratio"])

        results = {
            "member_counts":   member_counts,
            "variance_ratio":  variance_ratios,
            "psd_rms_error":   psd_rms_errors,
            "ks_statistic":    ks_statistics,
            "mean_ratio":      mean_ratios,
        }

        if verbose:
            print("Convergence test")
            print(f"  {'N':>5}  {'var_ratio':>10}  {'psd_rms':>10}  "
                  f"{'ks_stat':>10}  {'mean_ratio':>10}")
            for i, n in enumerate(member_counts):
                print(f"  {n:>5}  {variance_ratios[i]:>10.4f}  "
                      f"{psd_rms_errors[i]:>10.4f}  "
                      f"{ks_statistics[i]:>10.4f}  "
                      f"{mean_ratios[i]:>10.4f}")

        return results

    # ------------------------------------------------------------------
    # 4. plot_validation_suite
    # ------------------------------------------------------------------

    def plot_validation_suite(
        self,
        dataset: xr.Dataset,
        save_path: str | None = None,
    ) -> None:
        """
        Four-panel validation figure for the methods section.

        Panels:
          top-left:     PSD — observed (with 95% CI) vs ensemble mean
          top-right:    distribution — observed vs ensemble
          bottom-left:  ACF — observed vs representative synthetic member
          bottom-right: seasonal cycle — observed vs ensemble mean
        """
        gen    = self.generator
        obs    = self.smb.values
        ss     = gen.spectral_synthesizer
        syn_m0 = dataset["smb_syn"].isel(member=0).values

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        name = dataset.attrs.get("experiment_name", "")
        fig.suptitle(f"Validation suite{' — ' + name if name else ''}", fontsize=13)

        # ── top-left: PSD ──
        ax = axes[0, 0]
        syn_psds = []
        for i in range(dataset.sizes["member"]):
            _, p = welch(dataset["smb_syn"].isel(member=i).values,
                         fs=ss.fs, nperseg=min(gen.nperseg, gen.n_obs))
            syn_psds.append(p)
        _, psd_obs_plot = welch(obs, fs=ss.fs, nperseg=min(gen.nperseg, len(obs)))
        mean_psd_syn = np.array(syn_psds).mean(axis=0)
        f = np.fft.rfftfreq(min(gen.nperseg, len(obs)), d=gen.dt_years)[1:]

        ax.loglog(f, psd_obs_plot[1:], color="tab:blue", lw=2, label="Observed")
        ax.fill_between(f, ss.psd_ci_lower[1:len(f)+1],
                        ss.psd_ci_upper[1:len(f)+1],
                        color="tab:blue", alpha=0.15, label="95% CI")
        ax.loglog(f, mean_psd_syn[1:], color="tab:orange", lw=1.5,
                  label="Ensemble mean")
        ax.axvline(1.0, color="gray", linestyle=":", lw=1, alpha=0.6)
        ax.axvline(0.1, color="gray", linestyle=":", lw=1, alpha=0.6)
        ax.set_xlabel("Frequency (cycles/yr)")
        ax.set_ylabel("PSD")
        ax.set_title("Power spectral density")
        ax.legend(fontsize=9)

        # ── top-right: distribution ──
        ax = axes[0, 1]
        syn_flat = self._flatten_ensemble(dataset, "smb_syn")
        ax.hist(obs, bins=25, density=True, alpha=0.5,
                color="tab:blue", label="Observed")
        ax.hist(syn_flat, bins=50, density=True, alpha=0.4,
                color="tab:orange", label="Synthetic (all members)")
        ax.set_xlabel("SMB (m w.e. a⁻¹)")
        ax.set_ylabel("Density")
        ax.set_title("Marginal distribution")
        ax.legend(fontsize=9)

        # ── bottom-left: ACF ──
        ax = axes[1, 0]
        nlags = 36
        acf_obs = compute_acf(obs,    nlags=nlags, fft=True)
        acf_syn = compute_acf(syn_m0, nlags=nlags, fft=True)
        lags = np.arange(nlags + 1)
        ax.stem(lags,       acf_obs, linefmt="C0-", markerfmt="C0o",
                basefmt=" ", label="Observed")
        ax.stem(lags + 0.3, acf_syn, linefmt="C1-", markerfmt="C1o",
                basefmt=" ", label="Synthetic")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("ACF")
        ax.set_title("Autocorrelation function")
        ax.legend(fontsize=9)

        # ── bottom-right: seasonal cycle ──
        ax = axes[1, 1]
        months = np.arange(1, 13)
        labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        obs_clim = self._monthly_climatology(self.smb)
        syn_clim = np.array([
            float(dataset["smb_syn"].values[:, m::12].mean())
            for m in range(12)
        ])
        ax.plot(months, obs_clim, "o-", color="tab:blue", lw=1.5,
                markersize=5, label="Observed")
        ax.plot(months, syn_clim, "s--", color="tab:orange", lw=1.5,
                markersize=5, label="Synthetic mean")
        ax.set_xticks(months)
        ax.set_xticklabels(labels)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean SMB (m w.e. a⁻¹)")
        ax.set_title("Seasonal cycle")
        ax.legend(fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # 5. plot_ensemble_spaghetti
    # ------------------------------------------------------------------

    def plot_ensemble_spaghetti(
        self,
        dataset: xr.Dataset,
        save_path: str | None = None,
    ) -> None:
        """
        1000-year ensemble spaghetti with 45-year observed overlay.

        Shows the full synthetic ensemble as faint lines, the ensemble
        mean as a bold line, and the observed record overlaid at the
        start of the synthetic axis. Communicates both the plausibility
        of individual realisations and the ensemble spread.
        """
        smb_syn  = dataset["smb_syn"]
        n_members = smb_syn.sizes["member"]
        n_syn     = smb_syn.sizes["time"]
        n_obs     = self.generator.n_obs
        obs_years = n_obs / 12

        fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                 gridspec_kw={"height_ratios": [3, 1]})
        name = dataset.attrs.get("experiment_name", "")
        fig.suptitle(f"Synthetic ensemble{' — ' + name if name else ''}  "
                     f"({n_members} members, "
                     f"{n_syn // 12} years)", fontsize=12)

        # ── top: full time series ──
        ax = axes[0]
        t_syn = np.arange(n_syn) / 12          # model years
        t_obs = np.arange(n_obs) / 12          # observed years

        for i in range(n_members):
            ax.plot(t_syn, smb_syn.isel(member=i).values,
                    color="tab:orange", lw=0.4, alpha=0.3)
        ax.plot(t_syn, smb_syn.mean(dim="member").values,
                color="tab:red", lw=1.5, label="Ensemble mean", zorder=4)
        ax.plot(t_obs, self.smb.values,
                color="tab:blue", lw=1.2, label="Observed (45 yr)", zorder=5)
        ax.axhline(self.generator.smb_mean, color="tab:blue",
                   linestyle="--", lw=0.8, alpha=0.6)
        ax.set_xlim(0, n_syn / 12)
        ax.set_xlabel("Model year")
        ax.set_ylabel("SMB (m w.e. a⁻¹)")
        ax.legend(fontsize=9)

        # ── bottom: ensemble std over time (rolling 45-year windows) ──
        ax = axes[1]
        window = n_obs   # 45-year window in months
        n_windows = n_syn - window
        if n_windows > 0:
            rolling_std = np.array([
                smb_syn.values[:, i:i + window].std()
                for i in range(0, n_windows, 12)
            ])
            t_win = np.arange(len(rolling_std)) + obs_years / 2
            ax.plot(t_win, rolling_std, color="tab:purple", lw=1)
            ax.axhline(float(self.smb.std()), color="tab:blue",
                       linestyle="--", lw=1, label="Observed std")
            ax.set_xlabel("Window centre (model year)")
            ax.set_ylabel("Std (m w.e. a⁻¹)")
            ax.set_title(f"Rolling {int(obs_years)}-year std across all members")
            ax.legend(fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # 6. plot_running_windows
    # ------------------------------------------------------------------

    def plot_running_windows(
        self,
        dataset: xr.Dataset,
        save_path: str | None = None,
    ) -> None:
        """
        Distribution of 45-year window statistics from the 1000-year ensemble.

        Divides the synthetic series into non-overlapping windows of the
        same length as the observed record (45 years). Computes summary
        statistics for each window and plots their distribution against the
        single observed value. This answers the question: "how
        representative is the 45-year RACMO record of the long-run
        climatology?"
        """
        n_obs    = self.generator.n_obs
        n_obs_yr = n_obs // 12
        smb_syn  = dataset["smb_syn"]
        n_syn    = smb_syn.sizes["time"]

        obs_mean = float(self.smb.mean())
        obs_std  = float(self.smb.std())
        obs_clim_range = float(
            self.smb.groupby("time.month").mean().max()
            - self.smb.groupby("time.month").mean().min()
        )

        # Collect statistics from every non-overlapping window,
        # across all members
        window_means  = []
        window_stds   = []
        window_ranges = []

        for i in range(smb_syn.sizes["member"]):
            member = smb_syn.isel(member=i).values
            n_windows = n_syn // n_obs
            for w in range(n_windows):
                seg = member[w * n_obs: (w + 1) * n_obs]
                window_means.append(seg.mean())
                window_stds.append(seg.std())
                # Approximate seasonal range (max monthly mean - min)
                monthly = np.array([seg[m::12].mean() for m in range(12)])
                window_ranges.append(monthly.max() - monthly.min())

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        name = dataset.attrs.get("experiment_name", "")
        fig.suptitle(
            f"Distribution of {n_obs_yr}-year window statistics"
            f"{' — ' + name if name else ''}\n"
            f"({smb_syn.sizes['member']} members × "
            f"{n_syn // n_obs} windows per member = "
            f"{len(window_means)} windows total)",
            fontsize=11,
        )

        for ax, values, obs_val, xlabel, title in zip(
            axes,
            [window_means, window_stds, window_ranges],
            [obs_mean, obs_std, obs_clim_range],
            ["Mean SMB (m w.e. a⁻¹)", "Std SMB (m w.e. a⁻¹)",
             "Seasonal range (m w.e. a⁻¹)"],
            ["Window mean", "Window std dev", "Seasonal cycle range"],
        ):
            ax.hist(values, bins=30, density=True,
                    color="tab:orange", alpha=0.7, edgecolor="white")
            ax.axvline(obs_val, color="tab:blue", lw=2,
                       label=f"Observed ({obs_val:.4f})")
            pct = 100 * np.mean(np.array(values) <= obs_val)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.set_title(f"{title}\n(obs at {pct:.0f}th percentile)")
            ax.legend(fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # 7. plot_return_periods
    # ------------------------------------------------------------------

    def plot_return_periods(
        self,
        dataset: xr.Dataset,
        resample: str = "YS",
        save_path: str | None = None,
    ) -> None:
        """
        Empirical return period comparison: observed vs synthetic.

        Annual-mean SMB anomalies are computed from both the observed
        record and the full synthetic ensemble. The empirical return
        period (1 / exceedance probability) is plotted for both. The
        1000-year synthetic record characterises the tails far better
        than the 45-year observed record can, motivating the long
        synthetic series.

        Parameters
        ----------
        resample : str
            Temporal resampling for computing annual means. Default 'YS'.
        """
        # Annual means of observed
        obs_annual = self.smb.resample(time=resample).mean().values
        obs_anom   = obs_annual - obs_annual.mean()

        # Annual means across full synthetic ensemble (all members concatenated)
        syn_vals = dataset["smb_syn"].values  # (member, time)
        n_yrs    = syn_vals.shape[1] // 12
        syn_annual_all = []
        for i in range(syn_vals.shape[0]):
            ann = syn_vals[i].reshape(n_yrs, 12).mean(axis=1)
            syn_annual_all.append(ann - ann.mean())
        syn_anom = np.concatenate(syn_annual_all)

        def empirical_rp(x):
            """Return (sorted_values, return_periods) for a 1D array."""
            x_sorted = np.sort(x)[::-1]
            n = len(x_sorted)
            rp = (n + 1) / np.arange(1, n + 1)
            return x_sorted, rp

        obs_vals_rp, obs_rp = empirical_rp(obs_anom)
        syn_vals_rp, syn_rp = empirical_rp(syn_anom)

        fig, ax = plt.subplots(figsize=(9, 6))
        name = dataset.attrs.get("experiment_name", "")
        ax.set_title(
            f"Empirical return periods — annual-mean SMB anomaly"
            f"{chr(10) + name if name else ''}",
            fontsize=11,
        )
        ax.semilogx(obs_rp, obs_vals_rp, "o", markersize=4,
                    color="tab:blue", alpha=0.8,
                    label=f"Observed ({len(obs_anom)} years)")
        ax.semilogx(syn_rp, syn_vals_rp, ".", markersize=2,
                    color="tab:orange", alpha=0.3,
                    label=(f"Synthetic ({syn_vals.shape[0]} members × "
                           f"{n_yrs} yrs = {len(syn_anom)} yr-equivalents)"))
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Annual-mean SMB anomaly (m w.e. a⁻¹)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_obs_yr = self.generator.n_obs // 12
        return (
            f"Validator(generator={repr(self.generator)}, "
            f"n_obs_years={n_obs_yr})"
        )