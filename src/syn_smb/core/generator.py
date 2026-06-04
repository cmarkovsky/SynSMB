import xarray as xr
from pathlib import Path
from syn_smb.core.generator_utils import *
# from syn_smb.plotting_utils import *
# from syn_smb3.ensemble_utils import *
from statsmodels.tsa.stattools import acf
import pandas as pd

class SMBGenerator:

    def __init__(self, smb_path: str, smb_var = "smbgl", detrend = True, gaussianize = True, psd_method = "welch") -> None:
        self.smb_path = Path(smb_path)
        self.smb_var = smb_var
        self.smb = load_data(self.smb_path, smb_var=self.smb_var)
        self.smb_mean = self.smb.mean().item()
        self.detrend = detrend
        self.gaussianize = gaussianize
        self.psd_method = psd_method
        self.detrend_smb()
        self.gaussianize_smb()
        self.obs_freqs, self.obs_psd = estimate_psd(self.g_resid, method=self.psd_method)
        # self.bands = self.create_band_factors()

    def detrend_smb(self) -> None:
        self.smb_centered = self.smb - self.smb_mean
        if self.detrend:
            self.resid, self.trend = detrend_linear(self.smb_centered)
            self.smb_detrended = self.resid + self.smb_mean
        else:
            self.resid = self.smb_centered
            self.trend = None

    def gaussianize_smb(self) -> None:
        self.g_resid, self.u = gaussian_rank_transform(self.resid)
        self.inv_cdf = build_empirical_inv_cdf(self.resid)
        self.inv_cdf_resid = self.inv_cdf(self.resid.values)

    def create_band_factors(self, ann_min: float = 0.8, ann_max: float = 1.5, ann_factor: float = 1.0, dec_min: float = 8, dec_max: float = 20, dec_factor: float = 1.0) -> dict[str, tuple[float,float,float]]:
        """
        Create band factors for scaling PSD in specified frequency bands.
        Returns list of tuples (fmin, fmax, factor).
        """
        def period_to_freq_bounds(pmin, pmax):
            fmax = 1.0 / pmin
            fmin = 1.0 / pmax
            return fmin, fmax

        ann_fmin, ann_fmax = period_to_freq_bounds(ann_min, ann_max)
        dec_fmin, dec_fmax = period_to_freq_bounds(dec_min, dec_max)

        band_factors = {
            "annual": (ann_fmin, ann_fmax, ann_factor),
            "decadal": (dec_fmin, dec_fmax, dec_factor)
        }
        return band_factors

    def generate_gaussian_resid_series(
        self,
        g_resid: xr.DataArray,
        freqs_obs: np.ndarray,
        psd_obs: np.ndarray,
        N_syn: int = 100,
        dt_years: float = 1 / 12,
        band_factors: list[tuple[float,float,float]] | None = None,
        rng=None,
        ) -> xr.DataArray:
        """
        Generate one synthetic Gaussianized residual series of length N_syn.
        Returns xr.DataArray with dim 'time_syn'.
        """
        if rng is None:
            rng = np.random.default_rng()
        freqs_syn = synthetic_freq_grid(N_syn, dt_years=dt_years)
        psd_syn = interpolate_psd_to_grid(freqs_obs, psd_obs, freqs_syn)

        if band_factors is not None:
            # print("Applying band factors to PSD...")
            # print(f"Band factors: {band_factors}")
            psd_syn = scale_psd_bands(freqs_syn, psd_syn, band_factors)

        g_syn = simulate_gaussian_from_psd(freqs_syn, psd_syn, N_syn, dt_years=dt_years, rng=rng)

        # build synthetic time coordinate
        t0 = g_resid["time"].values[0]
        # treat as annual spacing
        time_syn = xr.cftime_range(start=str(pd.to_datetime(t0).year), periods=N_syn, freq="MS")
        g_syn_da = xr.DataArray(g_syn, coords={"time_syn": time_syn}, dims=("time_syn",), name="g_resid_syn")
        return g_syn_da
    
    def generate(self, N_years: int = 100, rng = None, band_factors: list[tuple[float,float,float]] | None = None) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Generate one synthetic SMB time series.
        Returns xr.DataArray with dim 'time_syn'.
        """
        if self.resid is None:
            raise ValueError("Residuals not available. Cannot generate synthetic series.")

        # Estimate PSD of Gaussianized residuals
        freqs_obs, psd_obs = estimate_psd(self.g_resid, method=self.psd_method)

        dt_years = 1 / 12  # assuming monthly data; adjust if needed

        N_syn = int(N_years * 1 / dt_years) # assuming monthly data; adjust if needed

        g_syn_da = self.generate_gaussian_resid_series(
            self.g_resid,
            freqs_obs,
            psd_obs,
            N_syn=N_syn,
            dt_years=dt_years,
            rng=rng,
            band_factors=band_factors
        )

        self.g_syn = g_syn_da
        # # Inverse Gaussianization
        resid_syn = inverse_gaussianize_to_resid(g_syn_da.values, self.inv_cdf)
        resid_syn_da = xr.DataArray(resid_syn, coords=g_syn_da.coords, dims=g_syn_da.dims, name="resid_syn")

        # # Reconstruct SMB
        smb_syn_values = reassemble_smb(resid_syn, self.smb_mean, None)
        smb_syn_da = xr.DataArray(smb_syn_values, coords=g_syn_da.coords, dims=g_syn_da.dims, name="smb_syn")
        # smb_syn_values = resid_syn_da.values + trend_syn
        # smb_syn_da = xr.DataArray(smb_syn_values, coords=g_syn_da.coords, dims=g_syn_da.dims, name="smb_syn")

        return smb_syn_da, resid_syn_da, g_syn_da

    ### Checks ###

    def check_gaussianization(self) -> None:
        if self.gaussianize and self.resid is not None:
            pre_gauss_corr = np.corrcoef(self.resid.values[:-1], self.resid.values[1:])[0,1]
            post_gauss_corr = np.corrcoef(self.g_resid.values[:-1], self.g_resid.values[1:])[0,1] # type: ignore
            print(f"Pre-Gaussianization autocorrelation: {pre_gauss_corr}")
            print(f"Post-Gaussianization autocorrelation: {post_gauss_corr}")
        else:
            print("Gaussianization not performed or residuals not available.")        

    ### Printing functions ###
    def print_smb_info(self) -> None:
        print(f"SMB DataArray info:")
        print(self.smb)
        if self.resid is not None:
            print(f"Detrended Residuals info:")
            print(self.resid)
        if self.g_resid is not None:
            print(f"Gaussianized Residuals info:")
            print(self.g_resid)
    
    def print_psd_info(self) -> None:
        if self.resid is None:
            print("Residuals not available. Cannot compute PSD info.")
            return
        freqs, psd = estimate_psd(self.resid, method=self.psd_method)
        print(f"Estimated PSD using method '{self.psd_method}':")
        for f, p in zip(freqs, psd):
            print(f"Freq: {f:.4f}, PSD: {p:.4f}")
    
    def print_band_factors(self) -> None:
        print("Band factors:")
        for band_name, (fmin, fmax, factor) in self.bands.items():
            print(f"{band_name}: pmin={1/fmax:.2f}, fmin={fmin:.4f}, pmax={1/fmin:.2f}, fmax={fmax:.4f}, factor={factor}")
    
    def print_moments(self, syn_smb = None) -> None:
        obs_smb = self.smb_detrended if self.detrend else self.smb
        obs_moments = calc_moments(obs_smb)

        print("Observed Detrended SMB moments:")

        for name, value in obs_moments.items():
            print(f"{name}: {value:.4f}")
        if syn_smb is not None:
            syn_moments = calc_moments(syn_smb)
            print("\nSynthetic SMB moments:")
            for name, value in syn_moments.items():
                print(f"{name}: {value:.4f}")


    ### Plotting functions ###

    def plot_smb(self, detrended = False) -> None:
        if detrended and self.resid is not None:
            plot_detrended_smb(self.smb, self.resid, self.trend) # type: ignore
        else:
            plot_smb_time_series(self.smb, self.trend) # type: ignore
    
    def plot_psd(self, save = False) -> None:
        from scipy import signal
        import numpy as np

        x1 = self.smb.values
        x2 = self.resid.values
        x3 = self.g_resid.values
        


        # PSD
        fs = 1.0 / (1 / 12)
        freqs1, psd1 = signal.welch(x1, fs=fs, nperseg=min(60, len(x1)))
        # freqs2, psd2 = signal.welch(x2, fs=fs, nperseg=min(60, len(x2)))

        fig, ax = plt.subplots(figsize=(10,6))

        ax.loglog(freqs1, psd1, color='tab:blue')
        # ax.loglog(freqs2, psd2, color='tab:orange', label='Detrended Residuals')

        # Define bands (cycles per year)
        annual_band = (0.8, 1.2)          # ~1 year period
        decadal_band = (1/15, 1/8)        # ~8-30 year periods

        # # Shade bands
        ymin, ymax = ax.get_ylim()
        ax.fill_betweenx([ymin, ymax], annual_band[0], annual_band[1], color='orange', alpha=0.18, label='Annual band')
        ax.fill_betweenx([ymin, ymax], decadal_band[0], decadal_band[1], color='green', alpha=0.12, label='Decadal band')

        # # Marker lines for center frequencies
        ax.axvline(1.0, color='orange', linestyle='--', linewidth=1)
        ax.axvline(1/10, color='green', linestyle='--', linewidth=1)

        ax.set_xlabel("Frequency (cycles/yr)")
        ax.set_ylabel("PSD")
        ax.set_title("Power Spectral Density of RACMO SMB")
        plt.legend(loc='best')
        if save:
            plt.savefig("./figures/smb_obs_psd.png", dpi=300)
        plt.show()



    def plot_gaussianization_check(self) -> None:
        self.check_gaussianization()
        if self.gaussianize and self.resid is not None:
            plot_gaussianization_check(self.resid, self.g_resid) # type: ignore
        else:
            print("Gaussianization not performed or residuals not available.")
    
    def plot_acf(self, gaussian = True) -> None:
        if gaussian and self.g_resid is not None:
            plot_acf(self.resid, self.g_resid, title="ACF of Gaussianized Residuals") # type: ignore
        else:
            print("Requested data not available for ACF plot.")

    def plot_psd_acf(self, plot_gaussian=True) -> None:
        if self.resid is not None and self.g_resid is not None:
            plot_psd_acf(self.resid, self.g_resid, method=self.psd_method, plot_gaussian=plot_gaussian) # type: ignore
        else:
            print("Residuals not available for PSD and ACF plot.")

    def compare_synthetic_psd(self, g_syn: xr.DataArray, dt: float = 1.0/12) -> None:
        if self.g_resid is None:
            print("Gaussianized residuals not available for PSD comparison.")
            return
        freqs_obs, psd_obs = estimate_psd(self.smb, method=self.psd_method, dt_years=dt)
        freqs_syn, psd_syn = estimate_psd(g_syn, method=self.psd_method, dt_years=dt)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.loglog(freqs_obs, psd_obs, label="Observed Gaussianized Residuals PSD")
        ax.loglog(freqs_syn, psd_syn, label="Synthetic Gaussianized Residuals PSD")
        ax.set_xlabel("Frequency (1/years)")
        ax.set_ylabel("PSD")
        ax.set_title("PSD Comparison")
        ax.legend()
        plt.show()

    def compare_smb_obs_syn(self, smb_syn: xr.DataArray) -> None:
        plot_smb_obs_syn(self.smb, smb_syn)
    
    def compare_gaussian_obs_syn(self, g_syn: xr.DataArray) -> None:
        plot_psd_comparison_gaussian(self.g_resid, g_syn, dt=1/12, method=self.psd_method) # type: ignore

    

        
