import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from syn_smb import BandpassFilter
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import welch
import seaborn as sns
from scipy.stats import linregress
from pmdarima import auto_arima

class SyntheticGenerator:
    def __init__(self, 
                 smb: xr.DataArray,
                 n_years: int = 100,
                 hf_cen: int = 1,
                 mf_cen: int = 10,
                 lf_cen: int = 25,
                 dt: float = 1.0 / 12,
                 dim: str = "time",
                 seed: int = 42):
        self.smb = smb
        self.dim = dim
        self._check_data()
        self.n_years = n_years
        self.hf_cen, self.mf_cen, self.lf_cen = hf_cen, mf_cen, lf_cen
        self.dt = dt
        self.seed = seed
        np.random.seed(self.seed)
        self.hf, self.mf, self.lf = self._calc_components(self.smb)

    def _check_data(self):
        """Check if the data is a valid xarray DataArray."""
        if not isinstance(self.smb, xr.DataArray):
            raise TypeError("Input must be an xarray.DataArray")
        if self.dim not in self.smb.dims:
            raise ValueError(f"DataArray must have a '{self.dim}' dimension")
        if self.smb.ndim != 1:
            raise ValueError("DataArray must be one-dimensional for AR model fitting")

    def _calc_components(self, smb: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Calculate the high-frequency, mid-frequency, and low-frequency components of the synthetic data.
        """
        hf_filter = BandpassFilter(filt_center=self.hf_cen, alpha=1.25)
        hf_filter.filter(smb)

        mf_filter = BandpassFilter(filt_center=self.mf_cen, alpha=1.5)
        mf_filter.filter(smb)

        lf_filter = BandpassFilter(filt_center=self.lf_cen, alpha=2)
        lf_filter.filter(smb)

        self.filter_params = {
            "hf": hf_filter.get_filter_params(),
            "mf": mf_filter.get_filter_params(),
            "lf": lf_filter.get_filter_params(),
        }
        return hf_filter.filter(smb), mf_filter.filter(smb), lf_filter.filter(smb)

    def get_components(self) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Return the high-frequency, mid-frequency, and low-frequency components."""
        return self.hf, self.mf, self.lf

    def analyze_psd(self, fs: float = 12.0, plot: bool = True):
        """Analyze the power spectral density of the data."""
        f_raw, Pxx_raw = welch(self.smb.values, fs=fs, nperseg=min(512, len(self.smb)))
        f_hf, Pxx_hf = welch(self.hf.values, fs=fs, nperseg=min(512, len(self.hf)))
        f_mf, Pxx_mf = welch(self.mf.values, fs=fs, nperseg=min(512, len(self.mf)))
        f_lf, Pxx_lf = welch(self.lf.values, fs=fs, nperseg=min(512, len(self.lf)))

        hf_filter, mf_filter, lf_filter = self.get_filter_params().values()

        self.psds = {
            "raw": (f_raw, Pxx_raw),
            "hf": (f_hf, Pxx_hf),
            "mf": (f_mf, Pxx_mf),
            "lf": (f_lf, Pxx_lf),
        }

        if plot:
            fig, axs = plt.subplot_mosaic([['A'], ['B'], ['C'], ['D']], figsize=(12, 8), layout='constrained', sharex=True)
            ax1, ax2, ax3, ax4 = axs['A'], axs['B'], axs['C'], axs['D']

            ax1.loglog(f_raw, Pxx_raw, label="Raw", color="black")
            ax1.axvspan(12*hf_filter['high_freq'], 12*hf_filter['low_freq'], color='orange', alpha=0.1, label="HF band")
            ax1.axvspan(12*mf_filter['high_freq'], 12*mf_filter['low_freq'], color='red', alpha=0.1, label="MF band")
            ax1.axvspan(12*lf_filter['high_freq'], 12*lf_filter['low_freq'], color='yellow', alpha=0.1, label="LF band")

            ax1.set_xlabel("")
            ax1.set_ylabel("Power")
            ax1.set_title("Power Spectral Density (log-log)")
            ax1.grid(True)
            ax1.legend()

            ax2.loglog(f_hf, Pxx_hf, label="High-Frequency", color="tab:blue")
            ax2.axvspan(12*hf_filter['high_freq'], 12*hf_filter['low_freq'], color='orange', alpha=0.1, label="HF band")
            # ax2.axvspan(12/169.7, 12/84.8, color='red', alpha=0.1, label="MF band")
            # ax2.axvspan(12/424.25, 12/212, color='yellow', alpha=0.1, label="LF band")
            ax2.set_xlabel("")
            ax2.set_ylabel("Power")
            ax2.set_title("")
            ax2.grid(True)
            ax2.legend()

            ax3.loglog(f_mf, Pxx_mf, label="Mid-Frequency", color="tab:orange")
            # ax3.axvspan(12*hf_filter['high_freq'], 12*hf_filter['low_freq'], color='tab:orange', alpha=0.1, label="HF band")
            ax3.axvspan(12*mf_filter['high_freq'], 12*mf_filter['low_freq'], color='tab:red', alpha=0.1, label="MF band")
            # ax3.axvspan(12*lf_filter['high_freq'], 12*lf_filter['low_freq'], color='tab:yellow', alpha=0.1, label="LF band")
            ax3.set_xlabel("")
            ax3.set_ylabel("Power")
            ax3.set_title("")
            ax3.grid(True)
            ax3.legend()

            ax4.loglog(f_lf, Pxx_lf, label="Low-Frequency", color="tab:green")
            # ax4.axvspan(12/16.97, 12/8.48, color='tab:orange', alpha=0.1, label="HF band")
            # ax4.axvspan(12/169.7, 12/84.8, color='tab:red', alpha=0.1, label="MF band")
            ax4.axvspan(12*lf_filter['high_freq'], 12*lf_filter['low_freq'], color='yellow', alpha=0.1, label="LF band")
            ax4.set_xlabel("Frequency (cycles/year)")
            ax4.set_ylabel("Power")
            ax4.set_title("")
            ax4.grid(True)
            ax4.legend()
            plt.show()

        return 
        # return f_raw, Pxx_raw

    def _calc_beta(self, f, Pxx, filter: dict) -> tuple[float, float]:
        # print(f"Calculating β for filter centered at {filter['filt_center']} years")
        print(filter)

        mask = (f >= filter['low_freq']*12) & (f <= filter['high_freq']*12)
        print(len(f[mask]), "points in fit")
        log_f = np.log(f[mask])
        log_P = np.log(Pxx[mask])

        slope, intercept, r_value, *_ = linregress(log_f, log_P)
        beta = -slope # type: ignore
        print(f"Estimated β ≈ {beta:.2f}, R² = {r_value**2:.2f}") # pyright: ignore[reportOperatorIssue]
        print("Band Power: ", np.trapz(Pxx[mask], f[mask]))
        return beta, r_value**2 # pyright: ignore[reportOperatorIssue]

    def check_psds(self):
        self.analyze_psd(plot=True)
        betas = {}
        for name, filter in self.get_filter_params().items():
            print(name)
            f, Pxx = self.psds[name]
            # print('f', f[:10], 'Pxx', Pxx[:10])
            # print(name, filter)
            betas[name] = self._calc_beta(f, Pxx, filter=filter) # type: ignore
            # print(f"β for {name}: {betas[name][0]:.2f}, R² = {betas[name][1]:.2f}")
        return betas

    def get_filter_params(self, filter_name = None):
        """
        Get the filter parameters for a specific frequency band.
        """
        if filter_name == None:
            return self.filter_params
        elif filter_name in self.filter_params:
            return self.filter_params[filter_name]
        else:
            raise ValueError(f"Unknown filter name: {filter_name}")

    def _fit_ar(self, filtered_data: xr.DataArray, max_p: int = 4, seasonal = False):
        """
        Fit an autoregressive model to the filtered data using autoarima.
        """

        if seasonal:
            model = auto_arima(
                filtered_data.values,
                start_p=1, max_p=max_p,
                d=0,
                seasonal=seasonal,
                m=12,  # 12 months in a year
                start_P=1, max_P=2,
                D=0,
                stepwise=True,
                suppress_warnings=True
            )
        else:
            model = auto_arima(
                filtered_data.values,
                start_p=1, max_p=max_p,
                d=0,
                seasonal=seasonal,
                m=12,  # 12 months in a year
            )

        return model

    def fit_ar_models(self):
        """
        Fit autoregressive models to each frequency component.
        """
        models = {}
        self.plot_components()
        # for name, filtered_data in zip(['hf', 'mf', 'lf'], self.get_components()):
        #     print(f"Fitting AR model to {name}-component")
        #     if name == 'hf':
        #         model = self._fit_ar(filtered_data, max_p=4, seasonal=True)
        #     elif name == 'mf':
        #         model = self._fit_ar(filtered_data, max_p=10, seasonal=False)
        #     elif name == 'lf':
        #         model = self._fit_ar(filtered_data, max_p=15, seasonal=False)
        #     else:
        #         raise ValueError(f"Unknown component name: {name}")
        #     print(model.summary())
        #     models[name] = model
        # return models
    
    def plot_components(self):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True, layout='constrained')
        colors = {
            'raw': 'tab:blue',
            'hf': 'tab:orange',
            'mf': 'tab:green',
            'lf': 'tab:red'
        }
        # Plot raw (unfiltered) data
        axs[0].plot(self.smb.time, self.smb.values, color=colors['raw'], label='raw')
        axs[0].set_title("Raw (Unfiltered) Data")
        axs[0].legend()
        # Plot each component
        for ax, (name, filtered_data) in zip(axs[1:], zip(['hf', 'mf', 'lf'], self.get_components())):
            ax.plot(filtered_data.time, filtered_data.values, color=colors[name], label=name)
            # ax.set_title(f"{name} Component")
            ax.legend()
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()