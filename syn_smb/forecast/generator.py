import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from syn_smb import BandpassFilter
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import welch
import seaborn as sns

# from syn_smb.forecast.bandpass import BandpassFilter

class Generator:
    def __init__(self, 
                 smb: xr.DataArray,
                 n_years: int = 100,
                 hf_band: tuple = (1/1.5/12, 1/0.5/12),
                 lf_band: tuple = (1/50/12, 1/10/12),
                 dt: float = 1.0 / 12,
                 dim: str = "time",
                 seed: int = 42):
        self.smb = smb
        self.dim = dim
        self._check_data()
        self.n_years = n_years
        self.hf_band = hf_band
        self.lf_band = lf_band
        self.dt = dt
        self.seed = seed
        np.random.seed(self.seed)
        self.hf, self.lf = self.calc_components(self.smb)

        # self.plot_smb_components()
        # self.phi, self.intercept, self.sigma = self._fit_ar()

    def _check_data(self):
        """Check if the data is a valid xarray DataArray."""
        if not isinstance(self.smb, xr.DataArray):
            raise TypeError("Input must be an xarray.DataArray")
        if self.dim not in self.smb.dims:
            raise ValueError(f"DataArray must have a '{self.dim}' dimension")
        if self.smb.ndim != 1:
            raise ValueError("DataArray must be one-dimensional for AR model fitting")

    def generate_smb(self, plot: bool = True) -> xr.DataArray:

        synthetic_hf = self.generate_hf(plot=False)
        synthetic_lf = self.generate_lf(plot=False)
        synthetic_smb = synthetic_hf + synthetic_lf + self.smb.mean(dim=self.dim)
        if plot:
            # Plot the original and synthetic SMB data
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=False)

            # Synthetic SMB
            sns.lineplot(ax=axes[0], x=synthetic_smb[self.dim], y=synthetic_smb, label='Synthetic SMB', color='tab:blue')
            sns.lineplot(ax=axes[0], x=synthetic_hf[self.dim], y=synthetic_hf + self.smb.mean(dim=self.dim), label='Synthetic HF', color='tab:orange', alpha = 0.7)

            axes[0].set_ylabel('SMB (m w.e.)')
            axes[0].set_title('Synthetic SMB')
            axes[0].legend()

            # Synthetic High-Frequency Component
            sns.lineplot(ax=axes[1], x=synthetic_hf[self.dim], y=synthetic_hf + self.smb.mean(dim=self.dim), label='Synthetic HF', color='tab:orange')
            axes[1].set_ylabel('SMB (m w.e.)')
            axes[1].set_title('Synthetic High-Frequency Component')
            axes[1].legend()

            # Synthetic Low-Frequency Component
            sns.lineplot(ax=axes[2], x=synthetic_lf[self.dim], y=synthetic_lf + self.smb.mean(dim=self.dim), label='Synthetic LF', color='tab:red')
            axes[2].set_xlabel(self.dim)
            axes[2].set_ylabel('SMB (m w.e.)')
            axes[2].set_title('Synthetic Low-Frequency Component')
            axes[2].legend()

            plt.tight_layout()
            plt.show()
        return synthetic_smb

    def generate_hf(self, plot: bool = True) -> xr.DataArray:
        """
        Generate high-frequency synthetic data using the high-frequency bandpass filter.
        """

        phi, intercept, sigma = self._fit_ar()
        # Generate synthetic high-frequency data
        synthetic_hf = self._generate(phi, intercept, sigma)
        # Apply the high-frequency bandpass filter
        hf_filter = BandpassFilter(filt_center=1, dim=self.dim)
        hf_filtered = hf_filter.filter(synthetic_hf)

        if plot:
            # Plot the original and filtered synthetic high-frequency data
            plt.figure(figsize=(10, 5))
            sns.lineplot(x=synthetic_hf[self.dim], y=synthetic_hf, label='Synthetic HF (raw)', alpha=0.7)
            sns.lineplot(x=hf_filtered[self.dim], y=hf_filtered, label='Synthetic HF (filtered)', alpha=0.7)
            plt.xlabel(self.dim)
            plt.ylabel('SMB (m w.e.)')
            plt.title('Synthetic High-Frequency SMB (Raw vs Filtered)')
            plt.legend()
            plt.tight_layout()
            plt.show()

        return hf_filtered
        # b_hf, a_hf = butter(4, [self.hf_band[0], self.hf_band[1]], btype='band')

    def _generate(self, phi, intercept, sigma) -> xr.DataArray:
        """
        Generate synthetic SMB data using the fitted AR model.
        """
        # Initialize the synthetic data array
        np.random.seed(self.seed)
        n = self.n_years * 12  # Total number of months
        synthetic_data = np.zeros(n)
        noise = np.random.normal(scale=sigma, size=n)
        # Set the first value to the mean of the original data
        synthetic_data[0] = intercept / (1 - phi)  # Mean of the stationary process

        # Generate the synthetic data
        for t in range(1, n):
            synthetic_data[t] = intercept + phi * synthetic_data[t - 1] + noise[t]

        # Convert the synthetic data to an xarray DataArray
        synthetic_da = xr.DataArray(synthetic_data, dims=[self.dim])
        return synthetic_da

    def _fit_ar(self):
        """
        Fit an AutoRegressive (AR) model to the SMB data.
        """
        
        # Drop NaN values along the specified dimension
        data = self.smb.dropna(self.dim)

        # Convert the DataArray to a numpy array for AR model fitting
        values = data.values

        # Fit the AR model
        model = AutoReg(values, lags=1, old_names=False).fit()
        phi = model.params[1]
        intercept = model.params[0]
        sigma = np.std(model.resid)

        return phi, intercept, sigma

        

    def _generate_spectral_match(self, n: int) -> xr.DataArray:
        """
        Generate synthetic data that matches the spectral properties of the original SMB data.
        """
        # Generate random noise
        np.random.seed(self.seed + 1)
        white_noise = np.random.normal(0, 1, n)
        f_ref, Pxx_ref = welch(self.smb.values, fs=1/self.dt, nperseg=min(len(self.smb), 256))
        freqs = np.fft.rfftfreq(n, d=self.dt)
        Pxx_interp = np.interp(freqs, f_ref, Pxx_ref)
        fft_white = np.fft.rfft(white_noise)
        shaped_fft = fft_white * np.sqrt(Pxx_interp)
        synthetic_values = np.fft.irfft(shaped_fft, n=n)
        synthetic_da = xr.DataArray(synthetic_values, dims=[self.dim])
        synthetic_da.attrs = self.smb.attrs  # Preserve original attributes
        return synthetic_da
    
    def generate_lf(self, plot: bool = True) -> xr.DataArray:
        # synthetic_lf = self._generate_spectral_match(self.n_years * 12)
        # Apply the low-frequency bandpass filter
        var_lf_obs = np.var(self.lf_band)

        np.random.seed(self.seed + 2)
        white_noise = xr.DataArray(np.random.normal(0, self.smb.std(), self.n_years * 12), dims=[self.dim])
        lf_filter = BandpassFilter(filt_center=10, dim=self.dim)
        synthetic_lf = lf_filter.filter(white_noise)
        # synthetic_lf *= np.sqrt(var_lf_obs / np.var(synthetic_lf))  # Scale to match the variance of the low-frequency band

        if plot:
            # Plot the original and synthetic low-frequency data
            plt.figure(figsize=(10, 5))
            sns.lineplot(x=synthetic_lf[self.dim], y=synthetic_lf, label='Synthetic LF', alpha=0.7)
            plt.xlabel(self.dim)
            plt.ylabel('SMB (m w.e.)')
            plt.title('Original vs Synthetic Low-Frequency SMB')
            plt.legend()
            plt.tight_layout()
            plt.show()
        return synthetic_lf
    

    def plot_smb_components(self):
        """
        Plot the high-frequency and low-frequency components of the SMB data.
        """
        plt.figure(figsize=(12, 9))

        # High-Frequency Component
        plt.subplot(3, 1, 1, sharex=True, sharey=True)
        sns.lineplot(x=self.smb[self.dim], y=self.smb, label='Original SMB', color='tab:blue')
        sns.lineplot(x=self.smb[self.dim], y=self.hf + self.smb.mean(), label='High-Frequency Component', color='tab:orange')
        plt.title('High-Frequency Component of SMB')
        plt.xlabel(self.dim)
        plt.ylabel('SMB (m w.e.)')
        plt.legend()

        # Low-Frequency Component
        plt.subplot(3, 1, 2)
        sns.lineplot(x=self.smb[self.dim], y=self.smb, label='Original SMB', color='tab:blue')
        sns.lineplot(x=self.smb[self.dim], y=self.lf + self.smb.mean(), label='Low-Frequency Component', color='tab:red')
        plt.title('Low-Frequency Component of SMB')
        plt.xlabel(self.dim)
        plt.ylabel('SMB (m w.e.)')
        plt.legend()

        # Both Components Together
        plt.subplot(3, 1, 3)
        sns.lineplot(x=self.smb[self.dim], y=self.smb, label='Original SMB', color='tab:blue')
        sns.lineplot(x=self.smb[self.dim], y=self.hf + self.smb.mean(), label='High-Frequency', color='tab:orange')
        sns.lineplot(x=self.smb[self.dim], y=self.lf + self.smb.mean(), label='Low-Frequency', color='tab:red')
        plt.title('Original, High-Frequency, and Low-Frequency Components')
        plt.xlabel(self.dim)
        plt.ylabel('SMB (m w.e.)')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def calc_components(self, smb: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Calculate the high-frequency and low-frequency components of the synthetic data.
        """
        hf_filter = BandpassFilter(filt_center=1)
        hf_filter.filter(smb)

        lf_filter = BandpassFilter(filt_center=10)
        lf_filter.filter(smb)

        return hf_filter.filter(smb), lf_filter.filter(smb)
        # High-frequency component
    # def calc_components(self, smb: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    #     """
    #     Calculate the high-frequency and low-frequency components of the synthetic data.
    #     """
    #     # High-frequency component
    #     b_hf, a_hf = butter(4, [self.hf_band[0] / self.nyquist, self.hf_band[1] / self.nyquist], btype='band') # type: ignore

    #     hf_component = filtfilt(b_hf, a_hf, smb.values)
    #     hf_component = xr.DataArray(hf_component, coords=smb.coords, dims=smb.dims)

    #     # Low-frequency component
    #     b_lf, a_lf = butter(4, [self.lf_band[0] / self.nyquist, self.lf_band[1] / self.nyquist], btype='band') # type: ignore
    #     lf_component = filtfilt(b_lf, a_lf, smb.values)
    #     lf_component = xr.DataArray(lf_component, coords=smb.coords, dims=smb.dims)

    #     return hf_component, lf_component