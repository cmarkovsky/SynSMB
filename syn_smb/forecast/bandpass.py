import xarray as xr
from scipy.signal import butter, filtfilt
import numpy as np


class BandpassFilter:
    def __init__(
        self,
        filt_center: int,
        alpha: float = np.sqrt(2),
        sample_rate: float = 1.0,
        dim: str = "time",
    ):
        """
        Initialize the BandpassFilter with the given parameters.
        """
        self.filt_center = filt_center
        self.alpha = alpha
        self.sample_rate = sample_rate
        self.dim = dim
        self.low_freq, self.high_freq = self._calc_freqs()

    def _calc_freqs(self):
        # Calculate the frequencies for the bandpass filter

        low_freq = 1 / (self.filt_center * 12 * self.alpha)  # Low frequency
        high_freq = 1 / (self.filt_center * 12 / self.alpha)  # High frequency
        return low_freq, high_freq

    def filter(self, data: xr.DataArray, order: int = 4):
        """
        Apply the bandpass filter to the provided xarray DataArray.
        """
        if not isinstance(data, xr.DataArray):
            raise TypeError("Input must be an xarray.DataArray")

        # Ensure the data has the correct dimension
        if self.dim not in data.dims:
            raise ValueError(f"DataArray must have a '{self.dim}' dimension")

        # Apply bandpass filter to the data
        nyquist = 0.5 * self.sample_rate
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist

        b, a = butter(order, [low, high], btype="bandpass")  # type: ignore

        def _apply_filter(values):
            return filtfilt(b, a, values)

        filtered = xr.apply_ufunc(
            _apply_filter,
            data,
            input_core_dims=[[self.dim]],
            output_core_dims=[[self.dim]],
            vectorize=True,
            output_dtypes=[data.dtype],
        )

        filtered.attrs.update(data.attrs)  # Preserve original attributes
        return filtered

    def get_filter_params(self):
        return {
            "filt_center": self.filt_center,
            "alpha": self.alpha,
            "low_freq": self.low_freq,
            "high_freq": self.high_freq,
        }
