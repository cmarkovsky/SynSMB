import numpy as np
import xarray as xr

class GaussianNoise:
    def __init__(self, filtered_smb: xr.DataArray):
        self.data = filtered_smb
        if not isinstance(filtered_smb, xr.DataArray):
            raise TypeError("Input must be an xarray.DataArray")
        self.mean = filtered_smb.mean()
        self.stddev = filtered_smb.std()

    def forecast(self, steps: int, seed: int = 42) -> xr.DataArray:
        """
        Generate Gaussian noise based on the mean and standard deviation of the filtered SMB data.
        Returns a DataArray with the same metadata as the input data.
        """
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("Steps must be a positive integer")
        
        np.random.seed(seed)  # For reproducibility
        noise = np.random.normal(loc=self.mean, scale=self.stddev, size=steps)
        time_index = xr.date_range(start=self.data.time[-1].values, periods=steps + 1, freq='M')[1:]
        
        return xr.DataArray(noise, coords=[time_index], dims=["time"], name="gaussian_noise")