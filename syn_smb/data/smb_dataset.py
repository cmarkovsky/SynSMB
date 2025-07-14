import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from syn_smb import Preprocessor, BandpassFilter, Plotter, ARModel, SARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

class SMBDataSet(Preprocessor):
    """A class to handle the Surface Mass Balance (SMB) dataset. 
    Inherits from Preprocessor to utilize its methods for loading and processing data."""

    def __init__(self, data_path: str):
        self.path = data_path
        self._set_region(self.path)
        super().__init__(data_path)
        self._filters = {}
        self._filtered_smbs = {}

    def _set_region(self, path: str) -> None:
        """Set the region for the dataset."""
        self.region = path.split('/')[-1].split('_')[1].split('.')[0].upper()
    
    def get_region(self):
        """Return the region of the dataset."""
        return self.region
    
    def filter_smb(self, n_years: int):
        """Apply a bandpass filter to the SMB data."""
        bandpass = BandpassFilter(n_years=n_years)
        filtered_smb = bandpass.filter(self.smb)
        self._filters[n_years] = bandpass
        self._filtered_smbs[n_years] = filtered_smb
        return filtered_smb

    def get_filtered_smb(self, n_years: int = -1):
        """Retrieve the filtered SMB data for a specific number of years."""
        if n_years == -1:
            return self._filtered_smbs
        elif n_years in self._filters:
            return self._filtered_smbs[n_years]
        else:
            raise ValueError(f"No filtered SMB data available for {n_years} years.")

    def get_filters(self, n_years: int = -1):
        """Return the available filters."""
        if n_years == -1:
            return self._filters
        elif n_years in self._filters:
            return self._filters[n_years]
        else:
            raise ValueError(f"No filter available for {n_years} years.")
    
    def forecast_smb(self, filt_center: int = 1, n_years: int = 10, plot: bool = True) -> xr.DataArray:
        """Forecast the SMB using an AR model."""
        if filt_center not in self._filtered_smbs:
            self.filter_smb(filt_center)
        
        filtered_smb = self._filtered_smbs[filt_center]
        filtered_smb_norm = self._normalize_smb(filtered_smb)
        # Assuming ARModel is implemented to handle the forecasting
        # ar_model = ARModel(filtered_smb_norm, order=12, dim='time')
        ar_model = ARModel(filtered_smb, order=12, dim='time')

        forecasted_smb = ar_model.forecast(steps=n_years * 12)
        # forecasted_smb = self._unnormalize_smb(forecasted_smb)
        if plot:
            self.plot_forecasted_smb(filt_center, filtered_smb + self.smb_mean, forecasted_smb + self.smb_mean)
        return forecasted_smb
    
    def plot_forecasted_smb(self, filt_center: int, filtered_smb: xr.DataArray, forecasted_smb: xr.DataArray):
        """Plot the actual SMB data followed by the forecasted SMB data."""
        plt.figure(figsize=(10, 5))
        # Plot actual SMB data
        sns.lineplot(x=self.smb['time'], y=self.smb, label='Actual SMB', color = 'tab:blue')
        sns.lineplot(x=filtered_smb['time'], y=filtered_smb, label=f'Filtered SMB - {filt_center} Year(s)', color = 'tab:orange')
        # Plot forecasted SMB data
        sns.lineplot(x=forecasted_smb['time'], y=forecasted_smb, label=f'Forecasted SMB Trend - {filt_center} Year(s)', color = 'black', linestyle='--')
        plt.title(f'Actual vs Forecasted Surface Mass Balance (SMB) for {self.region}')
        plt.xlabel('Time')
        plt.ylabel('SMB (m w.e.)')
        plt.legend()
        plt.show()

    def plot_smb(self):
        """Plot the SMB and annual SMB data."""
        plotter = Plotter(self.smb_norm, self.annual_smb, self.region)
        plotter.plot_smb()
    
    def plot_filtered_smb(self, n_years: int):
        """Plot the filtered SMB data."""
        if n_years not in self._filtered_smbs:
            raise ValueError(f"No filtered SMB data available for {n_years} years.")
        filtered_smb = self._filtered_smbs[n_years]
        plotter = Plotter(self.smb, self.annual_smb, self.region)
        plotter.plot_filtered_smb(filtered_smb, n_years=n_years)
        plt.title(f'{self.region} Filtered Surface Mass Balance (SMB) - {n_years} Year(s)')
        plt.show()
    
    def plot_pacf(self):
        """Plot the Partial Autocorrelation Function (PACF) of the SMB data."""
        plotter = Plotter(self.smb, self.annual_smb, self.region)
        plotter.plot_pacf(self.smb)
        plt.show()

    def plot_acf(self):
        """Plot the Autocorrelation Function (ACF) of the SMB data."""
        plotter = Plotter(self.smb, self.annual_smb, self.region)
        plotter.plot_acf(self.smb)
        plt.show()
    
    def season_decompose(self, model='additive'):
        """Perform seasonal decomposition of the SMB data."""
        if self.smb is None:
            raise ValueError("SMB data is not loaded. Please load the data first.")
        
        decomposed = seasonal_decompose(self.smb, model=model, period=12)
        decomposed.plot()
        plt.title(f'Seasonal Decomposition of SMB for {self.region}')
        plt.show()
        return decomposed
    
    def adf_test(self):
        """Perform Augmented Dickey-Fuller test on the SMB data."""
        if self.smb is None:
            raise ValueError("SMB data is not loaded. Please load the data first.")
        
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(self.smb.values, autolag='AIC')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items(): # type: ignore
            print(f'  {key}: {value}')
        
        return result
    
    def forecast_sarima(self, steps: int = 120):
        """Forecast using the SARIMA model."""
        if self.smb is None:
            raise ValueError("SMB data is not loaded. Please load the data first.")
        
        sarima_model = SARIMA(self.smb)
        print(sarima_model.model_fit.summary())
        
        return
