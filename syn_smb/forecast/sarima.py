import xarray as xr
from pmdarima import auto_arima
class SARIMA:
    """
    A class to handle the Seasonal Autoregressive Integrated Moving Average (SARIMA) model.
    """
    
    def __init__(self, data: xr.DataArray):
        self.data = data
        # self.order = order
        self.model_fit = self._fit_auto_arima()

    def _fit_auto_arima(self):
        """
        Fit the SARIMA model to the data using auto_arima.
        """
        if not isinstance(self.data, xr.DataArray):
            raise TypeError("Input must be an xarray.DataArray")
        if self.data.ndim != 1:
            raise ValueError("DataArray must be one-dimensional for SARIMA fitting")
        # Drop NaN values along the time dimension
        data = self.data.dropna('time')
        
        # Convert the DataArray to a pandas Series for auto_arima
        data_series = data.to_series()
        # print(data_series.head())  # Debugging line to check the data
        # Fit the SARIMA model using auto_arima
        # Note: auto_arima automatically handles seasonal components if m is specified
        # Here, we assume m=12 for monthly data; adjust as necessary for your data

    
        model = auto_arima(data_series, start_p=1, start_q=1, test='adf', m=12,seasonal=True,trace=True)
        
        return model

    def forecast(self, steps: int):
        """
        Forecast future values using the fitted SARIMA model.
        """
        # Implementation of forecasting goes here
        pass