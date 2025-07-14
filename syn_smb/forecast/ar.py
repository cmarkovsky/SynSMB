import xarray as xr
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

class ARModel:
    def __init__(self, data: xr.DataArray, order: int = 12, dim: str = "time", random_seed: int = 42):
        """
        Initialize the ARModel with the provided data and order.
        """
        self.data = data
        self.order = order
        self.dim = dim
        self.model_fit = self._fit()

    def forecast(self, steps: int = 120):
        """
        Generate forecasts from the fitted AR model.
        """
        if not self.fitted:
            raise ValueError("Model is not fitted yet. Call _fit() before forecasting.")

        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("Steps must be a positive integer.")
        

        forecast_values = self.model_fit.predict(start=len(self.data), end=len(self.data) + steps - 1)
        # Time coordinate generation
        dt = pd.infer_freq(self.data[self.dim].to_pandas()) # type: ignore
        if dt is None:
            raise ValueError("Could not infer frequency from the time coordinate. Please ensure it is regular.")
        last_time = self.data[self.dim].values[-1]
        future_time = pd.date_range(start=last_time, periods=steps+1, freq=dt)[1:]

        forecast_da = xr.DataArray(
            forecast_values,
            coords={self.dim: future_time},
            dims=[self.dim],
            name=f"{self.data.name}_forecast" if self.data.name else "forecast"
        )
        forecast_da.attrs.update(self.data.attrs)

        return forecast_da


    def _fit(self):
        """
        Fit the AR model to the data.
        """
        if not isinstance(self.data, xr.DataArray):
            raise TypeError("Input must be an xarray.DataArray")

        # Ensure the data has a time dimension
        if self.dim not in self.data.dims:
            raise ValueError(f"DataArray must have a '{self.dim}' dimension")
        
        if self.data.ndim != 1:
            raise ValueError("DataArray must be one-dimensional for AR model fitting")
        
        # Drop NaN values along the specified dimension
        data = self.data.dropna(self.dim)

        # Convert the DataArray to a numpy array for AR model fitting
        values = data.values

        if len(values) < self.order + 1:
            raise ValueError(f"Not enough data points to fit AR model of order {self.order}. Need at least {self.order + 1} points.")
        
        # Fit the AR model using statsmodels
        model = AutoReg(values, lags=self.order, seasonal=True, period=12)
        model_fit = model.fit()
        self.fitted = True
        
        return model_fit

