import xarray as xr
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt

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
       

        # Fit the SARIMA model using auto_arima
        # Note: auto_arima automatically handles seasonal components if m is specified
        # Here, we assume m=12 for monthly data; adjust as necessary for your data
        model = auto_arima(data.values, start_p=1, start_q=1, test='adf', m=12,seasonal=True,trace=True)
        
        return model

    def forecast(self, steps: int = 120, dim: str = 'time'):
        """
        Forecast future values using the fitted SARIMA model.
        """
        # Implementation of forecasting goes here
        if not self.model_fit:
            raise ValueError("Model is not fitted yet. Call _fit_auto_arima() before forecasting.")
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("Steps must be a positive integer.")
          # Forecast
        forecast_vals = self.model_fit.predict(n_periods=steps)

        # Create forecast time index
        dt = pd.infer_freq(self.data[dim].to_pandas()) # type: ignore
        last_time = self.data[dim].values[-1]
        future_time = pd.date_range(start=last_time, periods=steps + 1, freq=dt)[1:] # type: ignore

        forecast_da = xr.DataArray(
        forecast_vals,
        coords={dim: future_time},
        dims=[dim],
        name=f"{self.data.name}_forecast" if self.data.name else "forecast"
        )

        forecast_da.attrs.update(self.data.attrs)


        return forecast_da
    


    def forecast_sarima_xr(
            self,
            da: xr.DataArray,
            steps: int,
            dim: str = "time",
            seasonal: bool = True,
            m: int = 12,
            test_size: int = 100,
            plot: bool = True,
            **auto_arima_kwargs
        ) -> xr.DataArray:
        """
        Fit a SARIMA model to a time series and forecast future SMB, with optional train/test split.

        Parameters:
        - da: xarray.DataArray (1D time series)
        - steps: number of forecast steps (usually = test_size or longer)
        - dim: name of the time dimension
        - seasonal: whether to use seasonal component
        - m: number of time steps in one season (12 for monthly annual seasonality)
        - test_size: number of steps to hold out for validation (0 = no test split)
        - plot: if True, plot forecast vs test
        - **auto_arima_kwargs: passed to pmdarima.auto_arima

        Returns:
        - forecast_da: forecasted xarray.DataArray for the next 'steps' time steps
        """
        if da.ndim > 1:
            raise ValueError("Only 1D time series are supported.")

        da = da.dropna(dim)
        y = da.values
        times = da[dim].values

        if test_size >= len(y):
            raise ValueError("test_size must be smaller than the length of the time series")

        # Split into training and test sets
        if test_size > 0:
            y_train = y[:-test_size]
            y_test = y[-test_size:]
            time_train = times[:-test_size]
            time_test = times[-test_size:]
        else:
            y_train = y
            y_test = None
            time_train = times

        # Fit SARIMA on training data
        model = auto_arima(
            y_train,
            seasonal=seasonal,
            m=m,
            suppress_warnings=True,
            stepwise=True,
            error_action="ignore",
            **auto_arima_kwargs
        )

        # Forecast into future
        forecast_vals = model.predict(n_periods=steps)

        # Build future time coordinate
        dt = pd.infer_freq(pd.to_datetime(time_train))
        last_train_time = time_train[-1]
        future_time = pd.date_range(start=last_train_time, periods=steps+1, freq=dt)[1:] # type: ignore

        forecast_da = xr.DataArray(
            forecast_vals,
            coords={dim: future_time},
            dims=[dim],
            name=f"{da.name}_forecast" if da.name else "forecast"
        )
        forecast_da.attrs.update(da.attrs)

        # --- Plotting ---
        if plot:
            plt.figure(figsize=(12, 4))
            plt.plot(time_train, y_train, label="Training data", color="orange")
            if y_test is not None:
                plt.plot(time_test, y_test, label="Test data (ground truth)", color="green") # type: ignore
            plt.plot(future_time, forecast_vals, label="SARIMA Forecast", color="blue", linestyle="--")
            plt.axvline(time_train[-1], color="gray", linestyle=":", lw=1)
            plt.title("SARIMA Forecast of SMB with Train/Test Split")
            plt.xlabel("Time")
            plt.ylabel(da.name if da.name else "SMB") # type: ignore
            plt.legend()
            plt.tight_layout()
            plt.show()

        return forecast_da

    
