import cfgrib
import xarray as xr


class Preprocessor:
    def __init__(self, dataset_path: str):
        self.path = dataset_path
        self.ds_t2m, self.ds_tp = self._load_dataset()
        self.smb = self._calc_smb(self.ds_tp)
        self.annual_smb = self._calc_annual_smb()
        self.smb_norm = self._normalize_smb(self.smb)
        self.smb_mean = self.smb.mean(dim="time")
        self.smb_std = self.smb.std(dim="time")
        self.annual_smb_mean = self.annual_smb.mean(dim="year")
        self.annual_smb_std = self.annual_smb.std(dim="year")

    def _load_dataset(self) -> tuple[xr.Dataset, xr.Dataset]:
        # Load the dataset from the specified path
        path = self.path
        ds_t2m, ds_tp = cfgrib.open_datasets(path, decode_timedelta=True)
        return ds_t2m, ds_tp

    def get_data(self) -> tuple[xr.Dataset, xr.Dataset]:
        # Return the loaded datasets
        return (self.ds_t2m, self.ds_tp)

    def _calc_smb(self, ds_tp: xr.Dataset) -> xr.DataArray:
        # Calculate the Surface Mass Balance (SMB) from the total precipitation
        smb = ds_tp["tp"].mean(dim=["latitude", "longitude"]) / 1000  # m w.e.
        return smb

    def get_smb(self) -> xr.DataArray:
        # Return the calculated SMB
        return self.smb

    def _calc_annual_smb(self) -> xr.DataArray:
        # Calculate the annual SMB
        annual_smb = self.smb.groupby("time.year").sum(dim="time")
        return annual_smb

    def get_annual_smb(self) -> xr.DataArray:
        # Return the annual SMB
        return self.annual_smb

    def _normalize_smb(self, smb: xr.DataArray) -> xr.DataArray:
        """
        Standardize an xarray.DataArray: subtract mean, divide by std.
        Returns a standardized DataArray with the same metadata.
        """
        if not isinstance(smb, xr.DataArray):
            raise TypeError("Input must be an xarray.DataArray")
        if "time" not in smb.dims:
            raise ValueError(
                "DataArray must have a 'time' dimension for standardization"
            )

        # Calculate mean and standard deviation along the time dimension
        mean = smb.mean(dim="time")
        std = smb.std(dim="time")
        return (smb - mean) / std

    def get_normalized_smb(self):
        # Return the normalized SMB
        return self.smb_norm

    def get_stats(self) -> dict:
        # Return the statistics of the SMB
        return {
            "mean": self.smb_mean,
            "std": self.smb_std,
            "normalized_smb": self.smb_norm,
            "annual_mean": self.annual_smb_mean,
            "annual_std": self.annual_smb_std,
        }
