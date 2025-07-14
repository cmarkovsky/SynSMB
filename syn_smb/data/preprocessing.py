import cfgrib
import xarray as xr


class Preprocessor:
    def __init__(self, dataset_path: str):
        self.path = dataset_path
        self.raw_t2m, self.raw_tp = self._load_dataset()
        self.smb = self._calc_smb(self.raw_tp)
        self.annual_smb = self._calc_annual_smb()
        self.smb_norm = self._normalize_smb(self.smb)
        self.smb_mean = self.smb.mean(dim="time")
        self.smb_std = self.smb.std(dim="time")
        self.annual_smb_norm = self._normalize_smb(self.annual_smb, dim="year")
        self.annual_smb_mean = self.annual_smb.mean(dim="year")
        self.annual_smb_std = self.annual_smb.std(dim="year")

    def _load_dataset(self) -> tuple[xr.Dataset, xr.Dataset]:
        # Load the dataset from the specified path
        path = self.path
        ds_t2m, ds_tp = cfgrib.open_datasets(path, decode_timedelta=True)
        return ds_t2m, ds_tp

    def get_raw_data(self) -> tuple[xr.Dataset, xr.Dataset]:
        # Return the loaded datasets
        return (self.raw_t2m, self.raw_tp)

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

    def _normalize_smb(self, smb: xr.DataArray, dim: str = "time") -> xr.DataArray:
        """
        Standardize an xarray.DataArray: subtract mean, divide by std.
        Returns a standardized DataArray with the same metadata.
        """
        if not isinstance(smb, xr.DataArray):
            raise TypeError("Input must be an xarray.DataArray")
        if dim not in smb.dims:
            raise ValueError(
                "DataArray must have a 'time' dimension for standardization"
            )

        # Calculate mean and standard deviation along the time dimension
        mean = smb.mean(dim)
        std = smb.std(dim)
        return (smb - mean) / std

    def get_normalized_smb(self) -> xr.DataArray:
        # Return the normalized SMB
        return self.smb_norm

    def get_stats(self) -> dict:
        # Return the statistics of the SMB
        return {
            "smb_mean": self.smb_mean,
            "smb_std": self.smb_std,
            "smb_norm": self.smb_norm,
            "annual_mean": self.annual_smb_mean,
            "annual_std": self.annual_smb_std,
            "annual_norm": self.annual_smb_norm
        }
