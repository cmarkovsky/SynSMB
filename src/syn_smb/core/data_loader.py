"""
data_loader.py
==============
Loads RACMO SMB data from NetCDF and prepares it for the synthesis pipeline.

Handles unit conversion, basic validation, and extraction of the basin-mean
time series. Designed to sit at the top of the pipeline, before Preprocessor
and GaussianTransform.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import xarray as xr


class SMBDataLoader:
    """
    Loads and validates RACMO SMB data from a NetCDF file.

    Handles unit conversion from kg m⁻² to m w.e. a⁻¹ and performs
    basic sanity checks on the loaded data. The loaded DataArray is
    stored as an attribute and returned by load() for use downstream.

    Parameters
    ----------
    path : str or Path
        Path to the NetCDF file.
    var : str
        Name of the SMB variable in the dataset. Default is 'smbgl',
        which is the basin-integrated variable in RACMO2.4p1.

    Attributes
    ----------
    data : xr.DataArray or None
        The loaded SMB time series. None until load() is called.
    path : Path
        Resolved path to the NetCDF file.
    var : str
        Variable name used when loading.

    Examples
    --------
    >>> loader = SMBDataLoader("path/to/racmo.nc")
    >>> smb = loader.load()
    >>> loader.summarize()
    """

    UNIT_CONVERSIONS = {
        "kg m-2": 1.0 / 1000.0,   # kg m⁻² → m w.e.
        "kg m-2 yr-1": 1.0 / 1000.0,
    }
    TARGET_UNIT = "m w.e. a$^{-1}$"

    def __init__(self, path: str | Path, var: str = "smbgl") -> None:
        self.path = Path(path)
        self.var = var
        self.data: xr.DataArray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> xr.DataArray:
        """
        Load the SMB variable from the NetCDF file.

        Performs unit conversion if needed and runs basic validation.
        Stores the result in self.data and returns it.

        Returns
        -------
        data : xr.DataArray
            SMB time series in m w.e. a⁻¹, with a 'time' dimension.

        Raises
        ------
        FileNotFoundError
            If the NetCDF file does not exist at self.path.
        ValueError
            If the file format is unsupported, the variable is missing,
            or the loaded data fails validation.
        """
        self._check_path()

        ds = xr.open_dataset(self.path)
        self._check_variable(ds)

        da = ds[self.var]
        da = self._convert_units(da)
        da = self._ensure_time_dimension(da)
        self._validate(da)

        self.data = da
        return self.data

    def summarize(self) -> None:
        """
        Print a summary of the loaded data.
        Useful for a quick sanity check before running the pipeline.
        """
        if self.data is None:
            print("No data loaded. Call load() first.")
            return

        da = self.data
        time = da["time"]

        print("SMBDataLoader summary")
        print(f"  File:       {self.path.name}")
        print(f"  Variable:   {self.var}")
        print(f"  Units:      {da.attrs.get('units', 'unknown')}")
        print(f"  n points:   {da.sizes['time']}")
        print(f"  Time range: {str(time.values[0])[:10]} → {str(time.values[-1])[:10]}")
        print(f"  Mean:       {float(da.mean()):.5f}")
        print(f"  Std:        {float(da.std()):.5f}")
        print(f"  Min:        {float(da.min()):.5f}")
        print(f"  Max:        {float(da.max()):.5f}")
        print(f"  Any NaN:    {bool(np.any(np.isnan(da.values)))}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_path(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"NetCDF file not found: {self.path}\n"
                "Check that the path is correct and the file exists."
            )
        if self.path.suffix not in (".nc", ".nc4"):
            raise ValueError(
                f"Unsupported file format: '{self.path.suffix}'. "
                "Expected a NetCDF file (.nc or .nc4)."
            )

    def _check_variable(self, ds: xr.Dataset) -> None:
        if self.var not in ds.data_vars:
            available = list(ds.data_vars)
            raise ValueError(
                f"Variable '{self.var}' not found in dataset.\n"
                f"Available variables: {available}\n"
                "Pass the correct variable name to SMBDataLoader(var=...)."
            )

    def _convert_units(self, da: xr.DataArray) -> xr.DataArray:
        units = da.attrs.get("units", "").strip()
        if units in self.UNIT_CONVERSIONS:
            factor = self.UNIT_CONVERSIONS[units]
            da = da * factor
            da.attrs["units"] = self.TARGET_UNIT
        elif units == self.TARGET_UNIT or "w.e" in units:
            pass  # already in target units
        else:
            # Don't convert unknown units — warn but continue
            print(
                f"Warning: unrecognised units '{units}'. "
                "No unit conversion applied. "
                "Expected 'kg m-2' or 'm w.e. a$^{{-1}}$'."
            )
        return da

    def _ensure_time_dimension(self, da: xr.DataArray) -> xr.DataArray:
        """
        Squeeze out any length-1 spatial dimensions (e.g. if the NetCDF
        contains a single grid point or basin-mean value with extra dims).
        """
        for dim in list(da.dims):
            if dim != "time" and da.sizes[dim] == 1:
                da = da.squeeze(dim)
        if "time" not in da.dims:
            raise ValueError(
                f"No 'time' dimension found after loading. "
                f"Available dims: {list(da.dims)}"
            )
        return da

    def _validate(self, da: xr.DataArray) -> None:
        if da.sizes["time"] < 12:
            raise ValueError(
                f"Loaded time series has only {da.sizes['time']} points. "
                "At least 12 months are required."
            )
        n_nan = int(np.sum(np.isnan(da.values)))
        if n_nan > 0:
            raise ValueError(
                f"Loaded data contains {n_nan} NaN values. "
                "Check the NetCDF file for missing data."
            )

    def __repr__(self) -> str:
        if self.data is not None:
            n = self.data.sizes["time"]
            return f"SMBDataLoader(path='{self.path.name}', var='{self.var}', n={n}, loaded=True)"
        return f"SMBDataLoader(path='{self.path.name}', var='{self.var}', loaded=False)"