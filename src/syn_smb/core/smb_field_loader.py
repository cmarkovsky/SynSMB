"""
smb_field_loader.py
===================
Loads the full 2D RACMO SMB spatial field masked to a single drainage
basin, returning an xr.DataArray of shape (time, rlat, rlon) with NaN
outside the basin boundary.

This is the entry point for the 2D synthetic SMB generation pipeline.
The 1D pipeline (SMBDataLoader) collapses the field to a basin mean;
SMBFieldLoader preserves the full spatial structure so EOFDecomposer
can decompose it into temporal modes.

Typical usage
-------------
    from syn_smb.core.smb_field_loader import SMBFieldLoader

    loader = SMBFieldLoader(
        racmo_path = "./data/RACMO2.4p1_ANT11.nc",
        shp_path   = "./data/IceBoundaries_Antarctica_V2.shp",
        basin_name = "PineIsland",
    )
    field = loader.load()     # xr.DataArray (time, rlat, rlon)
    mask  = loader.basin_mask # xr.DataArray (rlat, rlon), bool
    lat   = loader.lat        # xr.DataArray (rlat, rlon)
    lon   = loader.lon        # xr.DataArray (rlat, rlon)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask


# ── Variable name aliases ────────────────────────────────────────────
_SMB_ALIASES = ["smbgl", "smb", "SMB", "smbcorr", "precip", "snowfall"]
_LAT_ALIASES = ["lat", "latitude", "LAT", "nav_lat"]
_LON_ALIASES = ["lon", "longitude", "LON", "nav_lon"]


class SMBFieldLoader:
    """
    Load a full 2D RACMO SMB spatial field masked to a single basin.

    Parameters
    ----------
    racmo_path : str or Path
        Path to the full-domain RACMO NetCDF file.
    shp_path : str or Path
        Path to the basin shapefile (e.g. IceBoundaries_Antarctica_V2.shp).
    basin_name : str
        Value of `name_col` that identifies the target basin
        (e.g. 'PineIsland').
    smb_var : str or None
        SMB variable name. Auto-detected from _SMB_ALIASES if None.
    name_col : str
        Shapefile column containing basin names. Default 'NAME'.

    Attributes (available after load())
    ------------------------------------
    basin_mask : xr.DataArray (rlat, rlon), bool
        True for grid cells inside the basin boundary.
    lat : xr.DataArray (rlat, rlon)
        2-D latitude array in degrees north.
    lon : xr.DataArray (rlat, rlon)
        2-D longitude array in degrees east, normalised to [-180, 180].
    n_valid_cells : int
        Number of RACMO grid cells inside the basin.
    """

    def __init__(
        self,
        racmo_path: str | Path,
        shp_path:   str | Path,
        basin_name: str,
        smb_var:    str | None = None,
        name_col:   str = "NAME",
    ) -> None:
        self.racmo_path = Path(racmo_path)
        self.shp_path   = Path(shp_path)
        self.basin_name = basin_name
        self.smb_var    = smb_var
        self.name_col   = name_col

        # Populated on first load() call
        self._field:      xr.DataArray | None = None
        self._basin_mask: xr.DataArray | None = None
        self._lat:        xr.DataArray | None = None
        self._lon:        xr.DataArray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, crop: bool = True) -> xr.DataArray:
        """
        Load, mask, and return the basin SMB field.

        Parameters
        ----------
        crop : bool
            If True (default), crop the output to the bounding box of the
            basin mask. Reduces memory dramatically — for PIG at 11km this
            shrinks the grid from 591×726 (~430k cells) to ~30×40 (~1.2k
            cells). The cropped field is still a spatial DataArray with the
            same lat/lon structure; only the outer NaN border is removed.
            Set False only if you need the full RACMO grid extent.

        Returns
        -------
        field : xr.DataArray, shape (time, rlat_crop, rlon_crop)
            Monthly SMB in m w.e., NaN outside the basin.
        """
        if self._field is not None:
            return self._field

        ds  = self._open_dataset()
        smb = self._extract_smb(ds)
        lat, lon = self._extract_latlon(ds)

        self._lat = lat
        self._lon = lon

        mask = self._build_basin_mask(lat, lon)
        self._basin_mask = mask

        field = smb.where(mask)

        if crop:
            field, mask, lat, lon = self._crop_to_basin(field, mask, lat, lon)
            self._basin_mask = mask
            self._lat        = lat
            self._lon        = lon

        field.attrs.update({
            "basin_name":    self.basin_name,
            "units":         smb.attrs.get("units", "m w.e."),
            "long_name":     f"Basin SMB — {self.basin_name}",
            "n_valid_cells": int(mask.sum()),
            "cropped":       int(crop),
        })
        field.name = "smb"

        self._field = field
        print(
            f"SMBFieldLoader: '{self.basin_name}' — "
            f"{int(mask.sum())} valid cells, "
            f"shape={tuple(field.dims)}: {dict(field.sizes)}"
        )
        return field

    def _crop_to_basin(
        self,
        field: xr.DataArray,
        mask:  xr.DataArray,
        lat:   xr.DataArray,
        lon:   xr.DataArray,
        pad:   int = 2,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Crop field, mask, lat, lon to the bounding box of valid cells.

        Parameters
        ----------
        pad : int
            Number of extra cells added around the bounding box on each
            side. Keeps a small border of NaN for context in plots.
        """
        mask_np = mask.values
        spatial_dims = [d for d in field.dims if d != "time"]

        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)

        row_idx = np.where(rows)[0]
        col_idx = np.where(cols)[0]

        if len(row_idx) == 0 or len(col_idx) == 0:
            return field, mask, lat, lon   # no valid cells — nothing to crop

        r0 = max(0, int(row_idx[0])  - pad)
        r1 = min(field.sizes[spatial_dims[0]] - 1, int(row_idx[-1]) + pad)
        c0 = max(0, int(col_idx[0])  - pad)
        c1 = min(field.sizes[spatial_dims[1]] - 1, int(col_idx[-1]) + pad)

        slices = {
            spatial_dims[0]: slice(r0, r1 + 1),
            spatial_dims[1]: slice(c0, c1 + 1),
        }

        field_crop = field.isel(slices)
        mask_crop  = mask.isel(slices)
        lat_crop   = lat.isel(slices)
        lon_crop   = lon.isel(slices)

        ny_orig = field.sizes[spatial_dims[0]]
        nx_orig = field.sizes[spatial_dims[1]]
        ny_crop = field_crop.sizes[spatial_dims[0]]
        nx_crop = field_crop.sizes[spatial_dims[1]]
        print(
            f"  Cropped grid: {ny_orig}×{nx_orig} → {ny_crop}×{nx_crop} "
            f"({ny_orig * nx_orig:,} → {ny_crop * nx_crop:,} cells)"
        )
        return field_crop, mask_crop, lat_crop, lon_crop

    @property
    def basin_mask(self) -> xr.DataArray:
        """Boolean mask (rlat, rlon): True inside basin."""
        if self._basin_mask is None:
            self.load()
        return self._basin_mask  # type: ignore

    @property
    def lat(self) -> xr.DataArray:
        """2-D latitude array (rlat, rlon)."""
        if self._lat is None:
            self.load()
        return self._lat  # type: ignore

    @property
    def lon(self) -> xr.DataArray:
        """2-D longitude array (rlat, rlon), normalised to [-180, 180]."""
        if self._lon is None:
            self.load()
        return self._lon  # type: ignore

    @property
    def n_valid_cells(self) -> int:
        """Number of RACMO grid cells inside the basin."""
        return int(self.basin_mask.sum())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_dataset(self) -> xr.Dataset:
        if not self.racmo_path.exists():
            raise FileNotFoundError(
                f"RACMO file not found: {self.racmo_path}"
            )
        return xr.open_dataset(self.racmo_path)

    def _extract_smb(self, ds: xr.Dataset) -> xr.DataArray:
        """Find, unit-convert, and squeeze the SMB variable."""
        # Resolve variable name
        var = self.smb_var
        if var is None:
            for alias in _SMB_ALIASES:
                if alias in ds:
                    var = alias
                    break
        if var is None or var not in ds:
            raise ValueError(
                f"SMB variable not found. Searched: {_SMB_ALIASES}. "
                f"Available: {list(ds.data_vars)}. Pass smb_var= explicitly."
            )

        smb = ds[var]

        # Unit conversion: kg m-2 → m w.e.
        units = smb.attrs.get("units", "").strip()
        if "kg" in units and "m-2" in units:
            smb = smb / 1000.0
            smb.attrs["units"] = "m w.e."

        # Squeeze any size-1 dimensions (e.g. height=1)
        size1 = [d for d in smb.dims if smb.sizes[d] == 1]
        if size1:
            smb = smb.squeeze(size1, drop=True)

        # Confirm we have a time dimension
        if "time" not in smb.dims:
            raise ValueError(
                f"No 'time' dimension found in '{var}'. "
                f"Dims: {list(smb.dims)}"
            )

        return smb

    def _extract_latlon(
        self, ds: xr.Dataset
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Extract 2-D lat and lon arrays from the dataset."""
        lat_name = next(
            (a for a in _LAT_ALIASES if a in ds), None
        )
        lon_name = next(
            (a for a in _LON_ALIASES if a in ds), None
        )
        if lat_name is None or lon_name is None:
            raise ValueError(
                f"Could not find lat/lon in dataset. "
                f"Searched lat={_LAT_ALIASES}, lon={_LON_ALIASES}. "
                f"Available: {list(ds.coords) + list(ds.data_vars)}"
            )

        lat = ds[lat_name]
        lon = ds[lon_name]

        # Broadcast 1-D to 2-D if necessary
        if lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon.values, lat.values)
            spatial_dims = [
                d for d in ds.dims if d not in ("time", "bnds")
            ][-2:]
            lat = xr.DataArray(lat2d, dims=spatial_dims)
            lon = xr.DataArray(lon2d, dims=spatial_dims)

        # Normalise lon to [-180, 180]
        lon = xr.where(lon > 180, lon - 360, lon)

        return lat, lon

    def _build_basin_mask(
        self,
        lat: xr.DataArray,
        lon: xr.DataArray,
    ) -> xr.DataArray:
        """
        Create a boolean (rlat, rlon) mask using regionmask.

        Returns True for every RACMO grid cell whose centre falls
        inside the target basin polygon.
        """
        if not self.shp_path.exists():
            raise FileNotFoundError(
                f"Shapefile not found: {self.shp_path}"
            )

        # Load and reproject shapefile to WGS84
        gdf = gpd.read_file(self.shp_path)
        if self.name_col not in gdf.columns:
            raise ValueError(
                f"Column '{self.name_col}' not in shapefile. "
                f"Available: {list(gdf.columns)}"
            )

        # Filter to the target basin
        gdf_basin = gdf[gdf[self.name_col] == self.basin_name].copy()
        if gdf_basin.empty:
            available = sorted(gdf[self.name_col].dropna().unique())
            raise ValueError(
                f"Basin '{self.basin_name}' not found in "
                f"column '{self.name_col}'. "
                f"Available (first 20): {available[:20]}"
            )

        # Reproject to WGS84 for regionmask
        if gdf_basin.crs is None:
            gdf_basin = gdf_basin.set_crs("EPSG:4326")
        elif gdf_basin.crs.to_epsg() != 4326:
            gdf_basin = gdf_basin.to_crs("EPSG:4326")

        # Repair invalid geometries
        invalid = ~gdf_basin.geometry.is_valid
        if invalid.any():
            gdf_basin.loc[invalid, "geometry"] = (
                gdf_basin.loc[invalid, "geometry"].buffer(0)
            )

        # Dissolve duplicate rows (basin split across multiple polygons)
        if len(gdf_basin) > 1:
            geom_col = gdf_basin.geometry.name
            gdf_basin = (
                gdf_basin[[geom_col]]
                .dissolve()
                .reset_index(drop=True)
            )
            # Restore name column for regionmask
            gdf_basin[self.name_col] = self.basin_name

        # Build regionmask regions and create 3D mask
        regions = regionmask.from_geopandas(
            gdf_basin,
            names=self.name_col,
            overlap=True,
        )
        mask_3d = regions.mask_3D(lon.values, lat.values)
        # mask_3d shape: (n_regions=1, rlat, rlon)
        mask_2d = mask_3d.isel(region=0)

        # Convert to a plain boolean DataArray aligned with smb's spatial dims
        spatial_dims = list(lat.dims)
        basin_mask = xr.DataArray(
            mask_2d.values.astype(bool),
            dims=spatial_dims,
            attrs={"basin_name": self.basin_name},
        )

        n_cells = int(basin_mask.sum())
        if n_cells == 0:
            warnings.warn(
                f"Basin '{self.basin_name}' has zero RACMO grid cells. "
                f"Check the shapefile covers the RACMO domain and that "
                f"the basin name matches exactly."
            )

        return basin_mask

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = self._field is not None
        cells  = f", n_cells={self.n_valid_cells}" if loaded else ""
        return (
            f"SMBFieldLoader("
            f"basin='{self.basin_name}', "
            f"loaded={loaded}{cells})"
        )