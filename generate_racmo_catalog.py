"""
generate_racmo_catalog.py
=========================
Extracts basin-mean SMB time series from a full-Antarctica RACMO NetCDF
file for each sector defined in a shapefile, saves one NetCDF per sector,
and builds a RACMOCatalog ready for multi_basin_run().

Usage
-----
# Basic
python generate_racmo_catalog.py \\
    --racmo  ./data/RACMO2.4p1_ANT27_SMB_monthly.nc \\
    --shp    ./data/sectors/antarctica_sectors.shp \\
    --outdir ./data/sectors_smb/

# With options
python generate_racmo_catalog.py \\
    --racmo  ./data/RACMO2.4p1_ANT27_SMB_monthly.nc \\
    --shp    ./data/sectors/antarctica_sectors.shp \\
    --outdir ./data/sectors_smb/ \\
    --var    smb \\
    --name   Subregion \\
    --sectors PIG Thwaites Getz \\
    --low-res \\
    --coarsen 4 \\
    --plot \\
    --save-plot catalog_overview.png

Dependencies
------------
    poetry add geopandas regionmask cartopy
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
# generate_racmo_catalog.py
from config import RACMO_PATH, SHP_PATH, SECTORS_DIR, BASIN_NAMES


try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    warnings.warn(
        "cartopy not found — map plot will use a simplified projection. "
        "Install with: poetry add cartopy"
    )

from syn_smb import RACMOCatalog

# Colorblind-safe palette (Wong 2011), extended for many sectors
_SECTOR_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
    "#8B4513", "#2E8B57", "#DC143C", "#4B0082",
]

# Common RACMO SMB variable name aliases (searched in order)
_SMB_ALIASES = ["smbgl", "smb", "SMB", "smbcorr", "precip", "pr", "snowfall"]

# Common lat/lon variable name aliases
_LAT_ALIASES = ["lat", "latitude", "LAT", "nav_lat", "rlat"]
_LON_ALIASES = ["lon", "longitude", "LON", "nav_lon", "rlon"]

# Common area variable name aliases
_AREA_ALIASES = ["areacell", "area", "cell_area", "s_grid"]


# ======================================================================
# Step 1: Load and inspect RACMO data
# ======================================================================

def load_racmo(
    path: str | Path,
    var: str | None = None,
    low_res: bool = False,
    coarsen_factor: int = 4,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray | None]:
    """
    Load RACMO SMB field, lat, lon, and optional cell area.

    Parameters
    ----------
    path : str or Path
    var : str or None
        SMB variable name. If None, searched from _SMB_ALIASES.
    low_res : bool
        Coarsen the grid by coarsen_factor for faster processing and plotting.
    coarsen_factor : int
        Coarsening factor along each spatial dimension.

    Returns
    -------
    smb : xr.DataArray
        SMB field, shape (time, y, x), units m w.e. a⁻¹.
    lat : xr.DataArray
        2D latitude array, degrees north.
    lon : xr.DataArray
        2D longitude array, degrees east, in range [-180, 180].
    area : xr.DataArray or None
        Cell area in m², or None if not found in the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RACMO file not found: {path}")

    print(f"Loading RACMO: {path.name}")
    ds = xr.open_dataset(path)  # load eagerly — dask not required

    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.dims)}")

    # ── Find SMB variable ──
    if var is not None:
        if var not in ds:
            raise ValueError(
                f"Variable '{var}' not in dataset. "
                f"Available: {list(ds.data_vars)}"
            )
        smb_var = var
    else:
        smb_var = None
        for alias in _SMB_ALIASES:
            if alias in ds:
                smb_var = alias
                break
        if smb_var is None:
            raise ValueError(
                f"Could not find SMB variable. Searched: {_SMB_ALIASES}. "
                f"Available: {list(ds.data_vars)}. "
                f"Pass --var explicitly."
            )

    print(f"  SMB variable: '{smb_var}'")
    smb = ds[smb_var]

    # ── Unit conversion ──
    units = smb.attrs.get("units", "").strip()
    print(f"  Units: '{units}'")

    if "kg" in units.lower() and "m-2" in units.lower():
        # Convert kg m-2 (per month or per year) to m w.e.
        # RACMO monthly files: units are kg m-2 per month or per timestep
        smb = smb / 1000.0
        smb.attrs["units"] = "m w.e."
        print(f"  Converted kg m-2 → m w.e.")
    elif "m w.e" in units.lower() or "m_w.e" in units.lower():
        print(f"  Units already in m w.e.")
    else:
        warnings.warn(
            f"Unrecognised units '{units}'. No conversion applied. "
            f"Verify units manually."
        )

    # ── Find lat/lon ──
    lat, lon = _find_latlon(ds)
    print(f"  Lat shape: {lat.shape}, range: [{float(lat.min()):.1f}, {float(lat.max()):.1f}]")
    print(f"  Lon shape: {lon.shape}, range: [{float(lon.min()):.1f}, {float(lon.max()):.1f}]")

    # ── Normalise lon to [-180, 180] ──
    if float(lon.max()) > 180:
        lon = xr.where(lon > 180, lon - 360, lon)
        lon.attrs = ds[_find_varname(ds, _LON_ALIASES)].attrs
        print(f"  Normalised lon to [-180, 180]")

    # ── Find cell area (optional) ──
    area = _find_area(ds)
    if area is not None:
        print(f"  Cell area variable found: area-weighted means will be used.")
    else:
        print(f"  No cell area variable found: simple spatial means will be used.")

    # ── Squeeze size-1 dimensions (e.g. 'height') unconditionally ──
    # These are broadcast dimensions that cause problems in spatial operations.
    size1_dims = [d for d in smb.dims if d != "time" and smb.sizes[d] == 1]
    if size1_dims:
        print(f"  Squeezing size-1 dimensions: {size1_dims}")
        smb = smb.squeeze(size1_dims, drop=True)

    # ── Optional coarsening ──
    if low_res:
        # Squeeze out any size-1 dimensions (e.g. 'height') before coarsening
        # so they don't cause a reshape failure
        size1_dims = [d for d in smb.dims if smb.sizes[d] == 1]
        if size1_dims:
            print(f"  Squeezing size-1 dimensions: {size1_dims}")
            smb = smb.squeeze(size1_dims, drop=True)

        spatial_dims_coarsen = [
            d for d in smb.dims
            if d != "time" and smb.sizes[d] >= coarsen_factor
        ]
        if not spatial_dims_coarsen:
            print(f"  Warning: no spatial dims large enough to coarsen "
                  f"by {coarsen_factor} — skipping coarsening.")
        else:
            print(f"  Coarsening {spatial_dims_coarsen} by {coarsen_factor}x...")
            coarsen_dict = {d: coarsen_factor for d in spatial_dims_coarsen}
            smb  = smb.coarsen(coarsen_dict,  boundary="trim").mean()
            lat  = lat.coarsen(coarsen_dict,  boundary="trim").mean()
            lon  = lon.coarsen(coarsen_dict,  boundary="trim").mean()
            if area is not None:
                area = area.coarsen(coarsen_dict, boundary="trim").sum()
            print(f"  Coarsened SMB shape: {smb.shape}")

    return smb, lat, lon, area


def _find_varname(ds: xr.Dataset, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in ds:
            return a
    return None


def _find_latlon(
    ds: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    lat_name = _find_varname(ds, _LAT_ALIASES)
    lon_name = _find_varname(ds, _LON_ALIASES)

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not find lat/lon variables. Searched {_LAT_ALIASES} / "
            f"{_LON_ALIASES}. Available: {list(ds.data_vars) + list(ds.coords)}"
        )

    lat = ds[lat_name]
    lon = ds[lon_name]

    # Broadcast to 2D if 1D
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon.values, lat.values)
        spatial_dims = [d for d in ds.dims if d != "time"]
        if len(spatial_dims) >= 2:
            lat = xr.DataArray(lat2d, dims=spatial_dims[-2:])
            lon = xr.DataArray(lon2d, dims=spatial_dims[-2:])

    return lat, lon


def _find_area(ds: xr.Dataset) -> xr.DataArray | None:
    name = _find_varname(ds, _AREA_ALIASES)
    return ds[name] if name else None


# ======================================================================
# Step 2: Load and reproject shapefile
# ======================================================================

def load_sectors(
    shp_path: str | Path,
    name_col:      str | None = None,
    subregion_col: str | None = None,
    region_col:    str | None = None,
    sectors: list[str] | None = None,
) -> tuple[gpd.GeoDataFrame, str, str | None, str | None]:
    """
    Load sector shapefile and reproject to WGS84 (EPSG:4326).

    Three-level categorisation
    --------------------------
    name_col      — unique glacier/basin identifier (e.g. NAME column).
                    Becomes the registry key and output filename stem.
    subregion_col — intermediate grouping code (e.g. 'Ipp-J', 'Ap-B').
                    Typically the IMBIE/MEaSUREs subregion identifier.
    region_col    — broad geographical region (e.g. 'West Antarctica').

    Parameters
    ----------
    shp_path : str or Path
    name_col : str or None
        Auto-detected: prefers exact 'NAME' / 'name', then other unique-id
        patterns.
    subregion_col : str or None
        Auto-detected: prefers exact 'subregion', then 'Subregion'.
    region_col : str or None
        Auto-detected: prefers exact 'Regions' / 'Region', then others.
    sectors : list of str or None
        Subset of NAME values to process.

    Returns
    -------
    gdf, name_col, subregion_col, region_col
    """
    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    print(f"\nLoading shapefile: {shp_path.name}")
    gdf = gpd.read_file(shp_path)
    print(f"  CRS: {gdf.crs}")
    print(f"  Columns: {list(gdf.columns)}")
    print(f"  N sectors: {len(gdf)}")

    # ── Find name column (unique identifier — level 1) ──
    if name_col is None:
        cols = list(gdf.columns)
        # Priority 1: exact 'NAME' or 'name'
        exact = [c for c in cols if c.upper() == "NAME"]
        # Priority 2: 'id', 'basin_id', etc. (but NOT region/subregion)
        other = [c for c in cols
                 if any(kw in c.lower() for kw in ["name", "id", "label"])
                 and "region" not in c.lower()
                 and "sub" not in c.lower()
                 and c not in exact]
        candidates = exact + other
        if candidates:
            name_col = candidates[0]
            print(f"  Auto-detected NAME column (unique id): '{name_col}'")
        else:
            name_col = next(c for c in cols if c != "geometry")
            print(f"  Falling back to first column for NAME: '{name_col}'")
    elif name_col not in gdf.columns:
        raise ValueError(f"name_col '{name_col}' not in shapefile. "
                         f"Available: {list(gdf.columns)}")

    print(f"  Unique names ({len(gdf[name_col].dropna().unique())}): "
          f"{sorted(gdf[name_col].dropna().unique())[:5]} ...")

    # ── Find subregion column (intermediate grouping — level 2) ──
    if subregion_col is None:
        cols = list(gdf.columns)
        # Priority 1: exact 'subregion' / 'Subregion'
        exact_sub = [c for c in cols if c.lower() == "subregion" and c != name_col]
        # Priority 2: contains 'sub' + 'region'
        sub_reg = [c for c in cols
                   if "sub" in c.lower() and "region" in c.lower()
                   and c not in exact_sub and c != name_col]
        sub_candidates = exact_sub + sub_reg
        if sub_candidates:
            subregion_col = sub_candidates[0]
            print(f"  Auto-detected subregion column (level 2): '{subregion_col}'")
        else:
            subregion_col = None
            print(f"  No subregion column found — subregion will be None.")
    elif subregion_col not in gdf.columns:
        warnings.warn(f"subregion_col '{subregion_col}' not in shapefile. Ignoring.")
        subregion_col = None

    # ── Find region column (broadest grouping — level 3) ──
    if region_col is None:
        cols = list(gdf.columns)
        # Priority 1: exact 'Regions', 'Region', 'region' (not subregion, not name)
        exact_reg = [c for c in cols
                     if c.lower() in ("region", "regions")
                     and c != name_col and c != subregion_col]
        # Priority 2: contains 'region' but not 'sub'
        reg_no_sub = [c for c in cols
                      if "region" in c.lower() and "sub" not in c.lower()
                      and c not in exact_reg and c != name_col]
        reg_candidates = exact_reg + reg_no_sub
        if reg_candidates:
            region_col = reg_candidates[0]
            print(f"  Auto-detected region column (level 3): '{region_col}'")
        else:
            region_col = None
            print(f"  No region column found.")
    elif region_col not in gdf.columns:
        warnings.warn(f"region_col '{region_col}' not in shapefile. Ignoring.")
        region_col = None

    if region_col is not None:
        gdf["_clean_region"] = (
            gdf[region_col].str.strip()
            .str.replace(" ", "_").str.replace("/", "-")
        )
        unique_r = sorted(r for r in gdf["_clean_region"].unique() if r is not None)
        print(f"  Unique regions: {unique_r}")
    else:
        gdf["_clean_region"] = None

    if subregion_col is not None:
        gdf["_clean_subregion"] = (
            gdf[subregion_col].str.strip()
            .str.replace(" ", "_").str.replace("/", "-")
        )
        unique_s = sorted(s for s in gdf["_clean_subregion"].unique() if s is not None)
        print(f"  Unique subregions: {unique_s}")
    else:
        gdf["_clean_subregion"] = None

    # ── Filter to requested sectors ──
    if sectors is not None:
        missing = [s for s in sectors if s not in gdf[name_col].values]
        if missing:
            warnings.warn(
                f"Requested sectors not found in shapefile: {missing}. "
                f"Available: {list(gdf[name_col])}"
            )
        gdf = gdf[gdf[name_col].isin(sectors)].copy()
        print(f"  Filtered to {len(gdf)} requested sectors.")

    # ── Reproject to WGS84 ──
    if gdf.crs is None:
        warnings.warn("Shapefile has no CRS. Assuming EPSG:4326 (WGS84).")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        print(f"  Reprojecting from {gdf.crs.to_epsg()} → EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")

    # Clean up sector names for use as filenames
    gdf["_clean_name"] = (
        gdf[name_col]
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "-")
        .str.replace("\\", "-")
    )

    # Drop rows where the subregion name is missing or empty —
    # regionmask raises ValueError if names contain NaN
    n_before = len(gdf)
    gdf = gdf.dropna(subset=[name_col]).copy()
    gdf = gdf[gdf["_clean_name"].str.len() > 0].copy()
    n_dropped = n_before - len(gdf)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with missing/empty subregion names.")
    if len(gdf) == 0:
        raise ValueError(
            f"No valid sectors remain after dropping missing names in "
            f"column '{name_col}'. Check the shapefile."
        )
    print(f"  {len(gdf)} valid sectors after cleaning.")

    return gdf, name_col, subregion_col, region_col


# ======================================================================
# Step 3: Create masks and extract basin means
# ======================================================================

def extract_basin_means(
    smb: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    area: xr.DataArray | None,
    gdf: gpd.GeoDataFrame,
    name_col: str,
) -> dict[str, xr.DataArray]:
    """
    Create a mask for each sector and compute the area-weighted mean SMB.

    Parameters
    ----------
    smb : xr.DataArray, shape (time, y, x)
    lat : xr.DataArray, shape (y, x)
    lon : xr.DataArray, shape (y, x)
    area : xr.DataArray or None
    gdf : GeoDataFrame
        Sectors in WGS84.
    name_col : str

    Returns
    -------
    basin_smb : dict[str, xr.DataArray]
        {sector_name: (time,) DataArray of basin-mean SMB}
    """
    print("\nCreating sector masks and extracting basin means...")

    # ── Validate and repair geometries before passing to regionmask ──
    print("  Validating geometries...")
    gdf = gdf.copy()
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        print(f"  Repairing {invalid.sum()} invalid geometries with buffer(0)...")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    empty = gdf.geometry.is_empty
    if empty.any():
        print(f"  Warning: {empty.sum()} empty geometries — dropping them.")
        gdf = gdf[~empty].copy()

    if len(gdf) == 0:
        raise ValueError(
            "No valid geometries remaining after validation. "
            "Check that the shapefile covers the RACMO domain."
        )

    # ── Dissolve duplicate subregion names ──
    # Must happen AFTER geometry repair, not in load_sectors.
    # Some shapefiles store a single basin as multiple polygon rows.
    # dissolve() merges them into one multipolygon per unique name.
    dupes = gdf[name_col].duplicated().sum()
    if dupes > 0:
        print(f"  Dissolving {dupes} duplicate subregion rows "
              f"into multipolygons...")
        # Two-step: dissolve geometry, then merge metadata back
        geom_col = gdf.geometry.name
        dissolved = (
            gdf[[name_col, geom_col]]
            .dissolve(by=name_col)
            .reset_index()
        )
        meta = (
            gdf.drop_duplicates(subset=[name_col], keep="first")
               .drop(columns=[geom_col])
        )
        gdf = dissolved.merge(meta, on=name_col, how="left")
        gdf = gpd.GeoDataFrame(gdf, geometry=geom_col, crs=dissolved.crs)
        print(f"  {len(gdf)} unique subregions after dissolve.")

    # ── Build regionmask regions ──
    # Note: overlap parameter omitted for cross-version compatibility
    regions = regionmask.from_geopandas(
        gdf,
        names=name_col,
        abbrevs="_clean_name",
        overlap=True,   # explicit: mask_3D handles overlaps correctly
    )
    print(f"  regionmask regions: {regions}")

    # ── Create 3D boolean mask (one layer per region) ──
    # mask_3D handles overlapping sector boundaries cleanly — each layer
    # is a boolean array (True = cell belongs to that region).
    # This replaces the 2D integer mask which raises ValueError for overlaps.
    lat_vals = lat.values
    lon_vals = lon.values

    try:
        mask_3d = regions.mask_3D(lon_vals, lat_vals)
    except Exception as e:
        raise RuntimeError(
            f"regionmask failed to create a 3D mask: {e}\n"
            f"Check that lat/lon arrays cover the shapefile domain and "
            f"that all geometries are valid."
        ) from e

    print(f"  3D mask shape: {mask_3d.shape}  "
          f"(n_regions={mask_3d.sizes['region']}, "
          f"spatial={[s for k, s in mask_3d.sizes.items() if k != 'region']})")

    # Identify spatial dimensions for .mean() and .where()
    spatial_dims = [d for d in smb.dims if d != "time"]
    if len(spatial_dims) == 0 or len(spatial_dims) == len(smb.dims):
        time_dim = None
        for d in smb.dims:
            coord = smb[d] if d in smb.coords else None
            if coord is not None and np.issubdtype(coord.dtype, np.datetime64):
                time_dim = d
                break
        if time_dim is None:
            time_dim = max(smb.dims, key=lambda d: smb.sizes[d])
        spatial_dims = [d for d in smb.dims if d != time_dim]
        print(f"  Time dim: '{time_dim}', spatial dims: {spatial_dims}")

    if not spatial_dims:
        raise ValueError(f"Could not identify spatial dims. dims={list(smb.dims)}")

    # Coverage check and basin extraction using the 3D mask.
    # Iterate over the mask's own region dimension (not gdf rows) to
    # guarantee the index never exceeds the mask size.
    n_regions = mask_3d.sizes["region"]
    print(f"  Processing {n_regions} regions...")

    # Build a lookup from clean_name → gdf row for metadata
    name_to_row = {
        row["_clean_name"]: row
        for _, row in gdf.iterrows()
    }

    basin_smb: dict[str, xr.DataArray] = {}

    for i in range(n_regions):
        layer = mask_3d.isel(region=i)
        n_cells = int(layer.values.sum())

        # Resolve the sector name from regionmask coordinates
        if "abbrevs" in mask_3d.coords:
            clean_name  = str(layer["abbrevs"].values)
            sector_name = str(layer["names"].values) if "names" in mask_3d.coords else clean_name
        elif "names" in mask_3d.coords:
            sector_name = str(layer["names"].values)
            clean_name  = sector_name
        else:
            clean_name  = str(i)
            sector_name = clean_name

        if n_cells == 0:
            warnings.warn(
                f"Sector '{sector_name}' has zero RACMO grid cells. "
                f"Check the shapefile covers the RACMO domain."
            )
            continue

        print(f"  {sector_name}: {n_cells} grid cells")

        # Boolean mask as numpy array aligned with spatial dims
        sector_mask = layer.values.astype(bool)

        # Convert to xr.DataArray aligned with smb's spatial dims
        sector_mask_da = xr.DataArray(sector_mask, dims=spatial_dims)

        if area is not None:
            area_masked = area.where(sector_mask_da)
            smb_masked  = smb.where(sector_mask_da)
            smb_basin   = (
                (smb_masked * area_masked).sum(dim=spatial_dims)
                / area_masked.sum(dim=spatial_dims)
            )
        else:
            smb_basin = smb.where(sector_mask_da).mean(dim=spatial_dims)

        # Pull metadata from gdf lookup; fall back gracefully if missing
        row       = name_to_row.get(clean_name)
        region    = str(row["_clean_region"])    if row is not None and row.get("_clean_region")    else ""
        subregion = str(row["_clean_subregion"]) if row is not None and row.get("_clean_subregion") else ""

        smb_basin = smb_basin.assign_attrs({
            "long_name":     f"Basin-mean SMB — {sector_name}",
            "units":         smb.attrs.get("units", "m w.e."),
            "name":          sector_name,
            "subregion":     subregion,
            "region":        region,
            "n_cells":       int(sector_mask.sum()),
            "area_weighted": int(area is not None),
        })
        smb_basin.name = "smbgl"

        basin_smb[clean_name] = smb_basin
        print(f"    mean={float(smb_basin.mean()):.4f} {smb_basin.attrs['units']}")

    return basin_smb


# ======================================================================
# Step 4: Save to NetCDF and build catalog
# ======================================================================

def save_and_build_catalog(
    basin_smb: dict[str, xr.DataArray],
    output_dir: str | Path,
    var: str = "smbgl",
) -> RACMOCatalog:
    """
    Save one NetCDF per sector and return a fitted RACMOCatalog with
    region and subregion metadata populated from the DataArray attributes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving sector NetCDF files to: {output_dir}")

    catalog = RACMOCatalog(data_dir=output_dir, var=var)
    catalog._registry.clear()   # start fresh — don't auto-discover

    for key, da in basin_smb.items():
        out_path = output_dir / f"{key}_smb.nc"
        ds = xr.Dataset({var: da})
        ds.to_netcdf(out_path)

        # Pull all three hierarchy levels from DataArray attributes
        glacier_name = da.attrs.get("name",      key)
        subregion    = da.attrs.get("subregion", "") or None
        region       = da.attrs.get("region",    "") or None

        catalog.register(
            key,
            out_path,
            region    = region,
            subregion = subregion,
            name      = glacier_name,
        )
        print(f"  Saved: {out_path.name}  "
              f"[{region} / {subregion} / {glacier_name}]")

    print(f"\nCatalog built: {catalog}")
    return catalog


# ======================================================================
# Step 5: Diagnostic plot
# ======================================================================

def plot_overview(
    smb: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    name_col: str,
    basin_smb: dict[str, xr.DataArray],
    save_path: str | None = None,
    low_res: bool = False,
) -> None:
    """
    Two-panel figure:
      Left:  Time-mean SMB field with sector boundaries overlaid
      Right: Basin-mean SMB time series for all sectors

    Parameters
    ----------
    smb : xr.DataArray, shape (time, y, x)
    lat, lon : 2D coordinate arrays
    gdf : GeoDataFrame of sectors in WGS84
    name_col : str
    basin_smb : dict[str, xr.DataArray]
    save_path : str or None
    low_res : bool
        If True, skip interpolation for faster rendering.
    """
    print("\nGenerating overview plot...")

    colors = {name: _SECTOR_COLORS[i % len(_SECTOR_COLORS)]
              for i, name in enumerate(basin_smb.keys())}

    fig = plt.figure(figsize=(16, 7))

    # ── Left: SMB map with sector outlines ──
    if HAS_CARTOPY:
        proj = ccrs.SouthPolarStereo()
        ax_map = fig.add_subplot(1, 2, 1, projection=proj)
        _plot_map_cartopy(ax_map, smb, lat, lon, gdf, name_col,
                          colors, low_res)
    else:
        ax_map = fig.add_subplot(1, 2, 1)
        _plot_map_simple(ax_map, smb, lat, lon, gdf, name_col, colors)

    # ── Right: Time series per sector ──
    ax_ts = fig.add_subplot(1, 2, 2)
    _plot_time_series(ax_ts, basin_smb, colors)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150 if not low_res else 100,
                    bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def _plot_map_cartopy(ax, smb, lat, lon, gdf, name_col, colors, low_res):
    """Map panel using cartopy."""
    proj_data = ccrs.PlateCarree()
    proj_map  = ccrs.SouthPolarStereo()

    # Time-mean SMB
    smb_mean = smb.mean(dim="time").values

    # Step down resolution further if low_res
    step = 2 if low_res else 1

    ax.set_extent([-180, 180, -90, -60], crs=proj_data)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="black")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)

    # SMB colormesh
    vmax = np.nanpercentile(np.abs(smb_mean), 95)
    pcm = ax.pcolormesh(
        lon.values[::step, ::step],
        lat.values[::step, ::step],
        smb_mean[::step, ::step],
        cmap="RdBu",
        vmin=-vmax, vmax=vmax,
        transform=proj_data,
        shading="auto",
        alpha=0.85,
        rasterized=True,
    )
    plt.colorbar(pcm, ax=ax, fraction=0.03, label="Mean SMB (m w.e.)",
                 orientation="horizontal", pad=0.05)

    # Sector boundaries
    for _, row in gdf.iterrows():
        name  = row[name_col]
        clean = row["_clean_name"]
        color = colors.get(clean, "black")
        geom  = row.geometry
        if hasattr(geom, "geoms"):
            for part in geom.geoms:
                xs, ys = part.exterior.xy
                ax.plot(xs, ys, color=color, lw=1.5,
                        transform=proj_data, zorder=5)
        else:
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color=color, lw=1.5,
                    transform=proj_data, zorder=5)
        # Label at centroid
        cx = float(geom.centroid.x)
        cy = float(geom.centroid.y)
        if cy < -62:   # only label if within map extent
            ax.text(cx, cy, name, fontsize=7, color=color,
                    ha="center", va="center",
                    fontweight="bold", transform=proj_data, zorder=6)

    ax.set_title("Time-mean SMB with sector boundaries\n"
                 f"({int(smb.sizes['time']//12)} year mean)", fontsize=10)


def _plot_map_simple(ax, smb, lat, lon, gdf, name_col, colors):
    """Fallback map panel without cartopy."""
    smb_mean = smb.mean(dim="time").values
    vmax = np.nanpercentile(np.abs(smb_mean), 95)
    pcm  = ax.pcolormesh(
        lon.values, lat.values, smb_mean,
        cmap="RdBu", vmin=-vmax, vmax=vmax,
        shading="auto", alpha=0.85, rasterized=True,
    )
    plt.colorbar(pcm, ax=ax, label="Mean SMB (m w.e.)")

    # Sector boundaries (in lon/lat space)
    for _, row in gdf.iterrows():
        name  = row[name_col]
        clean = row["_clean_name"]
        color = colors.get(clean, "black")
        geom  = row.geometry
        if hasattr(geom, "geoms"):
            for part in geom.geoms:
                xs, ys = part.exterior.xy
                ax.plot(xs, ys, color=color, lw=1.2, zorder=5)
        else:
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color=color, lw=1.2, zorder=5)
        cx = float(geom.centroid.x)
        cy = float(geom.centroid.y)
        # ax.text(cx, cy, name, fontsize=7, color=color,
        #         ha="center", va="center", fontweight="bold", zorder=6)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Time-mean SMB with sector boundaries")


def _plot_time_series(ax, basin_smb, colors):
    """Time series panel."""
    for name, da in basin_smb.items():
        try:
            time_vals = da["time"].values
        except Exception:
            time_vals = np.arange(len(da))
        ax.plot(time_vals, da.values,
                color=colors.get(name, "gray"),
                lw=0.8, alpha=0.85, label=name)

    ax.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Basin-mean SMB (m w.e.)")
    ax.set_title("SMB time series by sector")
    ax.legend(fontsize=8, loc="upper left",
              ncol=max(1, len(basin_smb) // 8))


# ======================================================================
# Main pipeline function
# ======================================================================

def generate_racmo_catalog(
    racmo_path: str,
    shp_path: str,
    output_dir: str,
    smb_var: str | None = None,
    name_col: str | None = None,
    sectors: list[str] | None = None,
    low_res: bool = False,
    coarsen_factor: int = 4,
    plot: bool = True,
    save_plot: str | None = None,
) -> RACMOCatalog:
    """
    Full pipeline: RACMO + shapefile → per-sector NetCDF files + RACMOCatalog.

    Parameters
    ----------
    racmo_path : str
        Path to full-Antarctica RACMO NetCDF file.
    shp_path : str
        Path to sector shapefile (.shp or .gpkg).
    output_dir : str
        Directory to save per-sector NetCDF files.
    smb_var : str or None
        SMB variable name in RACMO file. Auto-detected if None.
    name_col : str or None
        Column with sector names in shapefile. Auto-detected if None.
    sectors : list of str or None
        Subset of sectors to process. All sectors if None.
    low_res : bool
        Coarsen RACMO grid for fast processing. Default False.
    coarsen_factor : int
        Coarsening factor for low_res mode. Default 4.
    plot : bool
        Generate overview plot. Default True.
    save_plot : str or None
        Path to save plot. If None and plot=True, displays interactively.

    Returns
    -------
    catalog : RACMOCatalog
    """
    print("=" * 60)
    print("RACMO Catalog Generator")
    print("=" * 60)

    # ── Load RACMO ──
    smb, lat, lon, area = load_racmo(
        racmo_path, var=smb_var,
        low_res=low_res, coarsen_factor=coarsen_factor,
    )

    # ── Load shapefile ──
    gdf, name_col, subregion_col, region_col = load_sectors(
        shp_path, name_col=name_col, sectors=sectors
    )

    # ── Extract basin means ──
    basin_smb = extract_basin_means(smb, lat, lon, area, gdf, name_col)

    if not basin_smb:
        raise RuntimeError(
            "No basin means were extracted. "
            "Check that the shapefile covers the RACMO domain and that "
            "the lat/lon variables are correctly identified."
        )

    # ── Save and build catalog ──
    catalog = save_and_build_catalog(basin_smb, output_dir)

    # ── Validate ──
    print("\nValidating catalog...")
    catalog.validate(verbose=True)

    # ── Plot ──
    if plot:
        plot_overview(
            smb, lat, lon, gdf, name_col, basin_smb,
            save_path=save_plot,
            low_res=low_res,
        )

    print("\n" + "=" * 60)
    print("Catalog generation complete.")
    print(f"  {len(catalog)} sectors processed.")
    print(f"  Files saved to: {Path(output_dir).resolve()}")
    print(f"\nTo use the catalog:")
    print(f"  from racmo_catalog import RACMOCatalog")
    print(f"  catalog = RACMOCatalog('{Path(output_dir).resolve()}')")
    print(f"  results = multi_basin_run(catalog.paths())")
    print("=" * 60)

    return catalog


# ======================================================================
# Command-line interface
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a RACMOCatalog from a full-Antarctica "
                    "RACMO file and a sector shapefile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--racmo",   required=True,
                        help="Path to RACMO NetCDF file.")
    parser.add_argument("--shp",     required=True,
                        help="Path to sector shapefile (.shp or .gpkg).")
    parser.add_argument("--outdir",  required=True,
                        help="Output directory for per-sector NetCDF files.")
    parser.add_argument("--var",     default=None,
                        help="SMB variable name (auto-detected if not given).")
    parser.add_argument("--name",    default=None,
                        help="Shapefile column with subregion names (auto-detected).")
    parser.add_argument("--region",  default=None,
                        help="Shapefile column with broad region labels (auto-detected).")
    parser.add_argument("--sectors", nargs="*", default=None,
                        help="Subset of sector names to process.")
    parser.add_argument("--low-res", action="store_true",
                        help="Coarsen grid for faster processing and plotting.")
    parser.add_argument("--coarsen", type=int, default=4,
                        help="Coarsening factor (used only with --low-res).")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate and display the overview plot.")
    parser.add_argument("--save-plot", default=None,
                        help="Path to save the overview plot.")

    args = parser.parse_args()

    generate_racmo_catalog(
        racmo_path    = args.racmo,
        shp_path      = args.shp,
        output_dir    = args.outdir,
        smb_var       = args.var,
        name_col      = args.name,
        sectors       = args.sectors,
        low_res       = args.low_res,
        coarsen_factor= args.coarsen,
        plot          = args.plot,
        save_plot     = args.save_plot,
    )


if __name__ == "__main__":
    main()