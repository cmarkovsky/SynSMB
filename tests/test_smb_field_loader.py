"""
test_smb_field_loader.py
========================
Tests for SMBFieldLoader.

All unit tests use synthetic data — no RACMO file or real shapefile
needed. The fixtures create minimal NetCDF and GeoDataFrame objects that
replicate the structure of the real inputs.

Run unit tests:
    pytest test_smb_field_loader.py -v

Run real-data integration test:
    python test_smb_field_loader.py \
        ./data/RACMO2.4p1_ANT11.nc \
        ./data/IceBoundaries_Antarctica_V2.shp \
        PineIsland
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon

from syn_smb import SMBFieldLoader


# ======================================================================
# Fixtures — synthetic RACMO-like data
# ======================================================================

# Small grid — fast to run, realistic structure
NY, NX, NT = 12, 15, 60   # rlat, rlon, time (5 years monthly)

# Lat/lon range covering a simple Antarctica-like region
LAT_RANGE = (-80.0, -70.0)
LON_RANGE = (-120.0, -90.0)


@pytest.fixture(scope="module")
def synthetic_racmo_nc(tmp_path_factory) -> Path:
    """
    Write a minimal RACMO-like NetCDF file with:
      - smbgl (time, height, rlat, rlon) in kg m-2
      - lat, lon (rlat, rlon)
      - time coordinate
    """
    rng  = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("data") / "racmo_synthetic.nc"

    lats = np.linspace(LAT_RANGE[0], LAT_RANGE[1], NY)
    lons = np.linspace(LON_RANGE[0], LON_RANGE[1], NX)
    lon2d, lat2d = np.meshgrid(lons, lats)

    time = xr.cftime_range("1979-01", periods=NT, freq="MS")

    # smbgl: (time, height, rlat, rlon) — height=1 to match real RACMO
    data = rng.normal(40.0, 10.0, size=(NT, 1, NY, NX)).astype(np.float32)

    ds = xr.Dataset(
        {
            "smbgl": xr.DataArray(
                data,
                dims=["time", "height", "rlat", "rlon"],
                attrs={"units": "kg m-2", "long_name": "Surface mass balance"},
            ),
            "lat": xr.DataArray(
                lat2d.astype(np.float32),
                dims=["rlat", "rlon"],
                attrs={"units": "degrees_north"},
            ),
            "lon": xr.DataArray(
                lon2d.astype(np.float32),
                dims=["rlat", "rlon"],
                attrs={"units": "degrees_east"},
            ),
        },
        coords={"time": time},
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def synthetic_shapefile(tmp_path_factory) -> Path:
    """
    Write a shapefile with two basins:
      - 'TargetBasin': covers the central part of the synthetic grid
      - 'OtherBasin':  covers a small corner of the grid

    The target basin covers roughly the central 50% of the lat/lon range.
    """
    path = tmp_path_factory.mktemp("shp") / "basins.shp"

    lat_mid  = (LAT_RANGE[0] + LAT_RANGE[1]) / 2
    lon_mid  = (LON_RANGE[0] + LON_RANGE[1]) / 2
    lat_span = LAT_RANGE[1] - LAT_RANGE[0]
    lon_span = LON_RANGE[1] - LON_RANGE[0]

    # Central polygon covering ~50% of the grid
    target = Polygon([
        (lon_mid - lon_span * 0.25, lat_mid - lat_span * 0.25),
        (lon_mid + lon_span * 0.25, lat_mid - lat_span * 0.25),
        (lon_mid + lon_span * 0.25, lat_mid + lat_span * 0.25),
        (lon_mid - lon_span * 0.25, lat_mid + lat_span * 0.25),
    ])

    # Small corner polygon
    other = Polygon([
        (LON_RANGE[0], LAT_RANGE[0]),
        (LON_RANGE[0] + lon_span * 0.1, LAT_RANGE[0]),
        (LON_RANGE[0] + lon_span * 0.1, LAT_RANGE[0] + lat_span * 0.1),
        (LON_RANGE[0], LAT_RANGE[0] + lat_span * 0.1),
    ])

    gdf = gpd.GeoDataFrame(
        {"NAME": ["TargetBasin", "OtherBasin"]},
        geometry=[target, other],
        crs="EPSG:4326",
    )
    gdf.to_file(path)
    return path


@pytest.fixture(scope="module")
def loader(synthetic_racmo_nc, synthetic_shapefile) -> SMBFieldLoader:
    """A fitted loader for TargetBasin — session-scoped for speed."""
    return SMBFieldLoader(
        racmo_path = synthetic_racmo_nc,
        shp_path   = synthetic_shapefile,
        basin_name = "TargetBasin",
        name_col   = "NAME",
    )


# ======================================================================
# 1. Construction
# ======================================================================

class TestConstruction:

    def test_repr_before_load(self, synthetic_racmo_nc, synthetic_shapefile):
        ldr = SMBFieldLoader(synthetic_racmo_nc, synthetic_shapefile,
                             "TargetBasin")
        r = repr(ldr)
        assert "TargetBasin" in r
        assert "loaded=False" in r

    def test_repr_after_load(self, loader):
        loader.load()
        r = repr(loader)
        assert "loaded=True" in r
        assert "n_cells=" in r

    def test_missing_racmo_raises(self, synthetic_shapefile):
        ldr = SMBFieldLoader("nonexistent.nc", synthetic_shapefile,
                             "TargetBasin")
        with pytest.raises(FileNotFoundError, match="RACMO file"):
            ldr.load()

    def test_missing_shapefile_raises(self, synthetic_racmo_nc):
        ldr = SMBFieldLoader(synthetic_racmo_nc, "nonexistent.shp",
                             "TargetBasin")
        with pytest.raises(FileNotFoundError, match="Shapefile"):
            ldr.load()

    def test_unknown_basin_raises(self, synthetic_racmo_nc,
                                   synthetic_shapefile):
        ldr = SMBFieldLoader(synthetic_racmo_nc, synthetic_shapefile,
                             "NoSuchBasin")
        with pytest.raises(ValueError, match="not found"):
            ldr.load()

    def test_missing_name_col_raises(self, synthetic_racmo_nc,
                                      synthetic_shapefile):
        ldr = SMBFieldLoader(synthetic_racmo_nc, synthetic_shapefile,
                             "TargetBasin", name_col="WRONG_COL")
        with pytest.raises(ValueError, match="WRONG_COL"):
            ldr.load()


# ======================================================================
# 2. Output shape and dimensions
# ======================================================================

class TestOutputShape:

    def test_field_is_dataarray(self, loader):
        field = loader.load()
        assert isinstance(field, xr.DataArray)

    def test_field_has_three_dims(self, loader):
        field = loader.load()
        assert field.ndim == 3, f"Expected 3D, got {field.ndim}D: {field.dims}"

    def test_time_dim_present(self, loader):
        field = loader.load()
        assert "time" in field.dims

    def test_time_length_correct(self, loader):
        field = loader.load()
        assert field.sizes["time"] == NT

    def test_spatial_dims_match_grid(self, loader):
        field = loader.load()
        spatial_sizes = {d: s for d, s in field.sizes.items()
                         if d != "time"}
        assert set(spatial_sizes.values()) == {NY, NX}

    def test_height_dim_squeezed(self, loader):
        """height=1 must be removed — should not appear in output."""
        field = loader.load()
        assert "height" not in field.dims


# ======================================================================
# 3. Unit conversion
# ======================================================================

class TestUnits:

    def test_output_units_are_m_we(self, loader):
        field = loader.load()
        units = field.attrs.get("units", "")
        assert "m" in units.lower(), f"Expected m w.e. units, got '{units}'"

    def test_values_are_small_compared_to_kg_m2(self, loader):
        """
        Input is ~40 kg m-2; after /1000 output should be ~0.04 m w.e.
        Values well above 1.0 indicate unit conversion failed.
        """
        field = loader.load()
        valid = field.values[np.isfinite(field.values)]
        assert valid.mean() < 1.0, (
            f"Mean value {valid.mean():.3f} suggests kg m-2 not converted."
        )
        assert valid.mean() > 0.0


# ======================================================================
# 4. Basin masking
# ======================================================================

class TestBasinMask:

    def test_basin_mask_is_dataarray(self, loader):
        assert isinstance(loader.basin_mask, xr.DataArray)

    def test_basin_mask_is_bool(self, loader):
        assert loader.basin_mask.dtype == bool

    def test_basin_mask_shape_matches_field(self, loader):
        field = loader.load()
        spatial_dims = {d: s for d, s in field.sizes.items()
                        if d != "time"}
        for dim, size in spatial_dims.items():
            assert loader.basin_mask.sizes[dim] == size

    def test_some_cells_inside_basin(self, loader):
        assert loader.n_valid_cells > 0, "No cells inside basin."

    def test_some_cells_outside_basin(self, loader):
        total = NY * NX
        assert loader.n_valid_cells < total, (
            "All cells inside basin — masking has no effect."
        )

    def test_nan_outside_basin(self, loader):
        """Cells outside the basin must be NaN in the loaded field."""
        field = loader.load()
        outside = ~loader.basin_mask

        # At least one timestep must have NaN where outside==True
        any_nan_outside = np.isnan(
            field.isel(time=0).values[outside.values]
        ).all()
        assert any_nan_outside, (
            "Expected NaN outside basin but found finite values."
        )

    def test_valid_values_inside_basin(self, loader):
        """Cells inside the basin must have finite values (not all NaN)."""
        field  = loader.load()
        inside = loader.basin_mask
        inside_vals = field.isel(time=0).values[inside.values]
        assert np.isfinite(inside_vals).all(), (
            "Found NaN inside the basin mask."
        )

    def test_different_basins_give_different_masks(
        self, synthetic_racmo_nc, synthetic_shapefile
    ):
        ldr_target = SMBFieldLoader(synthetic_racmo_nc, synthetic_shapefile,
                                    "TargetBasin")
        ldr_other  = SMBFieldLoader(synthetic_racmo_nc, synthetic_shapefile,
                                    "OtherBasin")
        ldr_target.load()
        ldr_other.load()
        # TargetBasin should have more cells (larger polygon in our fixture)
        assert ldr_target.n_valid_cells > ldr_other.n_valid_cells


# ======================================================================
# 5. Lat/lon properties
# ======================================================================

class TestLatLon:

    def test_lat_is_dataarray(self, loader):
        loader.load()
        assert isinstance(loader.lat, xr.DataArray)

    def test_lon_is_dataarray(self, loader):
        loader.load()
        assert isinstance(loader.lon, xr.DataArray)

    def test_lat_shape_matches_spatial_dims(self, loader):
        field = loader.load()
        spatial_sizes = {d: s for d, s in field.sizes.items()
                         if d != "time"}
        for dim, size in spatial_sizes.items():
            assert loader.lat.sizes[dim] == size

    def test_lon_normalised(self, loader):
        """Longitude must be in [-180, 180]."""
        loader.load()
        assert float(loader.lon.min()) >= -180.0
        assert float(loader.lon.max()) <=  180.0

    def test_lat_in_southern_hemisphere(self, loader):
        loader.load()
        assert float(loader.lat.max()) < 0.0, (
            "Expected Antarctic latitudes (all < 0)."
        )


# ======================================================================
# 6. Caching — load() called twice returns same object
# ======================================================================

class TestCaching:

    def test_load_is_idempotent(self, loader):
        f1 = loader.load()
        f2 = loader.load()
        assert f1 is f2, "load() should return the cached DataArray."

    def test_basin_mask_available_before_explicit_load(
        self, synthetic_racmo_nc, synthetic_shapefile
    ):
        """Accessing .basin_mask should trigger load() automatically."""
        ldr = SMBFieldLoader(synthetic_racmo_nc, synthetic_shapefile,
                             "TargetBasin")
        mask = ldr.basin_mask       # no explicit load() call
        assert isinstance(mask, xr.DataArray)
        assert ldr._field is not None   # side effect: field loaded too


# ======================================================================
# 7. Variable auto-detection
# ======================================================================

class TestVariableDetection:

    def test_autodetect_smbgl(self, synthetic_racmo_nc, synthetic_shapefile):
        """smb_var=None should auto-detect 'smbgl'."""
        ldr = SMBFieldLoader(synthetic_racmo_nc, synthetic_shapefile,
                             "TargetBasin", smb_var=None)
        field = ldr.load()
        assert field is not None

    def test_explicit_var_name(self, synthetic_racmo_nc, synthetic_shapefile):
        """Passing smb_var='smbgl' explicitly should also work."""
        ldr = SMBFieldLoader(synthetic_racmo_nc, synthetic_shapefile,
                             "TargetBasin", smb_var="smbgl")
        field = ldr.load()
        assert field is not None

    def test_wrong_var_name_raises(self, synthetic_racmo_nc,
                                    synthetic_shapefile):
        ldr = SMBFieldLoader(synthetic_racmo_nc, synthetic_shapefile,
                             "TargetBasin", smb_var="nonexistent_var")
        with pytest.raises(ValueError, match="not found"):
            ldr.load()


# ======================================================================
# 8. Edge cases
# ======================================================================

class TestEdgeCases:

    def test_nc_with_positive_longitudes(self, tmp_path, synthetic_shapefile):
        """Longitudes in [0, 360] should be normalised to [-180, 180]."""
        rng = np.random.default_rng(1)
        ny, nx, nt = 6, 8, 12

        lats = np.linspace(-80, -70, ny)
        lons = np.linspace(240, 270, nx)   # positive longitudes
        lon2d, lat2d = np.meshgrid(lons, lats)
        time = xr.cftime_range("1979-01", periods=nt, freq="MS")

        ds = xr.Dataset({
            "smbgl": xr.DataArray(
                rng.normal(40, 10, (nt, ny, nx)).astype(np.float32),
                dims=["time", "rlat", "rlon"],
                attrs={"units": "kg m-2"},
            ),
            "lat": xr.DataArray(lat2d.astype(np.float32),
                                 dims=["rlat", "rlon"]),
            "lon": xr.DataArray(lon2d.astype(np.float32),
                                 dims=["rlat", "rlon"]),
        }, coords={"time": time})

        nc_path = tmp_path / "positive_lons.nc"
        ds.to_netcdf(nc_path)

        # Shapefile polygon covering the grid in [-180,180] lon space
        poly = Polygon([(-120, -80), (-90, -80), (-90, -70), (-120, -70)])
        gdf  = gpd.GeoDataFrame({"NAME": ["TestBasin"]},
                                  geometry=[poly], crs="EPSG:4326")
        shp_path = tmp_path / "shp.shp"
        gdf.to_file(shp_path)

        ldr = SMBFieldLoader(nc_path, shp_path, "TestBasin")
        ldr.load()
        assert float(ldr.lon.max()) <= 180.0

    def test_nc_already_in_m_we(self, tmp_path, synthetic_shapefile):
        """If units are already m w.e., no conversion should be applied."""
        rng = np.random.default_rng(2)
        ny, nx, nt = 6, 8, 12
        lats = np.linspace(-80, -70, ny)
        lons = np.linspace(-120, -90, nx)
        lon2d, lat2d = np.meshgrid(lons, lats)
        time = xr.cftime_range("1979-01", periods=nt, freq="MS")

        raw_values = rng.normal(0.04, 0.01, (nt, ny, nx)).astype(np.float32)
        ds = xr.Dataset({
            "smbgl": xr.DataArray(
                raw_values, dims=["time", "rlat", "rlon"],
                attrs={"units": "m w.e."},
            ),
            "lat": xr.DataArray(lat2d.astype(np.float32),
                                 dims=["rlat", "rlon"]),
            "lon": xr.DataArray(lon2d.astype(np.float32),
                                 dims=["rlat", "rlon"]),
        }, coords={"time": time})
        nc_path = tmp_path / "m_we.nc"
        ds.to_netcdf(nc_path)

        poly = Polygon([(-120, -80), (-90, -80), (-90, -70), (-120, -70)])
        gdf  = gpd.GeoDataFrame({"NAME": ["B"]}, geometry=[poly],
                                  crs="EPSG:4326")
        shp_path = tmp_path / "shp.shp"
        gdf.to_file(shp_path)

        ldr   = SMBFieldLoader(nc_path, shp_path, "B")
        field = ldr.load()
        inside = field.values[np.isfinite(field.values)]
        # Values should still be ~0.04, not divided by 1000 again (~0.00004)
        assert inside.mean() > 0.01, (
            "m w.e. input was divided by 1000 — double conversion."
        )


# ======================================================================
# Real-data integration test
# ======================================================================

def run_real_data(racmo_path: str, shp_path: str, basin_name: str) -> None:
    """
    Full integration test on real RACMO + shapefile.
    Pass paths as command-line arguments.
    """
    print(f"\n{'='*60}")
    print(f"Real-data integration test: {basin_name}")
    print(f"{'='*60}")

    ldr = SMBFieldLoader(
        racmo_path = racmo_path,
        shp_path   = shp_path,
        basin_name = basin_name,
    )

    field = ldr.load()

    print(f"\nField shape:        {dict(field.sizes)}")
    print(f"Valid cells:        {ldr.n_valid_cells}")
    print(f"Time range:         {field.time.values[0]} → "
          f"{field.time.values[-1]}")
    print(f"Units:              {field.attrs.get('units', '?')}")

    inside = field.values[np.isfinite(field.values)]
    print(f"Mean (valid cells): {inside.mean():.4f} m w.e.")
    print(f"Std  (valid cells): {inside.std():.4f} m w.e.")
    print(f"Range:              [{inside.min():.4f}, {inside.max():.4f}]")

    # Sanity checks
    assert ldr.n_valid_cells > 0,   "No valid cells — mask failed."
    assert inside.mean() < 1.0,     "Mean > 1 m w.e. suggests no unit conversion."
    assert "time" in field.dims,    "No time dimension."
    assert "height" not in field.dims, "height not squeezed."
    assert float(ldr.lon.max()) <= 180.0, "Longitude not normalised."

    print("\n✓ All sanity checks passed.")

    # Quick plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"SMBFieldLoader — {basin_name}", fontsize=12)

    # Time-mean field
    ax = axes[0]
    mean_field = field.mean(dim="time")
    pcm = ax.pcolormesh(mean_field.values, cmap="Blues")
    plt.colorbar(pcm, ax=ax, label="Mean SMB (m w.e.)")
    ax.set_title("Time-mean SMB field\n(NaN = outside basin)")

    # Basin-mean time series (for comparison with 1D)
    ax = axes[1]
    basin_mean = field.mean(dim=[d for d in field.dims if d != "time"])
    ax.plot(basin_mean.values, lw=0.8, color="tab:blue")
    ax.axhline(float(basin_mean.mean()), color="tab:red",
               linestyle="--", lw=1, label="Mean")
    ax.set_xlabel("Time step (months)")
    ax.set_ylabel("SMB (m w.e.)")
    ax.set_title("Basin-mean time series\n(collapsed from 2D field)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"smb_field_loader_{basin_name}.png",
                dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: smb_field_loader_{basin_name}.png")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_real_data(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(
            "Usage: python test_smb_field_loader.py "
            "<racmo.nc> <shapefile.shp> <basin_name>\n"
            "Running pytest instead..."
        )
        import subprocess
        subprocess.run(["python", "-m", "pytest", __file__, "-v"])