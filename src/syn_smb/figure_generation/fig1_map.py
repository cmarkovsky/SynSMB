"""
plot_basin_map.py
=================
Publication-quality map of Antarctica with the five target WAIS basins
highlighted in the same colours used throughout the paper.

Usage
-----
    poetry run python plot_basin_map.py \
        --shp  ./data/IceBoundaries_Antarctica_V2.shp \
        --out  figures/fig01_basin_map.png

    # or from Python:
    from plot_basin_map import plot_basin_map
    plot_basin_map(
        shp_path="./data/IceBoundaries_Antarctica_V2.shp",
        save_path="figures/fig01_basin_map.png",
    )

Dependencies
------------
    poetry add cartopy geopandas matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ======================================================================
# Basin configuration
# ======================================================================

# Five target basins — keys must match the NAME column in the shapefile.
# Colors are the Wong (2011) colorblind-safe palette, consistent with
# multi_basin.py _BASIN_COLORS and all other paper figures.
BASINS = {
    "PineIsland": {"color": "#0072B2", "label": "Pine Island"},
    "Ronne":   {"color": "#E69F00", "label": "Ronne"},
    "Getz":       {"color": "#009E73", "label": "Getz"},
    "LarsenC":     {"color": "#D55E00", "label": "Larsen C"},
    "Ross_East":    {"color": "#CC79A7", "label": "Ross East"},
}

# Projection — Antarctic Polar Stereographic (standard for WAIS maps)
PROJ = ccrs.SouthPolarStereo(central_longitude=0.0)
DATA_CRS = ccrs.PlateCarree()

# Map extent in the stereographic projection (metres from pole)
# Centred on the Amundsen Sea sector
WAIS_EXTENT = [-180, 180, -90, -62]   # lon_min, lon_max, lat_min, lat_max


# ======================================================================
# Main plotting function
# ======================================================================

def plot_basin_map(
    shp_path: str,
    name_col: str = "NAME",
    save_path: str | None = None,
    dpi: int = 300,
    show_labels: bool = True,
    show_smb_background: bool = False,
    smb_path: str | None = None,
) -> plt.Figure:
    """
    Draw the Antarctica basin map.

    Parameters
    ----------
    shp_path : str
        Path to IceBoundaries_Antarctica_V2.shp.
    name_col : str
        Column in the shapefile containing basin names. Default 'NAME'.
    save_path : str or None
        If given, save the figure to this path.
    dpi : int
        Output resolution. 300 for print, 150 for screen. Default 300.
    show_labels : bool
        Annotate each highlighted basin with its name. Default True.
    show_smb_background : bool
        If True and smb_path is given, plot the time-mean RACMO SMB as
        a background colourfield before drawing basin outlines.
    smb_path : str or None
        Path to the full-domain RACMO NetCDF (only used if
        show_smb_background=True).

    Returns
    -------
    fig : plt.Figure
    """
    # ── Load shapefile ──
    print(f"Loading shapefile: {Path(shp_path).name}")
    gdf = gpd.read_file(shp_path)

    if name_col not in gdf.columns:
        available = [c for c in gdf.columns if c != "geometry"]
        raise ValueError(
            f"Column '{name_col}' not found. Available: {available}"
        )

    # Reproject to WGS84 for cartopy PlateCarree transform
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Separate target basins from background
    target_names = set(BASINS.keys())
    gdf_bg     = gdf[~gdf[name_col].isin(target_names)]
    gdf_target = gdf[ gdf[name_col].isin(target_names)]

    # ── Figure layout: main map + inset ──
    fig = plt.figure(figsize=(10, 10))

    # Main axes — full Antarctica
    ax_main = fig.add_axes(
        [0.05, 0.05, 0.90, 0.90],
        projection=PROJ,
    )

    # Inset axes — WAIS zoom (bottom-left corner)
    ax_inset = fig.add_axes(
        [0.62, 0.08, 0.35, 0.35],
        projection=PROJ,
    )

    for ax, extent, is_inset in [
        (ax_main,  [-180, 180, -90, -60], False),
        (ax_inset, [-140, -80,  -80, -70], True),
    ]:
        _draw_panel(
            ax, gdf_bg, gdf_target, gdf, name_col, extent,
            is_inset=is_inset,
            show_labels=show_labels and not is_inset,
        )

    # ── Legend ──
    handles = [
        mpatches.Patch(
            facecolor=meta["color"],
            edgecolor="black",
            linewidth=0.5,
            label=meta["label"],
        )
        for meta in BASINS.values()
    ]
    ax_main.legend(
        handles=handles,
        loc="lower left",
        fontsize=10,
        framealpha=0.9,
        edgecolor="gray",
        title="WAIS basins",
        title_fontsize=10,
    )

    # ── Inset border ──
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1.5)

    # ── Title ──
    ax_main.set_title(
        "West Antarctic Ice Sheet drainage basins",
        fontsize=13, pad=12,
    )

    plt.tight_layout(pad=0)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        print(f"Saved: {save_path}")

    return fig


# ======================================================================
# Panel drawing helper
# ======================================================================

def _draw_panel(
    ax,
    gdf_bg,
    gdf_target,
    gdf_all,
    name_col,
    extent,
    is_inset=False,
    show_labels=True,
) -> None:
    """Draw one map panel (main or inset)."""

    ax.set_extent(extent, crs=DATA_CRS)

    # ── Ocean background ──
    ax.add_feature(
        cfeature.OCEAN.with_scale("50m"),
        facecolor="#C8E0F0",
        zorder=0,
    )

    # ── All other basins in light grey ──
    for _, row in gdf_bg.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        ax.add_geometries(
            [geom],
            crs=DATA_CRS,
            facecolor="#DCDCDC",
            edgecolor="#AAAAAA",
            linewidth=0.3 if not is_inset else 0.2,
            zorder=1,
        )

    # ── Target basins in colour ──
    for _, row in gdf_target.iterrows():
        basin_name = row[name_col]
        if basin_name not in BASINS:
            continue
        meta  = BASINS[basin_name]
        color = meta["color"]
        geom  = row.geometry
        if geom is None or geom.is_empty:
            continue

        ax.add_geometries(
            [geom],
            crs=DATA_CRS,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8 if not is_inset else 0.5,
            alpha=0.85,
            zorder=2,
        )

        # ── Basin label ──
        if show_labels:
            centroid = geom.centroid
            cx, cy   = centroid.x, centroid.y

            # Only label if the centroid is within the map extent
            lon_min, lon_max, lat_min, lat_max = extent
            if not (lon_min <= cx <= lon_max and lat_min <= cy <= lat_max):
                continue

            ax.text(
                cx, cy,
                meta["label"],
                transform=DATA_CRS,
                fontsize=8.5,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                zorder=5,
                path_effects=[
                    pe.withStroke(linewidth=2.0, foreground="black")
                ],
            )

    # ── Gridlines ──
    gl = ax.gridlines(
        crs=DATA_CRS,
        draw_labels=not is_inset,
        linewidth=0.4 if not is_inset else 0.2,
        color="gray",
        alpha=0.5,
        linestyle="--",
        x_inline=False,
        y_inline=False,
    )
    if not is_inset:
        gl.top_labels    = False
        gl.right_labels  = False
        gl.xlabel_style  = {"size": 8, "color": "#444444"}
        gl.ylabel_style  = {"size": 8, "color": "#444444"}
        gl.xlocator = mpl.ticker.FixedLocator(range(-180, 181, 30))
        gl.ylocator = mpl.ticker.FixedLocator(range(-90, -59, 10))

    # ── Inset zoom box on the main panel ──
    # (drawn by _add_zoom_box separately after both panels exist)


def _add_zoom_indicator(ax_main, ax_inset, extent_inset) -> None:
    """
    Draw a rectangle on the main map showing the inset extent, with
    connecting lines to the inset axes corners.
    (Optional — call after both axes are drawn.)
    """
    import matplotlib.patches as mpatches
    from cartopy.mpl.patch import geos_to_path

    lon0, lon1, lat0, lat1 = extent_inset
    rect = mpatches.Rectangle(
        (lon0, lat0), lon1 - lon0, lat1 - lat0,
        linewidth=1.5, edgecolor="#333333", facecolor="none",
        transform=DATA_CRS, zorder=6,
    )
    ax_main.add_patch(rect)


# ======================================================================
# Supplementary: quick per-basin thumbnail strip
# ======================================================================

def plot_basin_thumbnails(
    shp_path: str,
    name_col: str = "NAME",
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    One small map per basin in a horizontal strip — useful as a
    supplement figure showing each basin's geographic context.

    Parameters
    ----------
    shp_path, name_col, save_path, dpi : see plot_basin_map().
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    n = len(BASINS)
    fig, axes = plt.subplots(
        1, n,
        figsize=(4 * n, 5),
        subplot_kw={"projection": PROJ},
    )
    fig.suptitle("Target WAIS basins", fontsize=12, y=1.01)

    gdf_bg = gdf[~gdf[name_col].isin(BASINS.keys())]

    for ax, (basin_name, meta) in zip(axes, BASINS.items()):
        gdf_this = gdf[gdf[name_col] == basin_name]
        if gdf_this.empty:
            ax.set_title(meta["label"], fontsize=9)
            continue

        # Compute a tight extent around this basin (in WGS84)
        bounds = gdf_this.total_bounds   # [minx, miny, maxx, maxy]
        pad    = 3.0                     # degrees of padding
        extent = [
            bounds[0] - pad, bounds[2] + pad,
            bounds[1] - pad, bounds[3] + pad,
        ]

        _draw_panel(
            ax, gdf_bg, gdf_this, gdf, name_col,
            extent=extent,
            is_inset=True,
            show_labels=False,
        )

        ax.set_title(meta["label"], fontsize=10, color=meta["color"],
                     fontweight="bold")

        # Coloured border matching basin colour
        for spine in ax.spines.values():
            spine.set_edgecolor(meta["color"])
            spine.set_linewidth(2.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        print(f"Saved: {save_path}")

    return fig


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Antarctica basin map for Figure 1."
    )
    parser.add_argument("--shp",  required=True,
                        help="Path to IceBoundaries_Antarctica_V2.shp")
    parser.add_argument("--out",  default="figures/fig01_basin_map.png",
                        help="Output path for the main map.")
    parser.add_argument("--thumbnails", action="store_true",
                        help="Also produce the per-basin thumbnail strip.")
    parser.add_argument("--dpi",  type=int, default=300)
    parser.add_argument("--name-col", default="NAME",
                        help="Shapefile column with basin names.")
    args = parser.parse_args()

    fig = plot_basin_map(
        shp_path=args.shp,
        name_col=args.name_col,
        save_path=args.out,
        dpi=args.dpi,
    )
    plt.show()

    if args.thumbnails:
        thumb_path = str(Path(args.out).with_stem(
            Path(args.out).stem + "_thumbnails"
        ))
        plot_basin_thumbnails(
            shp_path=args.shp,
            name_col=args.name_col,
            save_path=thumb_path,
            dpi=args.dpi,
        )
        plt.show()


if __name__ == "__main__":
    main()