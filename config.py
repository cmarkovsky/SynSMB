# config.py  — project root, committed to the repo
"""
Central configuration for the multi-basin SMB synthesis project.
Edit BASINS to change which basins are included in all figures and runs.
Names must match the NAME column in IceBoundaries_Antarctica_V2.shp exactly.
"""

RACMO_PATH = "./data/smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc"
SHP_PATH   = "./data/IceBoundaries_Antarctica_v02.shp"
SECTORS_DIR = "./data/sectors_smb/"

# ── The five target basins ──────────────────────────────────────────
# Keys are the NAME column values from the shapefile.
# Labels are used in figure titles and legends.
# Colors are the Wong (2011) colorblind-safe palette — must match
# _BASIN_COLORS in multi_basin.py and plot_basin_map.py.
BASINS = {
    "Pine_Island": {"label": "Pine Island", "color": "#0072B2"},
    "LarsenC":   {"label": "Larsen C",    "color": "#E69F00"},
    # "Getz":       {"label": "Getz",        "color": "#009E73"},
    # "Ronne":     {"label": "Ronne",      "color": "#D55E00"},
    # "Ross_East":    {"label": "Ross East",     "color": "#CC79A7"},
}

BASIN_NAMES  = list(BASINS.keys())    # ["PineIsland", "Thwaites", ...]
BASIN_LABELS = [v["label"]  for v in BASINS.values()]
BASIN_COLORS = [v["color"]  for v in BASINS.values()]