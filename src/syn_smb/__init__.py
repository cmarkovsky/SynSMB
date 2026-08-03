"""
syn_smb — Synthetic SMB Generation Pipeline
============================================
Generates arbitrarily long, statistically consistent synthetic surface
mass balance (SMB) time series and spatial fields for West Antarctic
ice dynamics experiments.

Quick imports
-------------
    # 1D basin-mean pipeline
    from syn_smb import SMBGenerator, Experiment, Validator

    # Multi-basin tools
    from syn_smb import RACMOCatalog, multi_basin_run, BandAnalyser

    # 2D spatial pipeline
    from syn_smb import SMBFieldLoader, SMBFieldGenerator, SpatialValidator

Grouped imports (same as above, but explicit about the layer)
---------------------
    from syn_smb.core import SMBGenerator          # 1D
    from syn_smb.core import multi_basin_run       # multi-basin
    from syn_smb.core import SMBFieldGenerator     # 2D
"""

# ── 1D pipeline ───────────────────────────────────────────────────────
import syn_smb
from syn_smb.core import (
    SMBDataLoader,
    Preprocessor,
    GaussianTransform,
    SpectralSynthesizer,
    Experiment,
    SMBGenerator,
    Validator,
)

# ── Multi-basin ───────────────────────────────────────────────────────
from syn_smb.core import (
    RACMOCatalog,
    BandAnalyzer,
    compare_basin_bands,
    multi_basin_run,
    plot_multibasin_psd,
    multibasin_metrics_table,
    plot_multibasin_metrics,
    plot_multibasin_split_test,
    plot_multibasin_smb,
)

# ── 2D spatial pipeline ───────────────────────────────────────────────
from syn_smb.core import (
    SMBFieldLoader,
    SpatialPreprocessor,
    EOFDecomposer,
    SMBFieldGenerator,
    SpatialValidator,
)

__version__ = "0.2.0"
__all__ = list(syn_smb.core.__all__)   # mirror core exactly