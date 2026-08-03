"""
syn_smb.core
============
Core pipeline components for synthetic SMB generation.
 
Importing from this subpackage:
    from syn_smb.core import SMBGenerator, Experiment
    from syn_smb import SMBGenerator, Experiment   # preferred — via top-level __init__
"""
"""
syn_smb.core
============
All pipeline components, grouped by function.

1D pipeline:
    SMBDataLoader, Preprocessor, GaussianTransform,
    SpectralSynthesizer, Experiment, SMBGenerator, Validator

Multi-basin:
    RACMOCatalog, BandAnalyser,
    multi_basin_run, plot_multibasin_psd,
    multibasin_metrics_table, plot_multibasin_metrics,
    plot_multibasin_split_test, compare_basin_bands

2D spatial pipeline:
    SMBFieldLoader, SpatialPreprocessor, EOFDecomposer,
    SMBFieldGenerator, SpatialValidator
"""

# ── 1D pipeline ───────────────────────────────────────────────────────
from .data_loader  import SMBDataLoader
from .preprocessor import Preprocessor
from .gaussianize  import GaussianTransform
from .spectral     import SpectralSynthesizer
from .experiment   import Experiment
from .generator    import SMBGenerator
from .validator    import Validator

# ── Multi-basin ───────────────────────────────────────────────────────
from .racmo_catalog import RACMOCatalog
from .band_analyzer import BandAnalyzer, compare_basin_bands
from .multi_basin   import (
    multi_basin_run,
    plot_multibasin_psd,
    multibasin_metrics_table,
    plot_multibasin_metrics,
    plot_multibasin_split_test,
    plot_multibasin_smb,
)

# ── 2D spatial pipeline ───────────────────────────────────────────────
from .smb_field_loader     import SMBFieldLoader
from .spatial_preprocessor import SpatialPreprocessor
from .eof_decomposer       import EOFDecomposer
from .smb_field_generator  import SMBFieldGenerator
from .spatial_validator    import SpatialValidator

# ── Explicit group namespaces (for tab-completion and docs) ───────────
__pipeline_1d__ = [
    "SMBDataLoader", "Preprocessor", "GaussianTransform",
    "SpectralSynthesizer", "Experiment", "SMBGenerator", "Validator",
]
__pipeline_multi__ = [
    "RACMOCatalog", "BandAnalyzer", "compare_basin_bands",
    "multi_basin_run", "plot_multibasin_psd", "multibasin_metrics_table",
    "plot_multibasin_metrics", "plot_multibasin_split_test", "plot_multibasin_smb",
]
__pipeline_2d__ = [
    "SMBFieldLoader", "SpatialPreprocessor", "EOFDecomposer",
    "SMBFieldGenerator", "SpatialValidator",
]

__all__ = __pipeline_1d__ + __pipeline_multi__ + __pipeline_2d__

# from .data_loader import SMBDataLoader
# from syn_smb.core.preprocessor import Preprocessor
# from syn_smb.core.gaussianize import GaussianTransform
# from syn_smb.core.spectral import SpectralSynthesizer
# from syn_smb.core.experiment import Experiment
# from syn_smb.core.generator import SMBGenerator
# from syn_smb.core.validator import Validator
# from syn_smb.core.multi_basin import (
#     multi_basin_run,
#     plot_multibasin_psd,
#     multibasin_metrics_table,
#     plot_multibasin_metrics,
# )
# from syn_smb.core.racmo_catalog import RACMOCatalog
# from syn_smb.core.band_analyzer import BandAnalyser

# #2D Imports
# from syn_smb.core.smb_field_loader import SMBFieldLoader
# from syn_smb.core.spatial_preprocessor import SpatialPreprocessor
# from syn_smb.core.eof_decomposer import EOFDecomposer
# from syn_smb.core.smb_field_generator import SMBFieldGenerator
# from syn_smb.core.spatial_validator import SpatialValidator

