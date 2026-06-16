"""
syn_smb.core
============
Core pipeline components for synthetic SMB generation.
 
Importing from this subpackage:
    from syn_smb.core import SMBGenerator, Experiment
    from syn_smb import SMBGenerator, Experiment   # preferred — via top-level __init__
"""
 
from syn_smb.core.data_loader import SMBDataLoader
from syn_smb.core.preprocessor import Preprocessor
from syn_smb.core.gaussianize import GaussianTransform
from syn_smb.core.spectral import SpectralSynthesizer
from syn_smb.core.experiment import Experiment
from syn_smb.core.generator import SMBGenerator
from syn_smb.core.validator import Validator
from syn_smb.core.multi_basin import (
    multi_basin_run,
    plot_multibasin_psd,
    multibasin_metrics_table,
)
from syn_smb.core.racmo_catalog import RACMOCatalog
from syn_smb.core.band_analyzer import BandAnalyser

#2D Imports
from syn_smb.core.smb_field_loader import SMBFieldLoader
from syn_smb.core.spatial_preprocessor import SpatialPreprocessor
from syn_smb.core.eof_decomposer import EOFDecomposer
from syn_smb.core.smb_field_generator import SMBFieldGenerator
__all__ = [
    "SMBDataLoader",
    "Preprocessor",
    "GaussianTransform",
    "SpectralSynthesizer",
    "Experiment",
    "SMBGenerator",
    "Validator",
    "multi_basin_run",
    "plot_multibasin_psd",
    "multibasin_metrics_table",
    "RACMOCatalog",
    "BandAnalyser",
    "SMBFieldLoader",
    "SpatialPreprocessor",
    "EOFDecomposer",
    "SMBFieldGenerator",
]
