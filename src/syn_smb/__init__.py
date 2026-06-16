
"""
syn_smb
=======
Synthetic SMB generation pipeline for Pine Island Glacier.
 
Typical usage:
    from syn_smb import SMBGenerator, Experiment, Validator
 
    gen = SMBGenerator.from_path("./data/PIG_smb.nc")
    ds  = gen.generate(Experiment.baseline())
    val = Validator(gen, smb)
"""
 
from syn_smb.core import (
    SMBDataLoader,
    Preprocessor,
    GaussianTransform,
    SpectralSynthesizer,
    Experiment,
    SMBGenerator,
    Validator,
    multi_basin_run,
    plot_multibasin_psd,
    multibasin_metrics_table,
    RACMOCatalog,
    BandAnalyser,
    SMBFieldLoader,
    SpatialPreprocessor,
    EOFDecomposer,
    SMBFieldGenerator,
)
 
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
