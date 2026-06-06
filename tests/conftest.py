"""
conftest.py
===========
Shared pytest fixtures for the syn_smb test suite.

Placing fixtures here makes them available to all test files in the
tests/ directory without importing. pytest discovers conftest.py
automatically.
"""

import numpy as np
import xarray as xr
import pytest
from syn_smb import (
    SMBGenerator,
    Experiment,
    GaussianTransform,
    SpectralSynthesizer,
    Preprocessor,
    Validator,
)


# ======================================================================
# Shared data fixtures
# ======================================================================

@pytest.fixture(scope="session")
def rng():
    """Session-scoped RNG — same seed for the full test run."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def monthly_time():
    """45-year monthly cftime coordinate matching the RACMO record."""
    return xr.cftime_range(start="1979", periods=540, freq="MS")


@pytest.fixture(scope="session")
def synthetic_smb(monthly_time):
    """
    Realistic synthetic SMB with known trend, seasonal cycle, and noise.
    Session-scoped so it is constructed once and reused across all tests.
    """
    rng = np.random.default_rng(42)
    t      = np.arange(540)
    months = np.array([d.month for d in monthly_time.values])
    values = (
        0.04
        + 5e-5 * t
        + 0.03 * np.sin(2 * np.pi * (months - 3) / 12)
        + rng.normal(0, 0.02, 540)
    )
    return xr.DataArray(
        values,
        coords={"time": monthly_time},
        dims=["time"],
        name="smb",
        attrs={"units": "m w.e. a$^{-1}$"},
    )


@pytest.fixture(scope="session")
def fitted_generator(synthetic_smb):
    """
    SMBGenerator fitted once for the session.
    nperseg=30 keeps tests fast on short synthetic data.
    """
    gen = SMBGenerator(nperseg=30)
    gen.fit(synthetic_smb)
    return gen


@pytest.fixture(scope="session")
def baseline_dataset(fitted_generator):
    """Baseline ensemble dataset (10 members, 100 years) for validation tests."""
    return fitted_generator.generate(
        Experiment(n_years=100, n_members=10, seed=0)
    )


@pytest.fixture(scope="session")
def validator(fitted_generator, synthetic_smb):
    """Validator fitted on session-scoped generator and SMB."""
    return Validator(fitted_generator, synthetic_smb)