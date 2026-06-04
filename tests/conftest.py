import numpy as np
import pytest


@pytest.fixture
def monthly_freqs():
    """
    One-sided frequency array for a 45-year monthly record.
    Sampling frequency: 12 cycles/year.
    Matches the RACMO record length used in the paper.
    """
    n = 45 * 12  # 540 months
    return np.fft.rfftfreq(n, d=1.0 / 12.0)


@pytest.fixture
def short_freqs():
    """Minimal frequency array for fast unit tests."""
    n = 120  # 10 years monthly
    return np.fft.rfftfreq(n, d=1.0 / 12.0)