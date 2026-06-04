import xarray as xr
import numpy as np
from pathlib import Path
from scipy.stats import norm

def load_data(smb_path: Path, smb_var = "smbgl") -> xr.DataArray:
    if not smb_path.exists():
        raise FileNotFoundError(f"SMB data path {smb_path} does not exist.")
    if smb_path.suffix == '.nc':
        ds = xr.open_dataset(smb_path)
        if smb_var not in ds.data_vars:
            raise ValueError(f"Dataset does not contain '{smb_var}' variable.")
        if ds[smb_var].attrs.get("units", "unknown") == "kg m-2":
            ds[smb_var] = ds[smb_var] / 1000.0  # kg m-2 to m w.e.
            ds[smb_var].attrs["units"] = "m w.e. a$^{-1}$"
        return ds[smb_var]
    else:
        raise ValueError(f"Unsupported file format: {smb_path.suffix}")

def detrend_linear(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Detrend along 'time' using xarray.polyfit.
    Returns (residual, trend) as DataArrays.
    """
    fit = da.polyfit(dim='time', deg=1)
    trend = xr.polyval(da['time'], fit)
    trend_vals = xr.polyval(da['time'], fit.polyfit_coefficients)
    residual = da - trend_vals

    return (residual, trend.polyfit_coefficients)

def gaussian_rank_transform(resid: xr.DataArray) -> tuple[xr.DataArray, np.ndarray]:
    """
    Apply Gaussian rank transform to a 1D numpy array.
    """
    
    x = resid.values
    N = x.size

    ranks = np.argsort(np.argsort(x)) + 1
    u = (ranks - 0.5) / N  # (0,1)

    g = norm.ppf(u)  # ~ N(0,1)

    g_da = xr.DataArray(g, coords=resid.coords, dims=resid.dims, name="g_resid")
    return g_da, u

def build_empirical_inv_cdf(resid: xr.DataArray):
    """
    Build an inverse CDF function from residual SMB values.
    """
    x_sorted = np.sort(resid.values)
    n = x_sorted.size
    p = (np.arange(1, n+1) - 0.5) / n  # (0,1)

    def inv_cdf(u: np.ndarray) -> np.ndarray:
        return np.interp(u, p, x_sorted)

    return inv_cdf

def estimate_psd(da: xr.DataArray, dt_years: float = 1 / 12, method: str = "welch") -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the power spectral density (PSD) of the time series using specified method.
    Returns frequencies and PSD values.
    """
    from scipy.signal import welch

    x = da.values
    fs = 1.0 / dt_years  # Sampling frequency in 1/years

    # nperseg: choose something like 16 for ~50 years
    freqs, psd = welch(x, fs=fs, nperseg=min(60, len(x)))
    return freqs, psd

def period_to_freq_bounds(pmin: float, pmax: float) -> tuple[float, float]:
    fmax = 1.0 / pmin
    fmin = 1.0 / pmax
    return fmin, fmax

def scale_psd_bands(freqs: np.ndarray, psd: np.ndarray, factors: list[tuple[float,float,float]]):
    """
    factors: list of (pmin, pmax, factor), periods in years.
    """
    psd_scaled = psd.copy()
    for pmin, pmax, factor in factors:
        fmin, fmax = period_to_freq_bounds(pmin, pmax)
        mask = (freqs >= fmin) & (freqs <= fmax)
        psd_scaled[mask] *= factor
    return psd_scaled

def synthetic_freq_grid(N_syn: int, dt_years: float = 1 / 12) -> np.ndarray:
    return np.fft.rfftfreq(N_syn, d=dt_years)

def interpolate_psd_to_grid(freqs_obs: np.ndarray, psd_obs: np.ndarray, freqs_syn: np.ndarray) -> np.ndarray:
    """
    Interpolate observed PSD onto the synthetic frequency grid.
    """
    psd_syn = np.interp(freqs_syn, freqs_obs, psd_obs)
    return psd_syn

def simulate_gaussian_from_psd(freqs_syn: np.ndarray, psd_syn: np.ndarray, N_syn: int, dt_years: float = 1 / 12, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    n_freqs = freqs_syn.size
    # Simple scaling: amplitude ~ sqrt(psd)*sqrt(N_syn)

    df = (freqs_syn[1] - freqs_syn[0]) * N_syn
    amps = np.sqrt(psd_syn * df) * np.sqrt(N_syn) / np.sqrt(2)  # factor of sqrt(2) for rms

    phases = rng.uniform(-np.pi, np.pi, size=n_freqs)
    phases[0] = 0.0
    if N_syn % 2 == 0:
        phases[-1] = 0.0

    Z = amps * np.exp(1j * phases)
    g_syn = np.fft.irfft(Z, n=N_syn)

    return g_syn

def inverse_gaussianize_to_resid(g_syn: np.ndarray, inv_cdf) -> np.ndarray:
    """
    Map synthetic Gaussian series back to SMB residual units
    using empirical inverse CDF of residuals.
    """
    # Gaussian -> Uniform(0,1)
    u_syn = norm.cdf(g_syn)

    # Uniform -> SMB residual via empirical inverse CDF
    r_syn = inv_cdf(u_syn)

    return r_syn

def reassemble_smb(r_syn: np.ndarray, smb_mean: float, trend=None) -> np.ndarray:
    """
    Reassemble full SMB from synthetic residuals.
    If trend is None, assume no trend.
    """
    if trend is None:
        trend_component = 0.0
    else:
        # For 1000 years we might want a new trend definition,
        # not just extrapolated from the short record.
        t_syn = np.arange(len(r_syn))
        trend_component = trend[0] + trend[1] * t_syn  # if trend is [a, b]

    smb_syn = r_syn + smb_mean + trend_component
    return smb_syn

def calc_moments(da: xr.DataArray) -> dict[str, float]:
    """
    Calculate mean and standard deviation of the DataArray.
    """
    import scipy.stats as stats
    moments = {
        'mean': da.mean().item(),
        'std': da.std().item(),
        "skew": stats.skew(da.values), # type: ignore
        "kurtosis": stats.kurtosis(da.values)
        }

    return moments

def resample_to_yearly(da: xr.DataArray) -> xr.DataArray:
    """
    Resample DataArray to yearly frequency using mean. 
    Assumes 'time' coordinate is datetime-like.
    """
    try:
        da_yearly = da.resample(time='YS').sum()
    except Exception as e:
        da_yearly = da.resample(time_syn='YS').sum()
        
    return da_yearly

def _generate_bandpass(da: xr.DataArray, pmin: float, pmax: float, dt_years: float = 1 / 12) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate bandpass frequencies for given period bounds.
    """
    fmin, fmax = period_to_freq_bounds(pmin, pmax)
    # N = 1000  # Assuming a fixed N_syn for this function
    N = len(da)
    freqs_syn = synthetic_freq_grid(N, dt_years=dt_years)
    band_mask = (freqs_syn >= fmin) & (freqs_syn <= fmax)
    band_freqs = freqs_syn[band_mask]
    return band_freqs, band_mask

def apply_bandpass_scaling(psd_syn: np.ndarray, band_mask: np.ndarray, factor: float) -> np.ndarray:
    """
    Apply scaling factor to PSD within the band defined by band_mask.
    """

    psd_scaled = psd_syn.copy()
    psd_scaled[band_mask] *= factor
    return psd_scaled