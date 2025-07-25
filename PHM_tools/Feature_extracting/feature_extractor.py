from __future__ import annotations

"""Feature extraction functions for the PHM agent."""

from typing import Dict

import numpy as np
import scipy.stats
from smolagents import tool

from utils import ensure_3d
from utils.registry import register_tool


@register_tool("extract_time_features")
@tool
def extract_time_features(signal: list | np.ndarray) -> Dict[str, np.ndarray]:
    """Extract common time-domain statistics.

    Args:
        signal: Input vibration signal.

    Returns:
        Mapping from feature name to arrays of shape ``(B, C)``. The keys
        include ``Mean``, ``Std``, ``Var``, ``Entropy``, ``Max``, ``Min``,
        ``AbsMean``, ``Kurtosis``, ``RMS``, ``Median``, ``MAD``,
        ``CrestFactor``, ``ClearanceFactor``, ``Skewness``, ``ShapeFactor``,
        ``CrestFactorDelta``, ``PeakToPeak``, ``ImpulseFactor``, and
        ``MarginFactor``.
    """
    x = ensure_3d(np.asarray(signal))
    if x.size == 0:
        return {}

    B, L, C = x.shape

    mu = x.mean(axis=1)
    std = x.std(axis=1)
    var = x.var(axis=1)
    abs_x = np.abs(x)
    abs_mean = abs_x.mean(axis=1)
    rms = np.sqrt(np.mean(x**2, axis=1))
    median = np.median(x, axis=1)
    mad = np.median(np.abs(x - median[:, None, :]), axis=1)
    diff = np.diff(x, axis=1)

    hist = np.histogram(x, bins=10, axis=1)[0]
    entropy = scipy.stats.entropy(np.abs(hist), axis=1)
    kurtosis = scipy.stats.kurtosis(x, axis=1)
    skewness = scipy.stats.skew(x, axis=1)
    max_val = x.max(axis=1)
    min_val = x.min(axis=1)
    crest_factor_delta = np.sqrt(np.mean(diff**2, axis=1))

    rms_safe = np.where(rms > 0, rms, 1e-9)
    abs_mean_safe = np.where(abs_mean > 0, abs_mean, 1e-9)

    features = {
        "Mean": mu,
        "Std": std,
        "Var": var,
        "Entropy": entropy,
        "Max": max_val,
        "Min": min_val,
        "AbsMean": abs_mean,
        "Kurtosis": kurtosis,
        "RMS": rms,
        "Median": median,
        "MAD": mad,
        "CrestFactor": np.max(abs_x, axis=1) / rms_safe,
        "ClearanceFactor": np.max(abs_x, axis=1) / abs_mean_safe,
        "Skewness": skewness,
        "ShapeFactor": rms / abs_mean_safe,
        "CrestFactorDelta": crest_factor_delta / abs_mean_safe,
        "PeakToPeak": max_val - min_val,
        "ImpulseFactor": np.max(abs_x, axis=1) / abs_mean_safe,
        "MarginFactor": np.max(abs_x, axis=1)
        / (np.mean(np.sqrt(abs_x), axis=1) ** 2 + 1e-9),
    }

    out = {k: v.reshape(B, C) for k, v in features.items()}
    for k, v in out.items():
        v[np.isnan(v) | np.isinf(v)] = 0.0
    return out


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    sample = rng.normal(size=1024)
    time_res = extract_time_features(sample)
    freq_res = extract_frequency_features(sample, fs=10_000)
    print("Time feature sample keys:", list(time_res)[:3])
    print("Frequency feature sample keys:", list(freq_res)[:3])


@register_tool("extract_frequency_features")
@tool
def extract_frequency_features(signal: list | np.ndarray, fs: float = 1.0) -> Dict[str, np.ndarray]:
    """Compute various frequency-domain metrics.

    Args:
        signal: Input vibration signal.
        fs: Sampling frequency.

    Returns:
        Mapping from feature name to arrays of shape ``(B, C)``. The keys
        include ``EnvelopeMean``, ``EnvelopeStd``, ``TKE``, ``SpectralCentroid``,
        ``SpectralRolloff``, ``SpectralFlatness``, ``SpectralEntropy``, and
        ``SpectralSlope``.
    """
    x = ensure_3d(np.asarray(signal))
    if x.size == 0:
        return {}

    B, L, C = x.shape

    analytic = scipy.signal.hilbert(x, axis=1)
    envelope = np.abs(analytic)
    if L > 2:
        tke = np.mean(x[:, 1:-1, :] ** 2 - x[:, :-2, :] * x[:, 2:, :], axis=1)
    else:
        tke = np.zeros((B, C))

    f, Pxx = scipy.signal.periodogram(x, fs=fs, axis=1)
    psd_sum = np.sum(Pxx, axis=1) + 1e-12
    spectral_centroid = np.sum(Pxx * f[None, :, None], axis=1) / psd_sum
    cumulative = np.cumsum(Pxx, axis=1)
    idx = np.argmax(cumulative >= 0.85 * psd_sum[:, None, :], axis=1)
    spectral_rolloff = f[idx]
    spectral_flatness = scipy.stats.gmean(Pxx + 1e-12, axis=1) / (np.mean(Pxx, axis=1) + 1e-12)
    psd_norm = Pxx / psd_sum[:, None, :]
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=1)
    mean_f = f.mean()
    var_f = np.mean((f - mean_f) ** 2)
    mean_psd = np.mean(Pxx, axis=1, keepdims=True)
    spectral_slope = np.sum((f - mean_f) * (Pxx - mean_psd), axis=1) / (Pxx.shape[1] * var_f)

    features = {
        "EnvelopeMean": envelope.mean(axis=1),
        "EnvelopeStd": envelope.std(axis=1),
        "TKE": tke,
        "SpectralCentroid": spectral_centroid,
        "SpectralRolloff": spectral_rolloff,
        "SpectralFlatness": spectral_flatness,
        "SpectralEntropy": spectral_entropy,
        "SpectralSlope": spectral_slope,
    }

    out = {k: v.reshape(B, C) for k, v in features.items()}
    for k, v in out.items():
        v[np.isnan(v) | np.isinf(v)] = 0.0
    return out

